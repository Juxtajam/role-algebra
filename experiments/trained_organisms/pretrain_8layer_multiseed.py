"""Phase 7 Item 1 step 1 — 8-LAYER induction
pretraining, seeds 3-7 (extension arm). pretrain() is byte-identical to
phase6_pretrain8.py; only the app name and the entrypoint seed list differ.
Seeds failing the battery are reported and REPLACED (next unused seed id)
until five verified seeds have run.

The composition experiment, one variable: depth 8 instead of 4. Everything
else verbatim from Phase 5 Step 2 (phase5_pretrain.py): the v2 induction
corpus (src/trained/induction.py — unique random prefix L in [16,32]
over the full task vocab, tiled cyclically to 64, variable offset =>
content matching required), batch 256, AdamW lr 1e-3 wd 0.01 with name
rows in the explicit weight_decay=0.0 parameter group (Phase 4b freeze fix
retained), cosine over hard 50k with 500 warmup, full-sequence LM loss,
eval every 500, fp32, fresh batch per step.

Verification battery per seed (Item B: "same verification
battery"; seeds failing do not advance):
  1. behavioural: held-out copy accuracy > 0.9;
  2. mechanistic: some head >= 0.25 mass AND >= 5x uniform baseline on the
     prev-token-successor edge (Check-3 thresholds);
  3. OFF-PERIOD CONTENT PROBE (part of the battery per the Phase 6
     spec; in Phase 5 it was run beyond the mandated checks on seed 0
     and passed at 0.815-0.942 mass): at each probe period p in {11, 13,
     40} — all outside the training range [16, 32] — the induction edge
     must persist: max head mean mass from t to t-p+1 (probe positions
     t in [p, min(2p-1, 62)], unambiguous single prior occurrence)
     >= 0.25 AND >= 5x that period's uniform baseline. Rule declared here,
     in advance; the same mechanistic thresholds applied off-distribution.
     Behavioural copy accuracy at off periods is reported but NOT gating
     (Phase 5 precedent: readout degrades away from the training range
     while the edge persists; the circuit being verified is the edge).

Runs keyed stage2/induction8/pretrain/seed{s} on dv3-results.
"""

import modal
import pathlib

app = modal.App("dv3-phase7-pretrain8")
vol = modal.Volume.from_name("dv3-results")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.4.1", "numpy", "scipy", "pandas", "matplotlib")
    .add_local_dir(
        str(pathlib.Path(__file__).resolve().parents[2] / "src"),
        remote_path="/root/dv3",
    )
)

MAX_STEPS = 50_000
N_LAYERS = 8
OFF_PERIODS = (11, 13, 40)


@app.function(image=image, gpu="A10G", timeout=60 * 60 * 8, volumes={"/results": vol})
def pretrain(seed: int):
    import math, os, sys, threading, time, traceback

    os.environ["DV3_RESULTS"] = "/results"
    sys.path.insert(0, "/root/dv3")
    os.chdir("/root/dv3")

    stop_evt = threading.Event()

    def committer():
        while not stop_evt.is_set():
            time.sleep(60)
            try:
                vol.commit()
            except Exception:
                pass

    threading.Thread(target=committer, daemon=True).start()

    import numpy as np
    import torch
    from shared import progress
    from shared.progress import log
    from trained import induction as I
    from trained.data import VOCAB
    from trained.model import (
        TinyTransformer,
        lm_loss,
        make_optimizer,
        verify_tied_names,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    key = f"induction8/pretrain/seed{seed}"
    ckpt_path = progress.results_dir() / f"stage2/{key}/ckpt.pt"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    BATCH, EVAL_EVERY, WARMUP = 256, 500, 500

    def lr_at(step):
        if step < WARMUP:
            return 1e-3 * (step + 1) / WARMUP
        t = (step - WARMUP) / max(1, MAX_STEPS - WARMUP)
        return 1e-3 * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * min(t, 1.0))))

    torch.manual_seed(seed)
    model = TinyTransformer(seed=seed, n_layers=N_LAYERS).to(device)
    rt = verify_tied_names(model)
    log(f"phase6 pretrain8 {key}: tied-name round-trip verified ({rt:.3f})")
    opt = make_optimizer(model, lr=1e-3, weight_decay=0.01)
    assert [pg["weight_decay"] for pg in opt.param_groups] == [0.01, 0.0]
    log(
        f"phase6 pretrain8 {key}: START — device={device}, {N_LAYERS} LAYERS "
        f"(the one Item B variable), induction corpus v2 (unique prefix L in "
        f"[{I.L_MIN},{I.L_MAX}], variable offset), batch {BATCH}, AdamW lr 1e-3 "
        f"wd 0.01 (name rows wd=0.0 group, frozen), cosine/{MAX_STEPS} warmup "
        f"{WARMUP}, hard {MAX_STEPS}"
    )

    rng = np.random.default_rng((seed, 77))
    ev_seqs = I.build_eval_set(seed)  # held-out, disjoint rng stream
    frozen_init = model.emb_names.detach().clone()

    # ---- off-period probe machinery (rule in module docstring) -------------
    def periodic_eval_set(period, n=512, probe_seed_tag=55703):
        prng = np.random.default_rng((seed, probe_seed_tag + period))
        r = prng.random((n, VOCAB - 1))
        prefixes = np.argsort(r, axis=1)[:, :period] + 1
        idx = np.arange(I.IND_LEN)[None, :] % period
        seqs = np.take_along_axis(prefixes, idx, axis=1).astype(np.int64)
        Ls = np.full(n, period, dtype=np.int64)
        return dict(seqs=seqs, Ls=Ls, probes=I.probe_index_arrays(Ls))

    def off_period_probe(model):
        rows = {}
        for p_ in OFF_PERIODS:
            ev = periodic_eval_set(p_)
            base = float(
                np.mean(
                    [
                        1.0 / (t + 1.0)
                        for t in range(p_, min(2 * p_ - 1, I.IND_LEN - 2) + 1)
                    ]
                )
            )
            tab = I.induction_attention(model, ev, device, limit=256)
            acc = I.copy_accuracy(model, ev, device)
            mx = float(tab.max())
            li, h = np.unravel_index(tab.argmax(), tab.shape)
            ok = bool(mx >= I.EDGE_ABS and mx >= I.EDGE_REL * base)
            rows[p_] = dict(
                period=p_,
                copy_acc=float(acc),
                max_mass=mx,
                max_head=[int(li), int(h)],
                baseline=base,
                x_baseline=mx / base,
                edge_pass=ok,
            )
        return rows

    step, history = 0, []
    if ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        step, history = ck["step"], ck["history"]
        log(f"phase6 pretrain8 {key}: RESUMED at step {step}")

    t0 = time.time()
    model.train()
    try:
        while step < MAX_STEPS:
            seqs, _ = I.sample_batch(rng, BATCH)
            toks = torch.as_tensor(seqs, device=device)
            for pg in opt.param_groups:
                pg["lr"] = lr_at(step)
            loss = lm_loss(model(toks), toks)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            step += 1

            if step % EVAL_EVERY == 0 or step == MAX_STEPS:
                acc = I.copy_accuracy(model, ev_seqs, device)
                tab = I.induction_attention(model, ev_seqs, device, limit=256)
                li, h = np.unravel_index(tab.argmax(), tab.shape)
                mx = float(tab.max())
                with torch.no_grad():
                    frozen_ok = bool(torch.equal(model.emb_names.detach(), frozen_init))
                model.train()
                rec = dict(
                    step=step,
                    loss=float(loss.item()),
                    copy_acc=acc,
                    max_ind_mass=mx,
                    max_ind_head=[int(li), int(h)],
                    x_baseline=float(mx / I.UNIFORM_BASELINE),
                    frozen_rows_bit_exact=frozen_ok,
                )
                history.append(rec)
                rate = step / max(time.time() - t0, 1e-9)
                log(
                    f"phase6 pretrain8 {key}: step {step} loss {rec['loss']:.4f} "
                    f"copy_acc {acc:.3f} max_ind_mass {mx:.3f} "
                    f"(L{li} h{h}, {rec['x_baseline']:.1f}x baseline) "
                    f"frozen_exact={frozen_ok} ({rate:.1f} steps/s)"
                )
                if not frozen_ok:
                    log(
                        f"MILESTONE: phase6 pretrain8 {key} FREEZE VIOLATION at "
                        f"step {step} — run invalid"
                    )
                    raise RuntimeError("freeze violation")
                torch.save(
                    dict(
                        model=model.state_dict(),
                        opt=opt.state_dict(),
                        step=step,
                        history=history,
                    ),
                    ckpt_path,
                )
                progress.save_json(f"stage2/{key}/trajectory.json", history)

        # ---- Item B verification battery (all three checks gate) -----------
        ver = I.verify_induction(model, ev_seqs, device)
        off = off_period_probe(model)
        off_pass = all(r["edge_pass"] for r in off.values())
        ver.update(
            seed=seed,
            steps=step,
            final_loss=history[-1]["loss"],
            n_layers=N_LAYERS,
            off_period_probe=off,
            off_period_pass=off_pass,
        )
        ver["pass"] = bool(ver["pass"] and off_pass)
        progress.save_json(f"stage2/{key}/pretrain_verify.json", ver)
        off_str = "; ".join(
            f"p{p_}: mass {r['max_mass']:.3f} ({r['x_baseline']:.1f}x) copy "
            f"{r['copy_acc']:.2f} {'PASS' if r['edge_pass'] else 'FAIL'}"
            for p_, r in off.items()
        )
        log(
            f"MILESTONE: phase6 pretrain8 seed{seed} DONE at {step} — "
            f"copy_acc {ver['copy_acc']:.3f} "
            f"(behavioural {'PASS' if ver['behavioural_pass'] else 'FAIL'}), "
            f"max mass {ver['max_mass']:.3f} at L{ver['max_head'][0]} "
            f"h{ver['max_head'][1]} ({ver['x_baseline']:.1f}x; "
            f"mechanistic {'PASS' if ver['mechanistic_pass'] else 'FAIL'}) | "
            f"off-period [{off_str}] "
            f"({'PASS' if off_pass else 'FAIL'}) "
            f"-> seed {'ADVANCES' if ver['pass'] else 'DOES NOT ADVANCE'}"
        )
        return ver
    except Exception:
        log(
            f"MILESTONE: PHASE6 PRETRAIN8 seed{seed} CRASHED:\n"
            + traceback.format_exc()
        )
        raise
    finally:
        stop_evt.set()
        vol.commit()


@app.local_entrypoint()
def main(seeds: str = "3,4,5,6,7"):
    import json

    seed_list = [int(s) for s in seeds.split(",")]
    results = list(pretrain.map(seed_list))
    print(json.dumps(results, indent=2, default=str))

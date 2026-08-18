"""Phase 5 Step 2 — induction pretraining, 3 seeds.

Pretrain the standard T1 architecture (TinyTransformer, 4 layers, d_model
128, 4 heads — the exact class every prior run used, name-row wd exclusion
and frozen-name gradient mask retained as standing) on the Step 1 synthetic
induction corpus v2 (src/trained/induction.py): per sequence a unique
random prefix of L in [16, 32] tokens over the full task vocabulary (PAD
excluded — LM-loss ignore_index), tiled cyclically to the 64-token window,
so the repeat offset varies and copying requires content matching. The v1
fixed-offset corpus was voided (positional-shortcut probe; declared
deviation — see module docstring). No task episodes.

Hyperparameters: EXISTING, verbatim — batch 256, AdamW lr 1e-3, wd 0.01
with name rows in the explicit weight_decay=0.0 parameter group (Phase 4b
freeze fix, retained), cosine over hard 50k steps with 500 warmup,
full-sequence LM loss, eval every 500, fp32. The corpus is infinite
(fresh batch each step) — the task's pool/refresh machinery is
task-specific and does not apply to pretraining.

Per-eval logging: LM loss, held-out copy accuracy, max induction-head
mass (+ location). Final per-seed verification (both mandatory, thresholds
pre-declared): behavioural copy acc > 0.9; some head >= 0.25 mass and
>= 5x uniform baseline on the prev-token-successor edge. A seed failing
either check does not advance to Step 3. If all 3 fail: STOP, report.

Runs keyed stage2/induction/pretrain/seed{s} on dv3-results.
"""

import modal
import pathlib

app = modal.App("dv3-phase5-pretrain")
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
    from trained.model import (
        TinyTransformer,
        lm_loss,
        make_optimizer,
        verify_tied_names,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    key = f"induction/pretrain/seed{seed}"
    ckpt_path = progress.results_dir() / f"stage2/{key}/ckpt.pt"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    BATCH, EVAL_EVERY, WARMUP = 256, 500, 500

    def lr_at(step):
        if step < WARMUP:
            return 1e-3 * (step + 1) / WARMUP
        t = (step - WARMUP) / max(1, MAX_STEPS - WARMUP)
        return 1e-3 * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * min(t, 1.0))))

    torch.manual_seed(seed)
    model = TinyTransformer(seed=seed, n_layers=4).to(device)
    rt = verify_tied_names(model)
    log(f"phase5 pretrain {key}: tied-name round-trip verified ({rt:.3f})")
    opt = make_optimizer(model, lr=1e-3, weight_decay=0.01)
    assert [pg["weight_decay"] for pg in opt.param_groups] == [0.01, 0.0]
    log(
        f"phase5 pretrain {key}: START — device={device}, induction corpus "
        f"v2 (unique prefix L in [{I.L_MIN},{I.L_MAX}] over full vocab, tiled "
        f"cyclically to 64 — variable offset, content matching required), "
        f"batch {BATCH}, AdamW lr 1e-3 wd 0.01 (name rows in explicit wd=0.0 "
        f"group, frozen — standing protocol), cosine/{MAX_STEPS} warmup {WARMUP}, "
        f"hard {MAX_STEPS}"
    )

    rng = np.random.default_rng((seed, 77))
    ev_seqs = I.build_eval_set(seed)  # held-out, disjoint rng stream
    frozen_init = model.emb_names.detach().clone()

    step, history = 0, []
    if ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        step, history = ck["step"], ck["history"]
        log(f"phase5 pretrain {key}: RESUMED at step {step}")

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
                    f"phase5 pretrain {key}: step {step} loss {rec['loss']:.4f} "
                    f"copy_acc {acc:.3f} max_ind_mass {mx:.3f} "
                    f"(L{li} h{h}, {rec['x_baseline']:.1f}x baseline) "
                    f"frozen_exact={frozen_ok} ({rate:.1f} steps/s)"
                )
                if not frozen_ok:
                    log(
                        f"MILESTONE: phase5 pretrain {key} FREEZE VIOLATION at "
                        f"step {step} — frozen name rows not bit-exact; run invalid"
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

        # ---- Step 2 verification (both checks mandatory) ----
        ver = I.verify_induction(model, ev_seqs, device)
        ver.update(seed=seed, steps=step, final_loss=history[-1]["loss"])
        progress.save_json(f"stage2/{key}/pretrain_verify.json", ver)
        log(
            f"MILESTONE: phase5 pretrain seed{seed} DONE at {step} — "
            f"copy_acc {ver['copy_acc']:.3f} "
            f"(behavioural {'PASS' if ver['behavioural_pass'] else 'FAIL'}), "
            f"max mass {ver['max_mass']:.3f} at L{ver['max_head'][0]} "
            f"h{ver['max_head'][1]} ({ver['x_baseline']:.1f}x baseline; "
            f"mechanistic {'PASS' if ver['mechanistic_pass'] else 'FAIL'}) "
            f"-> seed {'ADVANCES' if ver['pass'] else 'DOES NOT ADVANCE'}"
        )
        return ver
    except Exception:
        log(
            f"MILESTONE: PHASE5 PRETRAIN seed{seed} CRASHED:\n" + traceback.format_exc()
        )
        raise
    finally:
        stop_evt.set()
        vol.commit()


@app.local_entrypoint()
def main():
    import json

    results = list(pretrain.map([0, 1, 2]))
    print(json.dumps(results, indent=2, default=str))

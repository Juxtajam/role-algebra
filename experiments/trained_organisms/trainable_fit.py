"""Item 3 (Phase 4b spec) — trainable fit-pool embeddings. The intervention.

T1, 4 layers, 3 seeds, hard 50k, ALL else unchanged (batch 256, AdamW lr 1e-3
wd 0.01, cosine over 50k with 500 warmup, full-sequence LM loss, candidate-
masked eval every 500, POOL_SIZE 8192, POOL_REFRESH 2500 — refresh restored
to the original protocol), with ONE change: name embeddings for the FIT pool
(T1: name ids 0..23, token rows NAME0..NAME0+23) are made TRAINABLE while
transfer-pool embeddings (and all other name rows outside the fit pool) stay
frozen and tied. Asymmetric by design (Item 3).

PHASE 4b: uses the corrected freeze (model.py emb_main/emb_names split +
make_optimizer putting ALL name rows in an explicit weight_decay=0.0 AdamW
group), verified in checks/item3_freeze_verify.txt — frozen name rows are
bit-exact constant across training, so the trainable-vs-frozen comparison
differs in gradient flow only. The gradient mask (names_grad_mask) is opened
for fit-pool rows 0..23 of emb_names. The tie (logits use emb.T) is untouched.
This supersedes the voided, which ran the old
decay-leaky freeze.

Pre-training verifications (mandated):
  1. tied-unembedding round-trip: every TRANSFER-pool embedding passed
     through the tied unembedding recovers its own token (argmax over V);
  2. gradient flow: after one real backward pass, fit-pool name rows have
     nonzero gradient, transfer-pool and non-fit name rows have zero.
  3. (Phase 4b freeze audit, logged every eval) frozen name rows bit-exact
     vs init; fit-pool row norm drift reported.

Stop/branch (handled at reporting, not in code): >=0.95 held-out single-hop
aux -> report and stop for a decision; chance at 50k -> stop the whole line
and write up.

Runs keyed stage2/trainable_fit/T1_tfit/seed{s} on the dv3-results volume.
"""

import modal
import pathlib

app = modal.App("dv3-trainable-fit")
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
def train_tfit(seed: int):
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
    from trained import data as D
    from trained.model import (
        TinyTransformer,
        lm_loss,
        make_optimizer,
        masked_answer_preds,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    key = f"trainable_fit/T1_tfit/seed{seed}"
    ckpt_path = progress.results_dir() / f"stage2/{key}/ckpt.pt"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    BATCH, EVAL_EVERY, POOL_SIZE, POOL_REFRESH, WARMUP = 256, 500, 8192, 2500, 500
    AUX_T1 = (D.A_PS, D.A_SN, D.A_NS)
    FIT_POOL = 24  # T1 name pool: ids 0..23 -> token rows NAME0..NAME0+24

    def lr_at(step):
        if step < WARMUP:
            return 1e-3 * (step + 1) / WARMUP
        t = (step - WARMUP) / max(1, MAX_STEPS - WARMUP)
        return 1e-3 * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * min(t, 1.0))))

    torch.manual_seed(seed)
    model = TinyTransformer(seed=seed, n_layers=4).to(device)
    # THE intervention: unfreeze fit-pool name rows in the gradient mask.
    # Everything else about the model (tie, init, norms) is untouched.
    with torch.no_grad():
        model.names_grad_mask[:FIT_POOL] = 1.0
    n_open = int(model.names_grad_mask.sum().item())
    frozen_init = model.emb_names.detach()[FIT_POOL:].clone()
    fit_init_norms = model.emb_names.detach()[:FIT_POOL].norm(dim=1).clone()
    log(
        f"trainable_fit {key}: names_grad_mask opened for fit-pool name rows "
        f"{D.NAME0}..{D.NAME0 + FIT_POOL - 1} ({n_open} name rows trainable); "
        f"transfer + non-fit names stay frozen, ALL name rows in wd=0.0 group "
        f"(Phase 4b corrected freeze)"
    )

    # --- pre-training verification 1: tied round-trip for TRANSFER pool ----
    with torch.no_grad():
        tr_ids = torch.arange(
            D.NAME0 + D.N_FIT, D.NAME0 + D.N_FIT + D.N_TRANSFER, device=device
        )
        rt_logits = model.emb[tr_ids] @ model.emb.T
        rt_ok = float((rt_logits.argmax(dim=1) == tr_ids).float().mean().item())
    assert rt_ok == 1.0, f"transfer-pool tied round-trip FAILED: {rt_ok:.4f}"
    log(
        f"trainable_fit {key}: VERIFIED transfer-pool tied round-trip "
        f"({D.N_TRANSFER}/{D.N_TRANSFER} tokens recovered, acc {rt_ok:.3f})"
    )

    # --- pre-training verification 2: fit-pool gradients actually flow -----
    rng_v = np.random.default_rng((seed, 12345))
    vb = [D.sample_base("T1", rng_v) for _ in range(64)]
    gs_v = [D.PERMS[i] for i in rng_v.integers(0, len(D.PERMS), size=64)]
    toks_v, _, _, _ = D.render_batch(vb, gs_v)
    toks_v = torch.as_tensor(toks_v, device=device)
    used = sorted(set(toks_v.flatten().tolist()))
    used_fit = [t for t in used if D.NAME0 <= t < D.NAME0 + FIT_POOL]
    loss_v = lm_loss(model(toks_v), toks_v)
    loss_v.backward()
    g = model.emb_names.grad  # name-row grads live on the emb_names parameter
    used_fit_rows = [t - D.NAME0 for t in used_fit]
    fit_gnorms = g[used_fit_rows].norm(dim=1)
    frozen_gnorm = g[FIT_POOL:].abs().max().item()
    assert len(used_fit) > 0 and bool(
        (fit_gnorms > 0).all()
    ), f"fit-pool gradients NOT flowing: {len(used_fit)} rows used, norms {fit_gnorms}"
    assert (
        frozen_gnorm == 0.0
    ), f"frozen name rows received gradient: max {frozen_gnorm}"
    log(
        f"trainable_fit {key}: VERIFIED fit-pool gradient flow — "
        f"{len(used_fit)} fit-name rows in probe batch, all grad norms > 0 "
        f"(min {fit_gnorms.min().item():.3e}, max {fit_gnorms.max().item():.3e}); "
        f"max |grad| on frozen name rows = {frozen_gnorm:.1e} (exactly 0)"
    )
    model.zero_grad(set_to_none=True)

    opt = make_optimizer(model, lr=1e-3, weight_decay=0.01)
    assert [pg["weight_decay"] for pg in opt.param_groups] == [0.01, 0.0]
    log(
        f"trainable_fit {key}: START — device={device}, wd=0.01 (name rows in "
        f"explicit wd=0.0 group), 4 layers, hard {MAX_STEPS}, "
        f"pool refresh every {POOL_REFRESH} (original protocol), fit-pool embeddings TRAINABLE"
    )

    rng = np.random.default_rng((seed, 77))
    eval_rng = np.random.default_rng((seed, 88))
    pool = [D.sample_base("T1", rng) for _ in range(POOL_SIZE)]

    ev_aux = {
        qt: D.build_eval_orbits("T1", "fit", 64, seed=9600 + 31 * qt + seed, qtok=qt)
        for qt in AUX_T1
    }
    ev_comp = D.build_eval_orbits("T1", "fit", 192, seed=9000 + seed)
    ev_transfer = D.build_eval_orbits("T1", "transfer", 96, seed=9500 + seed)

    def acc(ev):
        preds = masked_answer_preds(
            model, ev["tokens"], ev["answer_pos"], ev["candidates"]
        )
        return float((preds == ev["answers"]).mean())

    def train_acc(qtoks, n=96):
        bs = [b for b in pool if b.qtok in qtoks][:n]
        if not bs:
            return None
        gs = [D.PERMS[i] for i in eval_rng.integers(0, len(D.PERMS), size=len(bs))]
        toks, apos, ans, cands = D.render_batch(bs, gs)
        preds = masked_answer_preds(model, toks, apos, cands)
        return float((preds == ans).mean())

    step, history = 0, []
    if ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        step, history = ck["step"], ck["history"]
        log(f"trainable_fit {key}: RESUMED at step {step}")

    t0 = time.time()
    model.train()
    try:
        while step < MAX_STEPS:
            if step and step % POOL_REFRESH == 0:
                pool = [D.sample_base("T1", rng) for _ in range(POOL_SIZE)]
            idx = rng.integers(0, len(pool), size=BATCH)
            gs = [D.PERMS[i] for i in rng.integers(0, len(D.PERMS), size=BATCH)]
            toks, _, _, _ = D.render_batch([pool[i] for i in idx], gs)
            toks = torch.as_tensor(toks, device=device)
            for pg in opt.param_groups:
                pg["lr"] = lr_at(step)
            loss = lm_loss(model(toks), toks)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            step += 1

            if step % EVAL_EVERY == 0 or step == MAX_STEPS:
                model.eval()
                aux_held = {D.token_name(qt): acc(ev_aux[qt]) for qt in AUX_T1}
                aux_min = min(aux_held.values())
                comp_held = acc(ev_comp)
                tr_aux = train_acc(AUX_T1)
                tr_comp = train_acc((D.Q_P,))
                model.train()
                # Phase 4b freeze audit: frozen name rows bit-exact vs init;
                # fit-pool row norm drift (they SHOULD move — they train)
                with torch.no_grad():
                    frozen_ok = bool(
                        torch.equal(model.emb_names.detach()[FIT_POOL:], frozen_init)
                    )
                    fit_ratio = (
                        model.emb_names.detach()[:FIT_POOL].norm(dim=1) / fit_init_norms
                    )
                rec = dict(
                    step=step,
                    loss=float(loss.item()),
                    aux_held=aux_held,
                    aux_min=aux_min,
                    train_aux=tr_aux,
                    train_composed=tr_comp,
                    held_episode=comp_held,
                    transfer_episode=acc(ev_transfer),
                    frozen_rows_bit_exact=frozen_ok,
                    fit_norm_ratio_mean=float(fit_ratio.mean()),
                    fit_norm_ratio_min=float(fit_ratio.min()),
                    fit_norm_ratio_max=float(fit_ratio.max()),
                )
                history.append(rec)
                rate = step / max(time.time() - t0, 1e-9)
                log(
                    f"trainable_fit {key}: step {step} loss {rec['loss']:.3f} "
                    f"aux_held_min {aux_min:.3f} ({aux_held}) "
                    f"train_aux {tr_aux if tr_aux is None else round(tr_aux, 3)} "
                    f"train_composed {tr_comp if tr_comp is None else round(tr_comp, 3)} "
                    f"composed_held {comp_held:.3f} transfer {rec['transfer_episode']:.3f} "
                    f"frozen_exact={frozen_ok} fit_norm_ratio={fit_ratio.mean():.3f} "
                    f"({rate:.1f} steps/s)"
                )
                if not frozen_ok:
                    log(
                        f"MILESTONE: trainable_fit {key} FREEZE VIOLATION at step {step} "
                        f"— frozen name rows not bit-exact; run invalid"
                    )
                    raise RuntimeError("freeze violation")
                if aux_min >= 0.95:
                    log(
                        f"MILESTONE: trainable_fit {key} BRANCH CONDITION — aux held-out min "
                        f"{aux_min:.3f} >= 0.95 at step {step}"
                    )
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

        final = history[-1]
        max_aux_held = max(h["aux_min"] for h in history)
        max_any_aux = max(max(h["aux_held"].values()) for h in history)
        result = dict(
            seed=seed,
            final=final,
            max_aux_held_min=max_aux_held,
            max_aux_held_any=max_any_aux,
            max_train_aux=max(
                h["train_aux"] for h in history if h["train_aux"] is not None
            ),
        )
        progress.save_json(f"stage2/{key}/tfit_result.json", result)
        log(
            f"MILESTONE: trainable_fit seed{seed} DONE at {MAX_STEPS} — "
            f"final aux_held_min {final['aux_min']:.3f} (max-min ever {max_aux_held:.3f}, "
            f"max-any ever {max_any_aux:.3f}), final train_aux {final['train_aux']}, "
            f"final composed_held {final['held_episode']:.3f} transfer {final['transfer_episode']:.3f}"
        )
        return result
    except Exception:
        log(f"MILESTONE: TRAINABLE_FIT seed{seed} CRASHED:\n" + traceback.format_exc())
        raise
    finally:
        stop_evt.set()
        vol.commit()


@app.function(image=image, timeout=1800, volumes={"/results": vol})
def finalize():
    import os, sys

    os.environ["DV3_RESULTS"] = "/results"
    sys.path.insert(0, "/root/dv3")
    os.chdir("/root/dv3")
    from shared import progress
    from shared.progress import log

    summary = {}
    for s in (0, 1, 2):
        rel = f"stage2/trainable_fit/T1_tfit/seed{s}/tfit_result.json"
        if progress.exists(rel):
            summary[f"seed{s}"] = progress.load_json(rel)
    progress.save_json("stage2/trainable_fit_summary.json", summary)
    log(
        "MILESTONE: TRAINABLE FIT SUMMARY: "
        + "; ".join(
            f"seed{s}: aux_min_final={v['final']['aux_min']:.3f} "
            f"aux_min_max={v['max_aux_held_min']:.3f} "
            f"composed={v['final']['held_episode']:.3f} "
            f"transfer={v['final']['transfer_episode']:.3f}"
            for s, v in ((k[-1], v) for k, v in sorted(summary.items()))
        )
    )
    vol.commit()
    return summary


@app.local_entrypoint()
def main():
    results = list(train_tfit.map([0, 1, 2]))
    print("trainable_fit:", results)
    print("finalize:", finalize.remote())

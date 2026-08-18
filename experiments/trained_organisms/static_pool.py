"""Item 2 (Phase 4 spec) — static-pool control.

T1, 4 layers, 3 seeds, hard 50k steps, ALL hyperparameters unchanged
(batch 256, AdamW lr 1e-3 wd 0.01, cosine over 50k with 500 warmup,
full-sequence LM loss, candidate-masked eval every 500, POOL_SIZE 8192,
normal T1 mixture ~40% aux), with ONE fixed pool for the entire run —
the pool of 8192 base episodes over the fixed 24-name T1 pool is sampled
once at step 0 and NEVER refreshed.

This tests the memorise-then-regress hypothesis: under refresh every 2500
steps, train_aux rose to ~0.82 then regressed as the pool was replaced.
If held-out aux generalisation holds rather than regressing here, the
refresh dynamic is implicated. If it plateaus in the same place, it is not.

Aux train/held-out trajectories are logged at the same resolution as the
curriculum run (every 500 steps: per-aux-type held accuracy, min, train_aux,
train_composed, composed held, transfer).

Runs keyed stage2/static_pool/T1_static/seed{s} on the dv3-results volume.
"""

import modal
import pathlib

app = modal.App("dv3-static-pool")
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
def train_static(seed: int):
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
        masked_answer_preds,
        verify_tied_names,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    key = f"static_pool/T1_static/seed{seed}"
    ckpt_path = progress.results_dir() / f"stage2/{key}/ckpt.pt"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    BATCH, EVAL_EVERY, POOL_SIZE, WARMUP = 256, 500, 8192, 500
    AUX_T1 = (D.A_PS, D.A_SN, D.A_NS)

    def lr_at(step):
        if step < WARMUP:
            return 1e-3 * (step + 1) / WARMUP
        t = (step - WARMUP) / max(1, MAX_STEPS - WARMUP)
        return 1e-3 * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * min(t, 1.0))))

    torch.manual_seed(seed)
    model = TinyTransformer(seed=seed, n_layers=4).to(device)
    verify_tied_names(model)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    log(
        f"static_pool {key}: START — device={device}, wd=0.01, 4 layers, hard {MAX_STEPS}, "
        f"ONE fixed pool of {POOL_SIZE} bases (normal T1 mixture), NO refresh"
    )

    rng = np.random.default_rng((seed, 77))
    eval_rng = np.random.default_rng((seed, 88))

    # THE static pool: sampled once, deterministic on resume (first rng draw)
    pool = [D.sample_base("T1", rng) for _ in range(POOL_SIZE)]

    # fixed held-out eval sets — same construction/seeds as the curriculum run
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
        log(f"static_pool {key}: RESUMED at step {step}")

    t0 = time.time()
    model.train()
    try:
        while step < MAX_STEPS:
            # NO pool refresh — the single deliberate difference from train.py
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
                rec = dict(
                    step=step,
                    loss=float(loss.item()),
                    aux_held=aux_held,
                    aux_min=aux_min,
                    train_aux=tr_aux,
                    train_composed=tr_comp,
                    held_episode=comp_held,
                    transfer_episode=acc(ev_transfer),
                )
                history.append(rec)
                rate = step / max(time.time() - t0, 1e-9)
                log(
                    f"static_pool {key}: step {step} loss {rec['loss']:.3f} "
                    f"aux_held_min {aux_min:.3f} ({aux_held}) "
                    f"train_aux {tr_aux if tr_aux is None else round(tr_aux, 3)} "
                    f"train_composed {tr_comp if tr_comp is None else round(tr_comp, 3)} "
                    f"composed_held {comp_held:.3f} ({rate:.1f} steps/s)"
                )
                if aux_min >= 0.95:
                    log(
                        f"MILESTONE: static_pool {key} aux held-out min {aux_min:.3f} >= 0.95 "
                        f"at step {step} — held-out aux generalisation under the static pool"
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
        max_tr_aux = max(h["train_aux"] for h in history if h["train_aux"] is not None)
        result = dict(
            seed=seed,
            final=final,
            max_aux_held_min=max_aux_held,
            max_train_aux=max_tr_aux,
        )
        progress.save_json(f"stage2/{key}/static_result.json", result)
        log(
            f"MILESTONE: static_pool seed{seed} DONE at {MAX_STEPS} — "
            f"final aux_held_min {final['aux_min']:.3f} (max ever {max_aux_held:.3f}), "
            f"final train_aux {final['train_aux']} (max ever {max_tr_aux:.3f}), "
            f"final composed_held {final['held_episode']:.3f}"
        )
        return result
    except Exception:
        log(f"MILESTONE: STATIC_POOL seed{seed} CRASHED:\n" + traceback.format_exc())
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
        rel = f"stage2/static_pool/T1_static/seed{s}/static_result.json"
        if progress.exists(rel):
            summary[f"seed{s}"] = progress.load_json(rel)
    progress.save_json("stage2/static_pool_summary.json", summary)
    log(
        "MILESTONE: STATIC POOL SUMMARY: "
        + "; ".join(
            f"seed{s}: aux_min_final={v['final']['aux_min']:.3f} "
            f"aux_min_max={v['max_aux_held_min']:.3f} "
            f"train_aux_max={v['max_train_aux']:.3f} "
            f"composed={v['final']['held_episode']:.3f}"
            for s, v in ((k[-1], v) for k, v in sorted(summary.items()))
        )
    )
    vol.commit()
    return summary


@app.local_entrypoint()
def main():
    results = list(train_static.map([0, 1, 2]))
    print("static_pool:", results)
    print("finalize:", finalize.remote())

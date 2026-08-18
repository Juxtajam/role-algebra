"""Phase 7 Item 1 — RECONSTRUCTION of seed 2's mid-formation checkpoint.

The Item 1 mandate requires the second-hop aggregate attention diagnostic
on the final checkpoint AND one mid-formation checkpoint for EVERY
composing seed. Seed 2 (Phase 6) composes, but phase6_finetune8.py
overwrote ckpt.pt at every eval, so no mid-formation state survives.

NOTE (reconstruction, not recovery): this script replays
seed 2's fine-tuning EXACTLY as phase6_finetune8.py ran it — same gate,
same torch.manual_seed(2), same pretrained checkpoint load, same fresh
optimizer, same rng streams ((2,77) for data, (2,88) for eval), same pool,
batch, lr schedule, refresh — and stops at step 1,500, the recorded first
eval where composed held-out crossed 0.5 (0.539 in the Phase 6 trajectory).
The training-step code path is line-identical to phase6_finetune8.py; all
per-eval instrumentation that does not consume rng (attention probes,
freeze audit) is retained, and the eval-rng-consuming train-accuracy calls
are retained too so stream positions match. GPU float nondeterminism means
the state is a reconstruction, not a bit-identical recovery; the sanity
gate below (composed held-out at step 1500 within 0.05 of the recorded
0.539) is declared HERE, before launch. Writes
stage2/induction8/finetune/seed2/ckpt_midformation.pt (reconstructed=True).
No existing artifact is modified.
"""

import modal
import pathlib

app = modal.App("dv3-item1-seed2-midformation")
vol = modal.Volume.from_name("dv3-results")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.4.1", "numpy", "scipy", "pandas", "matplotlib")
    .add_local_dir(
        str(pathlib.Path(__file__).resolve().parents[2] / "src"),
        remote_path="/root/dv3",
    )
)

STOP_STEP = 1_500
RECORDED = 0.539
TOL = 0.05
MAX_STEPS = 50_000  # lr schedule must use the ORIGINAL horizon
N_LAYERS = 8
SEED = 2


@app.function(image=image, gpu="A10G", timeout=3600, volumes={"/results": vol})
def replay():
    import math, os, sys

    os.environ["DV3_RESULTS"] = "/results"
    sys.path.insert(0, "/root/dv3")
    os.chdir("/root/dv3")
    import numpy as np
    import torch
    from shared import progress
    from shared.progress import log
    from trained import data as D
    from trained import induction as I
    from trained.model import (
        TinyTransformer,
        lm_loss,
        make_optimizer,
        masked_answer_preds,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    key = f"induction8/finetune/seed{SEED}"
    out_path = progress.results_dir() / f"stage2/{key}/ckpt_midformation.pt"

    BATCH, EVAL_EVERY, POOL_SIZE, POOL_REFRESH, WARMUP = 256, 500, 8192, 2500, 500
    AUX_T1 = (D.A_PS, D.A_SN, D.A_NS)

    def lr_at(step):
        if step < WARMUP:
            return 1e-3 * (step + 1) / WARMUP
        t = (step - WARMUP) / max(1, MAX_STEPS - WARMUP)
        return 1e-3 * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * min(t, 1.0))))

    pre_ver = progress.load_json(
        f"stage2/induction8/pretrain/seed{SEED}/pretrain_verify.json"
    )
    assert pre_ver["pass"]

    torch.manual_seed(SEED)
    model = TinyTransformer(seed=SEED, n_layers=N_LAYERS).to(device)
    pre_ck = torch.load(
        progress.results_dir() / f"stage2/induction8/pretrain/seed{SEED}/ckpt.pt",
        map_location=device,
    )
    assert pre_ck["step"] == 50_000
    model.load_state_dict(pre_ck["model"])
    opt = make_optimizer(model, lr=1e-3, weight_decay=0.01)
    assert [pg["weight_decay"] for pg in opt.param_groups] == [0.01, 0.0]

    ev_ind = I.build_eval_set(SEED)
    ver0 = I.verify_induction(model, ev_ind, device)
    assert ver0["pass"]

    rng = np.random.default_rng((SEED, 77))
    eval_rng = np.random.default_rng((SEED, 88))
    pool = [D.sample_base("T1", rng) for _ in range(POOL_SIZE)]

    ev_aux = {
        qt: D.build_eval_orbits("T1", "fit", 64, seed=9600 + 31 * qt + SEED, qtok=qt)
        for qt in AUX_T1
    }
    ev_comp = D.build_eval_orbits("T1", "fit", 192, seed=9000 + SEED)
    ev_transfer = D.build_eval_orbits("T1", "transfer", 96, seed=9500 + SEED)

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

    log(
        f"ITEM1 seed2 mid-formation REPLAY start — stop at {STOP_STEP}, "
        f"sanity gate composed in [{RECORDED - TOL:.3f}, {RECORDED + TOL:.3f}]"
    )
    step = 0
    model.train()
    comp_at_stop = None
    while step < STOP_STEP:
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
        if step % EVAL_EVERY == 0:
            model.eval()
            comp_held = acc(ev_comp)
            aux_held = {D.token_name(qt): acc(ev_aux[qt]) for qt in AUX_T1}
            _ = train_acc(AUX_T1)  # keep eval_rng stream position aligned
            _ = train_acc((D.Q_P,))
            _ = acc(ev_transfer)
            model.train()
            log(
                f"ITEM1 replay seed2: step {step} loss {loss.item():.3f} "
                f"composed_held {comp_held:.3f} aux {aux_held}"
            )
            if step == STOP_STEP:
                comp_at_stop = comp_held

    ok = abs(comp_at_stop - RECORDED) <= TOL
    log(
        f"MILESTONE: ITEM1 seed2 mid-formation replay DONE at {STOP_STEP} — "
        f"composed {comp_at_stop:.3f} vs recorded {RECORDED} "
        f"({'SANITY PASS' if ok else 'SANITY FAIL — reconstruction rejected'})"
    )
    if ok:
        import torch as T

        T.save(
            dict(
                model=model.state_dict(),
                step=step,
                composed_held=comp_at_stop,
                reconstructed=True,
                recorded_composed=RECORDED,
            ),
            out_path,
        )
        vol.commit()
    assert ok, f"reconstruction sanity gate failed: {comp_at_stop:.3f}"
    return dict(step=step, composed_held=comp_at_stop, recorded=RECORDED)


@app.local_entrypoint()
def main():
    import json

    print(json.dumps(replay.remote(), indent=2))

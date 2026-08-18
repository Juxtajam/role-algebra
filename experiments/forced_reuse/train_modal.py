"""Forced-reuse ("permutation transfer") — train a d=128 transformer from
scratch and gate it, then the discriminator adjudicates the mechanism.

Self-contained: own task (phase10/forced_reuse/task.py), own training loop,
reuses only TinyTransformer. Trained from scratch (no induction pretraining —
different task). Fit-vocab names trainable; transfer-vocab names frozen+tied
(so the discriminator's disjoint-vocab C1 is posable). 3 seeds, hard 40k.

Gate: composed transfer accuracy (candidate-masked) >= 0.5 held-out on unseen
bases; strict-orbit reported. If it forms, cache answer-position activations
(all layers) on disc orbits for the discriminator.

Launch: modal run --detach phase10/forced_reuse/train_modal.py
"""

import json
import time

import modal
import pathlib

APP = modal.App("dv3-forced-reuse-train")
vol = modal.Volume.from_name("dv3-results")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.4.1", "numpy")
    .add_local_dir(
        str(pathlib.Path(__file__).resolve().parents[2] / "src"),
        remote_path="/root/dv3",
    )
    .add_local_dir(
        str(pathlib.Path(__file__).resolve().parents[2] / "experiments/forced_reuse"),
        remote_path="/root/fr",
    )
)

MAX_STEPS = 40_000
EVAL_EVERY = 1000
BATCH = 256
POOL_SIZE = 8192
POOL_REFRESH = 2500
SEEDS = [0, 1, 2]


@APP.function(image=image, gpu="A100-80GB", timeout=6 * 3600, volumes={"/results": vol})
def train(seed: int) -> dict:
    import os, sys, math

    sys.path.insert(0, "/root/dv3")
    sys.path.insert(0, "/root/fr")
    os.environ["DV3_RESULTS"] = "/results"
    import numpy as np, torch
    import trained.data as D

    D.VOCAB = 2011
    D.NAME0 = 11  # match task.py layout
    from trained.model import (
        TinyTransformer,
        lm_loss,
        make_optimizer,
        masked_answer_preds,
    )
    import permutation_task as T
    from shared.progress import log

    def key():
        return f"phase10/forced_reuse/seed{seed}"

    torch.manual_seed(seed)
    model = TinyTransformer(seed=seed, n_layers=8).to("cuda")
    # fit names [NAME0, NAME0+N_FIT) trainable; transfer names frozen+tied
    model.names_grad_mask[: T.N_FIT] = 1.0
    model.names_grad_mask[T.N_FIT :] = 0.0
    opt = make_optimizer(model, lr=1e-3, weight_decay=0.01)
    frozen_init = model.emb_names.detach().clone()

    def lr_at(s):
        if s < 500:
            return 1e-3 * (s + 1) / 500
        t = (s - 500) / max(1, MAX_STEPS - 500)
        return 1e-3 * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * min(t, 1.0))))

    # train perms = generators + a fixed subset of products (18 of 24);
    # held out = 6 products, to test unseen-permutation generalisation.
    rng0 = np.random.default_rng(1234)
    perm_order = list(T.PERMS)
    heldout = [T.PERMS[i] for i in rng0.choice(len(T.PERMS), 6, replace=False)]
    trainperms = [p for p in perm_order if p not in heldout]

    def batch_tokens(rng, pool):
        idx = rng.integers(0, len(pool), size=BATCH)
        gs = [trainperms[int(rng.integers(len(trainperms)))] for _ in idx]
        toks = [T.render(pool[i], g)[0] for i, g in zip(idx, gs)]
        return torch.as_tensor(np.stack(toks), device="cuda")

    def ev_sets():
        return dict(
            train_all=T.build_eval("fit", 96, 7000, perms=trainperms),
            held_perm=T.build_eval("fit", 96, 7001, perms=heldout),
            transfer=T.build_eval("transfer", 96, 7002, perms=trainperms),
        )

    @torch.no_grad()
    def acc(ev):
        p = masked_answer_preds(model, ev["tokens"], ev["answer_pos"], ev["candidates"])
        m = T.orbit_metrics(p, ev)
        return m["episode_acc"], m["strict_orbit"]

    rng = np.random.default_rng((seed, 77))
    pool = [T.sample_base(rng, "fit") for _ in range(POOL_SIZE)]
    evs = ev_sets()
    hist = []
    t0 = time.time()
    model.train()
    for step in range(1, MAX_STEPS + 1):
        if step % POOL_REFRESH == 0:
            pool = [T.sample_base(rng, "fit") for _ in range(POOL_SIZE)]
        for pg in opt.param_groups:
            pg["lr"] = lr_at(step)
        toks = batch_tokens(rng, pool)
        loss = lm_loss(model(toks), toks)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % EVAL_EVERY == 0 or step == MAX_STEPS:
            model.eval()
            frozen_ok = bool(
                torch.equal(model.emb_names[T.N_FIT :].detach(), frozen_init[T.N_FIT :])
            )
            a_tr, s_tr = acc(evs["train_all"])
            a_ho, s_ho = acc(evs["held_perm"])
            a_xf, s_xf = acc(evs["transfer"])
            hist.append(
                dict(
                    step=step,
                    loss=float(loss.item()),
                    train_acc=a_tr,
                    train_strict=s_tr,
                    heldperm_acc=a_ho,
                    heldperm_strict=s_ho,
                    transfer_acc=a_xf,
                    frozen_ok=frozen_ok,
                )
            )
            log(
                f"FR seed{seed} step{step} loss{loss.item():.3f} "
                f"train{a_tr:.3f}/{s_tr:.3f} heldperm{a_ho:.3f} xfer{a_xf:.3f} "
                f"frozen={frozen_ok} ({step/(time.time()-t0):.1f}/s)"
            )
            from shared import progress

            progress.save_json(f"{key()}/trajectory.json", hist)
            torch.save(
                dict(model=model.state_dict(), step=step, hist=hist),
                progress.results_dir() / f"{key()}/ckpt.pt",
            )
            vol.commit()
            model.train()
    # final verdict on formation
    final = hist[-1]
    formed = final["train_acc"] >= 0.5
    log(
        f"FR seed{seed} DONE: train_acc {final['train_acc']:.3f} "
        f"heldperm {final['heldperm_acc']:.3f} formed={formed}"
    )
    return dict(
        seed=seed, final=final, formed=formed, heldout_perms=[list(p) for p in heldout]
    )


@APP.local_entrypoint()
def main():
    print("Launching forced-reuse training (3 seeds, A100)...")
    res = list(train.map(SEEDS))
    for r in res:
        print(
            f"  seed{r['seed']}: train_acc {r['final']['train_acc']:.3f} "
            f"heldperm {r['final']['heldperm_acc']:.3f} formed={r['formed']}"
        )
    print("MILESTONE: forced-reuse training COMPLETE")
    print(json.dumps(res, indent=1)[:500])

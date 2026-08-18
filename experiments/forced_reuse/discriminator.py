"""Forced-reuse discriminator — does the trained transfer model implement the
permutation as a linear role operator, or by retrieval/routing?

Single-path task, so C2 (cross-path) is N/A (declared). The verdict rests on:
  C1  disjoint-vocabulary transport, fitted R vs the identity-fit ridge
  group laws (S_4: involution, both braid pairs, nonadjacent commutator)
  nontriv vs identity
A genuine role operator: R transports across disjoint vocab (C1 < identity < 1),
satisfies the S_4 laws non-trivially, and moves states more than identity.
Retrieval/routing: R is near-identity (C1 ~ identity, nontriv ~ identity, laws
satisfied only in the degenerate identity sense).

Runs only on seeds that FORMED (train_acc >= 0.5; gate-first). CPU-local:
loads the ckpt, caches answer-position residuals on disc orbits, fits.

Usage: .venv/bin/python experiments/phase10_forced_reuse_discriminator.py [seed]
Writes results/phase10/forced_reuse/disc_seed{S}.json
"""

import json
import pathlib
import sys

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments/forced_reuse"))

import trained.data as D  # noqa: E402

D.VOCAB = 2011
D.NAME0 = 11
import importlib  # noqa: E402
import trained.model as M  # noqa: E402

importlib.reload(M)
import permutation_task as T  # noqa: E402


def ridge_fit(X, Y, lam):
    """Frozen scale-free primal ridge (d=128 is small): R s.t. Y ~ X @ R.T."""
    d = X.shape[1]
    G = X.T @ X
    lam_eff = lam * np.trace(G) / d + 1e-12
    return np.linalg.solve(G + lam_eff * np.eye(d), X.T @ Y).T


GEN = T.GEN
N_LAYERS = 8
LAM_GRID = [1e-4, 1e-3, 1e-2, 1e-1, 1.0]
N_BASES = 200


def load_model(seed):
    ck = torch.load(
        ROOT / f"results/forced_reuse/seed{seed}/ckpt.pt",
        map_location="cpu",
        weights_only=False,
    )
    m = M.TinyTransformer(seed=seed, n_layers=N_LAYERS)
    m.load_state_dict(ck["model"])
    m.eval()
    return m, ck


@torch.no_grad()
def cache(model, ev, bs=256):
    toks = ev["tokens"]
    apos = ev["answer_pos"]
    n = len(toks)
    acts = np.zeros((n, N_LAYERS, 128), dtype=np.float32)
    for i in range(0, n, bs):
        t = torch.as_tensor(toks[i : i + bs])
        _, resids = model(t, capture=True)
        rows = torch.arange(len(t))
        ap = torch.as_tensor(apos[i : i + bs])
        for li, r in enumerate(resids):
            acts[i : i + len(t), li] = r[rows, ap].numpy()
    return acts


def pairs(acts, n_bases, perms, a, li):
    pidx = {p: i for i, p in enumerate(perms)}
    xi, yi = [], []
    for b in range(n_bases):
        for g in perms:
            xi.append(b * len(perms) + pidx[g])
            yi.append(b * len(perms) + pidx[T.compose(a, g)])
    return acts[np.array(xi), li].astype(np.float64), acts[np.array(yi), li].astype(
        np.float64
    )


def fit_gens(acts, n_bases, perms, li, lam, null=None, seed=0):
    rng = np.random.default_rng(seed)
    Rs = {}
    for a in GEN:
        X, Y = pairs(acts, n_bases, perms, a, li)
        if null == "identity":
            Y = X.copy()
        elif null == "shuffled":
            Y = Y[rng.permutation(len(Y))]
        Rs[a] = ridge_fit(X, Y, lam)  # Y ~ X @ R.T
    return Rs


def pair_error(Rs, acts, n_bases, perms, li):
    errs = []
    for a, R in Rs.items():
        X, Y = pairs(acts, n_bases, perms, a, li)
        errs.append(((X @ R.T - Y) ** 2).sum() / max(((Y - X) ** 2).sum(), 1e-12))
    return float(np.mean(errs))


def group_laws(Rs, H):
    R12, R23, R34 = Rs[GEN[0]], Rs[GEN[1]], Rs[GEN[2]]
    hn = (H**2).sum()
    ap = lambda V, R: V @ R.T
    inv = np.mean([((ap(ap(H, R), R) - H) ** 2).sum() / hn for R in (R12, R23, R34)])

    def braid(Ra, Rb):
        A, B = ap(ap(ap(H, Ra), Rb), Ra), ap(ap(ap(H, Rb), Ra), Rb)
        return ((A - B) ** 2).sum() / (0.5 * ((A**2).sum() + (B**2).sum()) + 1e-12 * hn)

    nontriv = np.mean([((ap(H, R) - H) ** 2).sum() / hn for R in (R12, R23, R34)])
    nc_nonadj = ((ap(ap(H, R12), R34) - ap(ap(H, R34), R12)) ** 2).sum() / hn
    return dict(
        law_inv_defect=float(inv),
        law_braid_defect=float(np.mean([braid(R12, R23), braid(R23, R34)])),
        nontriv=float(nontriv),
        noncommute_nonadj=float(nc_nonadj),
    )


def run(seed):
    model, ck = load_model(seed)
    traj = ck["hist"]
    train_acc = traj[-1]["train_acc"]
    print(
        f"seed{seed}: final train_acc={train_acc:.3f} "
        f"heldperm={traj[-1]['heldperm_acc']:.3f} formed={train_acc >= 0.5}"
    )
    if train_acc < 0.5:
        return dict(
            seed=seed,
            formed=False,
            train_acc=train_acc,
            verdict="not-formed (discriminator not run; gate-first)",
        )

    perms = T.PERMS
    ev_fit = T.build_eval("fit", N_BASES, 8100 + seed, perms=perms)
    ev_xf = T.build_eval("transfer", N_BASES, 8200 + seed, perms=perms)
    A_fit, A_xf = cache(model, ev_fit), cache(model, ev_xf)

    # verdict layer + lambda: min disjoint-vocab (C1) transfer error
    best = (None, None, np.inf)
    half = N_BASES // 2
    for lam in LAM_GRID:
        for li in range(N_LAYERS):
            Rs = fit_gens(A_fit, half, perms, li, lam)  # fit on first half fit-vocab
            e = pair_error(Rs, A_xf, half, perms, li)  # eval on transfer
            if e < best[2]:
                best = (lam, li, e)
    lam, li, _ = best

    Rs = fit_gens(A_fit, N_BASES, perms, li, lam)
    Rid = fit_gens(A_fit, N_BASES, perms, li, lam, null="identity")
    Hx = pairs(A_xf, N_BASES, perms, GEN[0], li)[0]
    m = dict(
        content_transfer_err=pair_error(Rs, A_xf, N_BASES, perms, li),
        content_transfer_err_identity=pair_error(Rid, A_xf, N_BASES, perms, li),
        identity_baseline=1.0,
    )
    m.update(group_laws(Rs, Hx))
    m["nontriv_identity"] = group_laws(Rid, Hx)["nontriv"]
    m["C1_beats_identity"] = bool(
        m["content_transfer_err"] < m["content_transfer_err_identity"]
    )
    m["C1_below_1"] = bool(m["content_transfer_err"] < 1.0)
    m["nontriv_beats_identity"] = bool(m["nontriv"] > m["nontriv_identity"] * 1.5)

    # verdict: role operator needs disjoint-vocab transport beating identity AND
    # non-trivial group structure. Otherwise retrieval/routing.
    role_like = (
        m["C1_beats_identity"] and m["C1_below_1"] and m["nontriv_beats_identity"]
    )
    verdict = "H_role (path-local; C2 N/A single-path)" if role_like else "H_retrieval"
    out = dict(
        seed=seed,
        formed=True,
        train_acc=train_acc,
        heldperm_acc=traj[-1]["heldperm_acc"],
        layer=li,
        lam=lam,
        metrics=m,
        verdict=verdict,
        reading=(
            f"C1 fitted {m['content_transfer_err']:.3f} vs identity "
            f"{m['content_transfer_err_identity']:.3f} vs 1.0; nontriv "
            f"{m['nontriv']:.4f} vs identity {m['nontriv_identity']:.4f}. "
            "Role operator iff C1 beats identity (<1) AND nontriv >> identity."
        ),
    )
    return out


def main():
    seeds = [int(sys.argv[1])] if len(sys.argv) > 1 else [0, 1, 2]
    (ROOT / "results/forced_reuse").mkdir(parents=True, exist_ok=True)
    allout = {}
    for s in seeds:
        o = run(s)
        allout[s] = o
        json.dump(
            o,
            open(ROOT / f"results/forced_reuse/disc_seed{s}.json", "w"),
            indent=2,
        )
        print(f"  seed{s}: {o['verdict']}")
        if o.get("formed"):
            print(f"    {o['reading']}")
    json.dump(
        allout,
        open(ROOT / "results/forced_reuse/disc_summary.json", "w"),
        indent=2,
    )


if __name__ == "__main__":
    main()

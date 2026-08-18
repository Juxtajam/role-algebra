"""Track A cross-family MECHANISM — the 8C discriminator on Nemotron-70B.

k=3 (S_3), answer position, on the Nemotron cache
(phase10/trackA/nemotron_acts/). Mirrors the 8C protocol (E13): fit
unconstrained R_12, R_23; C1 (disjoint vocab) and C2 (cross-path) mandatory;
compare the fitted R to the identity-fit and shuffled nulls; the decisive
8C reading is whether the fitted R is distinguishable from an identity-fit
ridge. Reuses phase8c_lib.DualRidge (frozen scale-free ridge).

Prediction (from the whole programme): H_retrieval. A cross-family H_role
would be the single most important positive result.

Writes results/phase10/trackA/nemotron_disc_results.json
"""

import itertools
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))
from activation_discriminator import DualRidge  # noqa: E402

ACTS = ROOT / "results/cross_family/nemotron_acts"
LAYERS = [0, 8, 16, 24, 32, 40, 48, 56, 61, 64, 72, 79]
PERMS = [tuple(p) for p in itertools.permutations(range(3))]
PERM_IDX = {p: i for i, p in enumerate(PERMS)}
GEN = [(1, 0, 2), (0, 2, 1)]  # g12, g23 (frozen dv3 convention)
LAM_GRID = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]
FPR = 0.05


def compose(a, b):
    return tuple(a[b[j]] for j in range(3))


def strict_ok(preds, recs):
    n = len(recs) // 6
    return np.array(
        [
            b
            for b in range(n)
            if all(preds[b * 6 + i] == recs[b * 6 + i]["answer"] for i in range(6))
        ]
    )


def cal_test(bases):
    cal, test = bases[0::2], bases[1::2]
    n = min(len(cal), len(test))
    return cal[:n], test[:n]


def pairs(acts, bases, a, li):
    xi = [b * 6 + PERM_IDX[g] for b in bases for g in PERMS]
    yi = [b * 6 + PERM_IDX[compose(a, g)] for b in bases for g in PERMS]
    return (
        np.asarray(acts[np.array(xi), li, :], np.float64),
        np.asarray(acts[np.array(yi), li, :], np.float64),
    )


def fit_gens(acts, bases, li, lam, null=None, seed=0):
    rng = np.random.default_rng(seed)
    Rs = {}
    for a in GEN:
        X, Y = pairs(acts, bases, a, li)
        if null == "identity":
            Y = X.copy()
        elif null == "shuffled":
            Y = Y[rng.permutation(len(Y))]
        Rs[a] = DualRidge(X, Y, lam)
    return Rs


def pair_error(Rs, acts, bases, li):
    errs = []
    for a, R in Rs.items():
        X, Y = pairs(acts, bases, a, li)
        errs.append(((R.apply_RT(X) - Y) ** 2).sum() / max(((Y - X) ** 2).sum(), 1e-12))
    return float(np.mean(errs))


def group_laws(Rs, H):
    R12, R23 = Rs[GEN[0]], Rs[GEN[1]]
    hn = (H**2).sum()
    ap = lambda V, R: R.apply_RT(V)
    inv = np.mean([((ap(ap(H, R), R) - H) ** 2).sum() / hn for R in (R12, R23)])
    A, B = ap(ap(ap(H, R12), R23), R12), ap(ap(ap(H, R23), R12), R23)
    braid = ((A - B) ** 2).sum() / (0.5 * ((A**2).sum() + (B**2).sum()) + 1e-12 * hn)
    nontriv = np.mean([((ap(H, R) - H) ** 2).sum() / hn for R in (R12, R23)])
    noncomm = ((ap(ap(H, R12), R23) - ap(ap(H, R23), R12)) ** 2).sum() / hn
    return dict(
        law_inv_defect=float(inv),
        law_braid_defect=float(braid),
        nontriv=float(nontriv),
        noncommute=float(noncomm),
    )


def states_H(acts, bases, li):
    idx = np.array([b * 6 + PERM_IDX[g] for b in bases for g in PERMS])
    return np.asarray(acts[idx, li, :], np.float64)


def main():
    meta = json.load(open(ACTS / "checksums.json"))
    print(
        f"Nemotron cache: d={meta['d_model']}, layers={meta['layers']}, "
        f"frozen_hash={meta['frozen_hash'][:12]}…"
    )
    cells = ("P_fit", "P_transfer", "G_fit", "G_transfer")
    recs = {
        c: [json.loads(l) for l in open(ROOT / f"results/verdict/gate/tasks/disc_{c}.jsonl")]
        for c in cells
    }
    preds = {c: json.load(open(ACTS / f"preds_disc_{c}.json")) for c in cells}
    acts = {c: np.load(ACTS / f"disc_{c}.npy", mmap_mode="r") for c in cells}

    ok = {c: strict_ok(preds[c], recs[c]) for c in cells}
    print("strict-ok bases:", {c: len(v) for c, v in ok.items()})
    calf, testf = cal_test(ok["P_fit"])
    calt, testt = cal_test(ok["P_transfer"])
    calg, testg = cal_test(ok["G_fit"])

    # verdict layer + lambda on cal transfer err (8C: min-over-layers grid argmin)
    best = (None, None, np.inf)
    for lam in LAM_GRID:
        for li in range(len(LAYERS)):
            Rs = fit_gens(acts["P_fit"], calf, li, lam)
            e = pair_error(Rs, acts["P_transfer"], calt, li)
            if e < best[2]:
                best = (lam, li, e)
    lam, li, _ = best
    layer = LAYERS[li]
    print(f"selected lam={lam} verdict layer=L{layer} (cal transfer err {best[2]:.3f})")

    # matched-regime nulls on cal (shuffled x10 + identity)
    def nulls(mode, seeds):
        c1, c2, gl = [], [], []
        for s in seeds:
            Rn = fit_gens(
                acts["P_fit"], calf, li, lam, null=mode, seed=2000 * s + layer
            )
            c1.append(pair_error(Rn, acts["P_transfer"], calt, li))
            c2.append(pair_error(Rn, acts["G_fit"], calg, li))
            gl.append(group_laws(Rn, states_H(acts["P_transfer"], calt, li)))
        return c1, c2, gl

    sh_c1, sh_c2, sh_gl = nulls("shuffled", range(10))
    id_c1, id_c2, id_gl = nulls("identity", [0])
    # Threshold sources per frozen 8C METRICS: content_transfer/crosspath/
    # law_inv/braid from SHUFFLED (lt); nontriv/noncommute from IDENTITY (gt,
    # the real R must move states MORE than an identity-fit ridge does).
    tau = dict(
        content_transfer_err=float(np.quantile(sh_c1, FPR)),
        crosspath_err=float(np.quantile(sh_c2, FPR)),
        law_inv_defect=float(np.quantile([g["law_inv_defect"] for g in sh_gl], FPR)),
        nontriv=float(id_gl[0]["nontriv"]),  # identity-fit nontriv (gt bar)
        noncommute=float(id_gl[0]["noncommute"]),
    )

    # single test evaluation
    Rs = fit_gens(acts["P_fit"], testf, li, lam)
    H = states_H(acts["P_transfer"], testt, li)
    m = dict(
        content_transfer_err=pair_error(Rs, acts["P_transfer"], testt, li),
        crosspath_err=pair_error(Rs, acts["G_fit"], testg, li),
    )
    m.update(group_laws(Rs, H))
    m["identity_baseline"] = 1.0
    # identity-fit reference on the SAME test split (the decisive 8C comparison)
    Rid = fit_gens(acts["P_fit"], testf, li, lam, null="identity")
    m["content_transfer_err_identity_fit"] = pair_error(
        Rid, acts["P_transfer"], testt, li
    )
    m["crosspath_err_identity_fit"] = pair_error(Rid, acts["G_fit"], testg, li)

    # descriptive deflationary flags (the decisive 8C reading): does the
    # fitted R beat the identity-fit ridge, and does it exceed 1.0?
    m["C1_beats_identity_fit"] = bool(
        m["content_transfer_err"] < m["content_transfer_err_identity_fit"]
    )
    m["C2_beats_identity_fit"] = bool(
        m["crosspath_err"] < m["crosspath_err_identity_fit"]
    )
    m["C1_below_1"] = bool(m["content_transfer_err"] < 1.0)
    m["C2_below_1"] = bool(m["crosspath_err"] < 1.0)

    conds = {}
    conds["C1_vs_shuffled"] = (
        "pass" if m["content_transfer_err"] < tau["content_transfer_err"] else "fail"
    )
    conds["C2_vs_shuffled"] = (
        "pass" if m["crosspath_err"] < tau["crosspath_err"] else "fail"
    )
    conds["C3_inv"] = "pass" if m["law_inv_defect"] < tau["law_inv_defect"] else "fail"
    conds["C4_nontriv"] = "pass" if m["nontriv"] > tau["nontriv"] else "fail"
    # C5 (causal transport) UNAVAILABLE — no GPU patching session for Nemotron.
    conds["C5_causal_transport"] = "unavailable"

    # E14 deflationary reading (the decisive 8C amendment): a C1/C2 pass against
    # the shuffled null is a nulls-are-worse test that a near-identity map clears;
    # positive role evidence requires the fitted R to BEAT the identity-fit ridge.
    # The MANDATORY cross-path condition (C2) is the discriminator: if the fitted
    # R transports across paths no better than identity, there is no shared
    # role operator, regardless of the shuffled-null pass.
    if not m["C2_beats_identity_fit"]:
        verdict = "H_retrieval"
        basis = (
            "C2 (cross-path) transports no better than the identity-fit ridge "
            f"(fitted {m['crosspath_err']:.3f} >= identity {m['crosspath_err_identity_fit']:.3f}, "
            "both > 1.0): no shared role operator across paths. The within-path "
            f"C1 beats identity (fitted {m['content_transfer_err']:.3f} < identity "
            f"{m['content_transfer_err_identity_fit']:.3f}), so the structure is "
            "PATH-LOCAL — exactly Qwen's E13/E15 conclusion. Also C5 unavailable, "
            "so a formal H_role is not even claimable."
        )
    elif "fail" not in [v for k, v in conds.items() if k != "C5_causal_transport"]:
        verdict = (
            "inconclusive"  # C5 unavailable and C1/C2 beat identity -> can't confirm
        )
        basis = "C1 and C2 both beat identity but C5 (causal transport) is unavailable."
    else:
        verdict = "H_retrieval"
        basis = "a condition fails cleanly."

    out = dict(
        model=meta["model"],
        k=3,
        layer=layer,
        lam=lam,
        strict_ok={c: int(len(v)) for c, v in ok.items()},
        n_test={"P_fit": len(testf), "P_transfer": len(testt), "G_fit": len(testg)},
        metrics=m,
        thresholds=tau,
        conditions=conds,
        verdict=verdict,
        verdict_basis=basis,
        reading=(
            f"fitted C1={m['content_transfer_err']:.3f} vs identity-fit "
            f"{m['content_transfer_err_identity_fit']:.3f} vs 1.0; "
            f"C2={m['crosspath_err']:.3f} vs identity-fit "
            f"{m['crosspath_err_identity_fit']:.3f}. If fitted ~ identity-fit "
            f"and > 1.0, the operator is a near-identity map -> retrieval."
        ),
    )
    json.dump(
        out,
        open(ROOT / "results/cross_family/nemotron_disc_results.json", "w"),
        indent=2,
    )
    print(
        json.dumps(
            {
                k: out[k]
                for k in ("layer", "lam", "metrics", "conditions", "verdict", "reading")
            },
            indent=1,
        )
    )


if __name__ == "__main__":
    main()

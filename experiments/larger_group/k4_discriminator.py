"""Track C — k=4 (S_4) discriminator on the Qwen2.5-72B answer position.

Runs only if the k=4 gate passed and disc activations were cached
(phase10/trackC/acts_k4/). Tests the same question as 8C at a larger group:
is there a transportable role operator, or chained retrieval?

S_4 adaptation (discriminator_design.md §2.2, Appendix D): 3 generators
g12,g23,g34; group laws use 3-generator involution, both adjacent braid pairs,
and the S_4-unique NONADJACENT commutator (g12,g34 must commute). C1 (disjoint
vocab) and C2 (cross-path) are mandatory and generator-count-independent.

Reuses the frozen scale-free ridge (phase8c_lib.ridge_fit) and the frozen
verdict logic; the pairing and group-law formulas are S_4 versions of the
frozen k=3 forms. Answer position, verdict layer selected on calibration.

Writes results/phase10/trackC/k4_disc_results.json
"""

import itertools
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))
from activation_discriminator import DualRidge  # noqa: E402  (frozen scale-free ridge, factored)

ACTS = ROOT / "results/larger_group/acts_k4"
TASKS = ROOT / "phase10/trackC/tasks_k4"
LAYERS = [0, 8, 16, 24, 32, 40, 48, 56, 61, 64, 72, 79]
PERMS = [tuple(p) for p in itertools.permutations(range(4))]  # 24
PERM_IDX = {p: i for i, p in enumerate(PERMS)}
GEN = [(1, 0, 2, 3), (0, 2, 1, 3), (0, 1, 3, 2)]  # g12,g23,g34
LAM_GRID = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]
FPR = 0.05
MIN_UNIVERSE = 20


def compose(a, b):
    return tuple(a[b[j]] for j in range(4))


def load(cell):
    return np.load(ACTS / f"disc_{cell}.npy", mmap_mode="r")  # (n_ep, 12, d)


def strict_ok_bases(preds, recs):
    """bases where all 24 perms are answered correctly."""
    n = len(recs) // 24
    ok = []
    for b in range(n):
        if all(preds[b * 24 + i] == recs[b * 24 + i]["answer"] for i in range(24)):
            ok.append(b)
    return np.array(ok)


def cal_test(bases):
    cal, test = bases[0::2], bases[1::2]
    n = min(len(cal), len(test))
    return cal[:n], test[:n]


def pairs(acts, bases, a, li):
    xi, yi = [], []
    for b in bases:
        for g in PERMS:
            xi.append(b * 24 + PERM_IDX[g])
            yi.append(b * 24 + PERM_IDX[compose(a, g)])
    X = np.asarray(acts[np.array(xi), li, :], np.float64)
    Y = np.asarray(acts[np.array(yi), li, :], np.float64)
    return X, Y


def fit_gens(acts, bases, li, lam, null=None, seed=0):
    """{generator: DualRidge} — factored ridge (efficient at d=8192, n<<d)."""
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
        P = R.apply_RT(X)
        errs.append(((P - Y) ** 2).sum() / max(((Y - X) ** 2).sum(), 1e-12))
    return float(np.mean(errs))


def group_laws_s4(Rs, H):
    """S_4 group-law defects (design Appendix D), activation-weighted on H."""
    R12, R23, R34 = Rs[GEN[0]], Rs[GEN[1]], Rs[GEN[2]]
    hn = (H**2).sum()
    ap = lambda V, R: R.apply_RT(V)
    inv = np.mean([((ap(ap(H, R), R) - H) ** 2).sum() / hn for R in (R12, R23, R34)])

    def braid(Ra, Rb):
        A = ap(ap(ap(H, Ra), Rb), Ra)
        B = ap(ap(ap(H, Rb), Ra), Rb)
        return ((A - B) ** 2).sum() / (0.5 * ((A**2).sum() + (B**2).sum()) + 1e-12 * hn)

    braid_defect = float(np.mean([braid(R12, R23), braid(R23, R34)]))
    nontriv = np.mean([((ap(H, R) - H) ** 2).sum() / hn for R in (R12, R23, R34)])
    nc_adj = np.mean(
        [
            ((ap(ap(H, R12), R23) - ap(ap(H, R23), R12)) ** 2).sum() / hn,
            ((ap(ap(H, R23), R34) - ap(ap(H, R34), R23)) ** 2).sum() / hn,
        ]
    )
    nc_nonadj = ((ap(ap(H, R12), R34) - ap(ap(H, R34), R12)) ** 2).sum() / hn
    return dict(
        law_inv_defect=float(inv),
        law_braid_defect=braid_defect,
        nontriv=float(nontriv),
        noncommute_adj=float(nc_adj),
        noncommute_nonadj=float(nc_nonadj),
    )


def states_H(acts, bases, li):
    idx = np.array([b * 24 + PERM_IDX[g] for b in bases for g in PERMS])
    return np.asarray(acts[idx, li, :], np.float64)


def main():
    meta = json.load(open(ACTS / "checksums.json"))
    d = meta["d_model"]
    recs = {
        c: [json.loads(l) for l in open(TASKS / f"disc_{c}.jsonl")]
        for c in ("P_fit", "P_transfer", "G_fit")
    }
    preds = {
        c: json.load(open(ROOT / f"results/larger_group/preds_disc_{c}.json"))
        for c in ("P_fit", "P_transfer", "G_fit")
    }
    acts = {c: load(c) for c in ("P_fit", "P_transfer", "G_fit")}

    ok = {c: strict_ok_bases(preds[c], recs[c]) for c in recs}
    print("strict-ok bases:", {c: len(v) for c, v in ok.items()})
    calf, testf = cal_test(ok["P_fit"])
    calt, testt = cal_test(ok["P_transfer"])
    calg, testg = cal_test(ok["G_fit"])
    avail = dict(C1=len(testt) >= MIN_UNIVERSE, C2=len(testg) >= MIN_UNIVERSE)

    # lambda + verdict layer on cal transfer error (fall back to G if C1 thin)
    ev_c, ev_cal = ("P_transfer", calt) if avail["C1"] else ("G_fit", calg)
    best = (None, None, np.inf)
    for gi, lam in enumerate(LAM_GRID):
        for li in range(len(LAYERS)):
            Rs = fit_gens(acts["P_fit"], calf, li, lam)
            e = pair_error(Rs, acts[ev_c], ev_cal, li)
            if e < best[2]:
                best = (lam, li, e)
    lam, li, _ = best
    layer = LAYERS[li]
    print(f"selected lam={lam} layer=L{layer} (cal err {best[2]:.3f} on {ev_c})")

    # thresholds from matched-regime shuffled nulls on cal
    nulls = {
        "content_transfer_err": [],
        "crosspath_err": [],
        "law_inv_defect": [],
        "law_braid_defect": [],
        "nontriv": [],
    }
    for s in range(10):
        Rn = fit_gens(
            acts["P_fit"], calf, li, lam, null="shuffled", seed=2000 * s + layer
        )
        if avail["C1"]:
            nulls["content_transfer_err"].append(
                pair_error(Rn, acts["P_transfer"], calt, li)
            )
        if avail["C2"]:
            nulls["crosspath_err"].append(pair_error(Rn, acts["G_fit"], calg, li))
        Hn = states_H(
            acts["P_transfer"] if avail["C1"] else acts["P_fit"],
            calt if avail["C1"] else calf,
            li,
        )
        gl = group_laws_s4(Rn, Hn)
        for k in ("law_inv_defect", "law_braid_defect", "nontriv"):
            nulls[k].append(gl[k])
    tau = {}
    for k, vals in nulls.items():
        if vals:
            q = FPR if k != "nontriv" else 1 - FPR
            tau[k] = float(np.quantile(vals, q))

    # single test evaluation
    Rs = fit_gens(acts["P_fit"], testf, li, lam)
    m = {}
    if avail["C1"]:
        m["content_transfer_err"] = pair_error(Rs, acts["P_transfer"], testt, li)
    if avail["C2"]:
        m["crosspath_err"] = pair_error(Rs, acts["G_fit"], testg, li)
    Heval = states_H(
        acts["P_transfer"] if avail["C1"] else acts["P_fit"],
        testt if avail["C1"] else testf,
        li,
    )
    m.update(group_laws_s4(Rs, Heval))
    # identity baseline (transfers exactly 1.0 by construction)
    m["identity_baseline"] = 1.0

    # verdict (frozen logic on the available mandatory conditions)
    conds = {}
    if avail["C1"]:
        conds["C1"] = (
            "pass"
            if m["content_transfer_err"] < tau["content_transfer_err"]
            else "fail"
        )
    else:
        conds["C1"] = "unavailable"
    if avail["C2"]:
        conds["C2"] = "pass" if m["crosspath_err"] < tau["crosspath_err"] else "fail"
    else:
        conds["C2"] = "unavailable"
    conds["C3_inv"] = (
        "pass" if m["law_inv_defect"] < tau.get("law_inv_defect", np.inf) else "fail"
    )
    if "fail" in conds.values():
        verdict = "H_retrieval"
    elif conds.get("C1") == "pass" and conds.get("C2") == "pass":
        verdict = "H_role"
    else:
        verdict = "inconclusive"

    out = dict(
        k=4,
        layer=layer,
        lam=lam,
        strict_ok={c: int(len(v)) for c, v in ok.items()},
        n_test={"P_fit": len(testf), "P_transfer": len(testt), "G_fit": len(testg)},
        availability=avail,
        metrics=m,
        thresholds=tau,
        conditions=conds,
        verdict=verdict,
        note="S_4 discriminator; C1/C2 mandatory; identity baseline = 1.0 "
        "(fitted C1 > 1 means worse than not moving).",
    )
    json.dump(
        out, open(ROOT / "results/larger_group/k4_disc_results.json", "w"), indent=2
    )
    print(
        json.dumps(
            {
                k: out[k]
                for k in (
                    "layer",
                    "lam",
                    "availability",
                    "metrics",
                    "conditions",
                    "verdict",
                )
            },
            indent=1,
        )
    )


if __name__ == "__main__":
    main()

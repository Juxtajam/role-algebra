"""Does the late C1xC7 dissociation live in the
name-token subspace?

For each C1xC7-dissociation cell (query_arg L56-79, answer L56/64-79; E16),
refit R_12, R_23 exactly as Phase 9 did (importing the frozen fit module),
verify the refit reproduces the stored cell `c1` (fidelity gate), then
measure what fraction of the fitted (R - I) Frobenius energy is captured by
the 24-dim fit-vocabulary name-READOUT subspace, against a matched-dimension
random-subspace baseline.

Reading, fixed before the numbers are seen (the plan A1 adjudication):
  name-subspace fraction >> random baseline  -> vocabulary-bound readout texture
  name-subspace fraction ~ random baseline   -> position-mixture / scaffold geometry
Either way this is a DESCRIPTIVE label on an already-negative result; no
verdict changes.

Reuses: results/phase9/code/phase9_item3_fit.py (module-level machinery).
Reads:  acts_P_local/ cache, results/phase8c/name_vectors.npz, cell_*.json.
Writes: results/phase9/A1_name_support.json
"""

import importlib.util
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
FITSRC = ROOT / "results/binding_sites/code/phase9_item3_fit.py"

spec = importlib.util.spec_from_file_location("p9fit", FITSRC)
fit = importlib.util.module_from_spec(spec)
sys.modules["p9fit"] = fit
spec.loader.exec_module(fit)  # loads CELLS (mmap), splits, name vectors

# C1xC7 dissociation cells (E16 / phase9_report Late-Stack section)
CELLS_QA = [56, 60, 61, 64, 68, 72, 76, 79]
CELLS_ANS = [56, 64, 68, 72, 76, 79]
DISSOC = [("query_arg", L) for L in CELLS_QA] + [("answer", L) for L in CELLS_ANS]

READOUT = fit.READOUT  # {name: (8192,) readout row}
NAME_LIST = fit.NAME_LIST
D = 8192


def name_subspace():
    """Orthonormal basis of span{24 fit-vocab name readout rows}."""
    U = np.stack([READOUT[n] for n in NAME_LIST]).astype(np.float64)  # (24, d)
    Q, s, _ = np.linalg.svd(U.T, full_matrices=False)  # Q: (d, 24)
    r = int((s >= 1e-9 * s[0]).sum())
    return Q[:, :r], r


def frac_in_subspace(R, Q):
    """||(R - I) Q||_F^2 / ||R - I||_F^2  — fraction of (R-I) energy acting on
    the column subspace spanned by orthonormal Q (d x r)."""
    RQ = R.apply_R_cols(Q)  # (d, r) = R Q
    num = float(((RQ - Q) ** 2).sum())
    den = max(R.frob2_R_minus_I(), 1e-12)
    return num / den


def refit(pc, li):
    """Reproduce the Phase 9 fit at (pc, layer-index li): lambda selected on
    cal transfer-err, R fit on cal fit-vocab bases."""
    fit_cell, tr_cell = fit.CELLS[("frozen", "fit")], fit.CELLS[("frozen", "transfer")]
    calF, testF = fit.SPLIT[("frozen", "fit")]
    calT, testT = fit.SPLIT[("frozen", "transfer")]
    shared, Ycal = {}, {}
    for a in fit.GENERATORS:
        rx, cx, ry, cy = fit.pair_samples(fit_cell, calF, a, pc)
        X = fit_cell.states(rx, li, cx)
        Y = fit_cell.states(ry, li, cy)
        shared[a] = fit.SharedRidge(X)
        Ycal[a] = Y
    best = (None, np.inf)
    for lam in fit.LAMBDAS:
        Rs = {a: shared[a].fit(Ycal[a], lam) for a in fit.GENERATORS}
        e = fit.transport_err(Rs, tr_cell, calT, li, pc)
        if e < best[1]:
            best = (lam, e)
    lam = best[0]
    Rs = {a: shared[a].fit(Ycal[a], lam) for a in fit.GENERATORS}
    c1 = fit.transport_err(Rs, tr_cell, testT, li, pc)
    return Rs, lam, c1


def main():
    Qn, rn = name_subspace()
    print(
        f"name-readout subspace dim = {rn} of {D}; isotropic baseline r/d = {rn / D:.4e}"
    )
    rng = np.random.default_rng(20260815)
    # matched random subspaces (shared across cells for comparability)
    Qrs = []
    for _ in range(5):
        A = rng.standard_normal((D, rn))
        Qr, _ = np.linalg.qr(A)
        Qrs.append(Qr[:, :rn])

    out = {"name_subspace_dim": rn, "isotropic_baseline": rn / D, "cells": {}}
    print(
        f"\n{'cell':>16} | {'lam':>5} | {'c1':>7} {'stored':>7} {'ok':>4} | "
        f"{'name_frac':>9} | {'rand_frac':>9} | {'ratio':>6} | label"
    )
    for pc, L in DISSOC:
        li = fit.LAYERS.index(L)
        Rs, lam, c1 = refit(pc, li)
        stored = json.load(open(ROOT / f"results/binding_sites/fits/cell_{pc}_L{L}.json"))
        ok = abs(c1 - stored["c1"]) < 1e-6
        name_fracs = [frac_in_subspace(R, Qn) for R in Rs.values()]
        name_frac = float(np.mean(name_fracs))
        rand_frac = float(
            np.mean([frac_in_subspace(R, Qr) for R in Rs.values() for Qr in Qrs])
        )
        ratio = name_frac / max(rand_frac, 1e-12)
        label = "vocab-bound" if ratio >= 2.0 else "not-name-concentrated"
        out["cells"][f"{pc}/L{L}"] = dict(
            lam=lam,
            c1=c1,
            stored_c1=stored["c1"],
            fidelity_ok=bool(ok),
            name_frac=name_frac,
            rand_frac=rand_frac,
            ratio=ratio,
            per_generator_name_frac=name_fracs,
            label=label,
        )
        print(
            f"{pc+'/L'+str(L):>16} | {lam:>5.0e} | {c1:7.3f} {stored['c1']:7.3f} "
            f"{'YES' if ok else 'NO!':>4} | {name_frac:9.4f} | {rand_frac:9.4f} | "
            f"{ratio:6.2f} | {label}"
        )

    all_ok = all(v["fidelity_ok"] for v in out["cells"].values())
    ratios = [v["ratio"] for v in out["cells"].values()]
    n_vocab = sum(v["label"] == "vocab-bound" for v in out["cells"].values())
    out["summary"] = dict(
        fidelity_all_ok=all_ok,
        n_cells=len(out["cells"]),
        n_vocab_bound=n_vocab,
        median_ratio=float(np.median(ratios)),
        adjudication=(
            "vocabulary-bound readout texture"
            if n_vocab >= (len(out["cells"]) + 1) // 2
            else "position-mixture / scaffold geometry"
        ),
    )
    json.dump(out, open(ROOT / "results/binding_sites/A1_name_support.json", "w"), indent=1)
    print(
        f"\nfidelity: {'PASS' if all_ok else 'FAIL'} (refit c1 == stored c1 at all cells)"
    )
    print(
        f"vocab-bound cells: {n_vocab}/{len(out['cells'])}; "
        f"median name/rand ratio {np.median(ratios):.2f}"
    )
    print(f"ADJUDICATION: {out['summary']['adjudication']}")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()

"""Per-clause breakdown of the fact_final cells.

The Phase 9 fit pooled all 6 fact_final columns (one column per episode,
cycled). This splits them by clause type and refits
separately, to check whether a clause-specific role code was diluted by the
pooling. Path P fact clauses: 3 "bears" clauses (sigil bears mark) and 3
"holds" clauses (name holds sigil). Canonical fact index 0-2 = bears,
3-5 = holds (verified from base.fact_order vs rendered text); a fact_final
column's clause type is a per-base property (fact_order is shared across the
orbit), so every (x, g.x) pair has a well-defined clause type.

EXPLORATORY, labelled as such. Adjudication: a subclass
beating identity on C1 AND C7 -> flag for a separately preregistered
confirmatory re-run; anything else -> the D2 pooling was immaterial and closes.

Reuses the frozen Phase 9 fit module. Reads acts_P_local/ (P path only, so
C1 = disjoint-vocab transfer is available; C7 = joint transport available;
C2 cross-path is NOT — no G cache — and is not attempted).
Writes results/phase9/A2_factfinal_perclause.json
"""

import importlib.util
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "p9fit", ROOT / "results/binding_sites/code/phase9_item3_fit.py"
)
fit = importlib.util.module_from_spec(spec)
sys.modules["p9fit"] = fit
spec.loader.exec_module(fit)

FF_COLS = fit.POS["fact_final"]  # [3,4,5,6,7,8]
LAYERS_A2 = [56, 61, 64, 68, 72, 76, 79]
CLAUSE = {"bears": lambda ci: ci < 3, "holds": lambda ci: ci >= 3}


def clause_of_column(rec, col):
    """Clause type of manifest fact_final column `col` for this episode's
    base: textual index = col-3; canonical fact index = fact_order[textual];
    canonical 0-2 = bears, 3-5 = holds."""
    fo = rec["base"]["fact_order"]
    canonical = fo[col - FF_COLS[0]]
    return "bears" if canonical < 3 else "holds"


def pair_samples_factfinal_clause(cell, bases, a, clause):
    """Like fit.pair_samples for fact_final, but restricted to columns whose
    clause type == `clause`. Same one-column-per-(episode) cycling rule; the
    pair uses the same column for x and y (a per-base property)."""
    rx, cx, ry, cy = [], [], [], []
    for b in bases:
        for g in fit.PERMS:
            i = cell.row(b, g)
            j = cell.row(b, fit.compose(a, g))
            c = FF_COLS[(fit.PERM_IDX[tuple(g)] + b) % len(FF_COLS)]
            if clause_of_column(cell.recs[i], c) != clause:
                continue
            rx.append(i)
            cx.append(c)
            ry.append(j)
            cy.append(c)
    return (np.array(rx), np.array(cx), np.array(ry), np.array(cy))


def transport_err_clause(Rs, cell, bases, li, clause, joint=False):
    errs = []
    for a, R in Rs.items():
        rx, cx, ry, cy = pair_samples_factfinal_clause(cell, bases, a, clause)
        if joint:
            # joint set: same bases, joint rendering; carry cols differ but
            # fact_final columns are re-derived per joint episode's own base
            pass
        X = cell.states(rx, li, cx)
        Y = cell.states(ry, li, cy)
        errs.append(((R.apply_RT(X) - Y) ** 2).sum() / max(((Y - X) ** 2).sum(), 1e-12))
    return float(np.mean(errs)), len(rx)


def fit_clause(clause, li):
    fit_cell, tr_cell = fit.CELLS[("frozen", "fit")], fit.CELLS[("frozen", "transfer")]
    jfit = fit.CELLS[("joint", "fit")]
    calF, testF = fit.SPLIT[("frozen", "fit")]
    calT, testT = fit.SPLIT[("frozen", "transfer")]
    calJF, testJF = fit.SPLIT[("joint", "fit")]
    shared, Ycal = {}, {}
    for a in fit.GENERATORS:
        rx, cx, ry, cy = pair_samples_factfinal_clause(fit_cell, calF, a, clause)
        shared[a] = fit.SharedRidge(fit_cell.states(rx, li, cx))
        Ycal[a] = fit_cell.states(ry, li, cy)
    best = (None, np.inf)
    for lam in fit.LAMBDAS:
        Rs = {a: shared[a].fit(Ycal[a], lam) for a in fit.GENERATORS}
        e, _ = transport_err_clause(Rs, tr_cell, calT, li, clause)
        if e < best[1]:
            best = (lam, e)
    lam = best[0]
    Rs = {a: shared[a].fit(Ycal[a], lam) for a in fit.GENERATORS}
    # identity-fit baseline (Y := X)
    Rid = {a: shared[a].fit(shared[a].X, lam) for a in fit.GENERATORS}
    c1, n1 = transport_err_clause(Rs, tr_cell, testT, li, clause)
    c1_id, _ = transport_err_clause(Rid, tr_cell, testT, li, clause)
    c7, n7 = transport_err_clause(Rs, jfit, testJF, li, clause, joint=True)
    c7_id, _ = transport_err_clause(Rid, jfit, testJF, li, clause, joint=True)
    # matched shuffled null for C1
    null_c1 = []
    for seed in range(10):
        rng = np.random.default_rng(seed)
        Rn = {
            a: shared[a].fit(Ycal[a][rng.permutation(len(Ycal[a]))], lam)
            for a in fit.GENERATORS
        }
        e, _ = transport_err_clause(Rn, tr_cell, calT, li, clause)
        null_c1.append(e)
    tau_c1 = float(np.min(null_c1))
    return dict(
        lam=lam,
        c1=c1,
        c1_identity=c1_id,
        tau_c1=tau_c1,
        c1_beats_identity=bool(c1 < c1_id),
        c1_pass=bool(c1 < tau_c1 and c1 < c1_id),
        c7=c7,
        c7_identity=c7_id,
        c7_beats_identity=bool(c7 < c7_id),
        n_test_pairs=n1,
    )


def main():
    out = {
        "note": "EXPLORATORY (D2 pooling audit); P-path only; C2 unavailable",
        "cells": {},
    }
    print(
        f"{'cell':>18} {'clause':>6} | {'lam':>5} | {'c1':>7} {'c1_id':>7} "
        f"{'c1<id':>5} | {'c7':>6} {'c7_id':>6} {'c7<id':>5} | flag"
    )
    flagged = []
    for L in LAYERS_A2:
        li = fit.LAYERS.index(L)
        for clause in ("bears", "holds"):
            r = fit_clause(clause, li)
            key = f"fact_final/L{L}/{clause}"
            out["cells"][key] = r
            flag = "FLAG" if (r["c1_pass"] and r["c7_beats_identity"]) else ""
            if flag:
                flagged.append(key)
            print(
                f"{'fact_final/L'+str(L):>18} {clause:>6} | {r['lam']:>5.0e} | "
                f"{r['c1']:7.3f} {r['c1_identity']:7.3f} "
                f"{'YES' if r['c1_beats_identity'] else 'no':>5} | "
                f"{r['c7']:6.3f} {r['c7_identity']:6.3f} "
                f"{'YES' if r['c7_beats_identity'] else 'no':>5} | {flag}"
            )
    out["flagged_for_confirmatory"] = flagged
    out["adjudication"] = (
        "D2 pooling immaterial — no subclass beats identity on C1 AND C7"
        if not flagged
        else f"{len(flagged)} subclass cell(s) flagged for preregistered confirmatory re-run"
    )
    json.dump(
        out, open(ROOT / "results/binding_sites/A2_factfinal_perclause.json", "w"), indent=1
    )
    print(f"\nADJUDICATION: {out['adjudication']}")


if __name__ == "__main__":
    main()

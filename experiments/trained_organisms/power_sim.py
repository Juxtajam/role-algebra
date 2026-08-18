"""Phase 7 Item 3 — pricing simulation for the dose-response redesign memo.

Question (spec): given the Item 1 composition formation rate r, how many
seeds per rung does a Spearman test across four rungs (T0-T3) need to
distinguish MONOTONIC from FLAT at reasonable power?

Design priced: per-run outcome is BINARY (composition forms: composed
held-out >= 0.95 at any eval — the standing branch statistic). Test:
one-sided Spearman (rho > 0) between rung index (0-3, n runs/rung) and the
binary outcome, permutation p-value, alpha = 0.05. With binary y the
permutation null of the Spearman numerator T = sum(centered-x-ranks at
success positions) depends only on the success count k, so the null is
precomputed per k by vectorised Monte Carlo (20,000 draws) and each
simulated experiment is a table lookup. Power by 20,000 simulated
experiments per cell.

Alternatives, anchored so the SECOND rung (T1, the measured organism)
equals r:
  strong:   d = r     -> p = clip([r-d, r, r+d, r+2d], .02, .98)
  moderate: d = r/2
  weak:     d = r/4
Flat null: p = r everywhere (reported column = realised false-positive
rate, should be <= alpha).
"""

import sys
import numpy as np

r = float(sys.argv[1]) if len(sys.argv) > 1 else 0.375
ALPHA = 0.05
N_SIM = 20_000
N_NULL = 20_000
rng = np.random.default_rng(7)


def make_null_tables(n):
    """For each success count k in 0..4n: the MC null distribution of
    T = sum of centered x-ranks at k uniformly-random positions."""
    x = np.repeat(np.arange(4), n).astype(float)
    rx = (
        x.argsort().argsort().astype(float)
    )  # ranks 0..4n-1 (ties none in x? x has ties)
    # proper average ranks for tied x:
    rx = np.empty(4 * n)
    for v in range(4):
        idx = np.where(x == v)[0]
        # average rank of this tie group (1-based irrelevant, centered later)
        lo = v * n
        rx[idx] = lo + (n - 1) / 2.0
    rxc = rx - rx.mean()
    N = 4 * n
    tables = []
    for k in range(N + 1):
        if k == 0 or k == N:
            tables.append(np.zeros(1))
            continue
        # vectorised: N_NULL random subsets of size k via argpartition of random keys
        keys = rng.random((N_NULL, N))
        part = np.argpartition(keys, k, axis=1)[:, :k]
        tables.append(np.sort(rxc[part].sum(axis=1)))
    return rxc, tables


def power(ps, n, rxc, tables):
    N = 4 * n
    p_vec = np.repeat(ps, n)
    y = rng.random((N_SIM, N)) < p_vec
    T = (y * rxc).sum(axis=1)
    k = y.sum(axis=1)
    hits = 0
    for i in range(N_SIM):
        tab = tables[k[i]]
        if len(tab) == 1:
            continue  # degenerate all-0/all-1: never significant
        # one-sided p = P(T_null >= T_obs)
        pval = 1.0 - np.searchsorted(tab, T[i], side="left") / len(tab)
        pval = max(pval, 1.0 / len(tab))
        if pval <= ALPHA:
            hits += 1
    return hits / N_SIM


def alt(d):
    return np.clip(np.array([r - d, r, r + d, r + 2 * d]), 0.02, 0.98)


alts = {"weak": alt(r / 4), "moderate": alt(r / 2), "strong": alt(r)}
print(
    f"Item 1 formation rate anchor r = {r:.3f}; alpha {ALPHA} one-sided; "
    f"{N_SIM} sims, {N_NULL}-draw permutation nulls"
)
for name, ps in alts.items():
    print(f"  {name}: p = {np.round(ps, 3)}")
print(
    f"{'n/rung':>7} {'total':>6} | {'flat FPR':>8} | {'weak':>6} "
    f"{'moderate':>9} {'strong':>7}"
)
for n in (5, 8, 10, 12, 15, 20, 25, 30, 40):
    rxc, tables = make_null_tables(n)
    fpr = power(np.full(4, r), n, rxc, tables)
    row = {k_: power(v, n, rxc, tables) for k_, v in alts.items()}
    print(
        f"{n:>7} {4 * n:>6} | {fpr:>8.3f} | {row['weak']:>6.3f} "
        f"{row['moderate']:>9.3f} {row['strong']:>7.3f}",
        flush=True,
    )

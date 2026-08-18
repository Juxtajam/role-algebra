"""Exact power of one-sided Fisher test for two-rung contrast, small n."""

import math


def fisher_one_sided_p(a, b, c, d):
    # H1: group2 rate > group1 rate. p = P(X >= c) hypergeom with margins fixed
    # table: group1 successes a of n1=a+b; group2 successes c of n2=c+d
    n1, n2 = a + b, c + d
    k = a + c
    N = n1 + n2
    denom = math.comb(N, k)
    p = 0.0
    for c2 in range(c, min(k, n2) + 1):
        a2 = k - c2
        if 0 <= a2 <= n1:
            p += math.comb(n2, c2) * math.comb(n1, a2) / denom
    return p


def power(n1, p1, n2, p2, alpha=0.05):
    # exact over binomial outcomes
    pw = 0.0
    for a in range(n1 + 1):
        pa = math.comb(n1, a) * p1**a * (1 - p1) ** (n1 - a)
        for c in range(n2 + 1):
            pc = math.comb(n2, c) * p2**c * (1 - p2) ** (n2 - c)
            if fisher_one_sided_p(a, n1 - a, c, n2 - c) <= alpha:
                pw += pa * pc
    return pw


print("Two-rung contrast, one-sided Fisher exact, alpha 0.05")
print("H1: top rung > bottom rung; n per arm, total = 2n")
for p1, p2, label in [
    (0.25, 0.75, "strong (0.25 vs 0.75)"),
    (0.25, 0.50, "moderate (0.25 vs 0.50)"),
    (0.02, 0.50, "T0-pinned vs moderate T3 (0.02 vs 0.50)"),
    (0.02, 0.75, "T0-pinned vs strong T3 (0.02 vs 0.75)"),
]:
    row = []
    for n in (8, 10, 12, 15, 20, 25, 30):
        row.append(f"n={n}({2*n}): {power(n, p1, n, p2):.3f}")
    print(f"  {label}: " + "  ".join(row))
# false positive rate check at p1=p2=0.25
for n in (12, 20, 30):
    print(f"  FPR p=0.25 both, n={n}: {power(n, 0.25, n, 0.25):.4f}")

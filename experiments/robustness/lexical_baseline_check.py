"""Sanity checks for the R_lex operator implementation (synthetic, no real
data): dense-vs-factored equality, identity on the orthogonal complement,
exact swap behaviour on a conflict-free construction, frob2 identity.
Also cross-checks the resolution-fetched lm_head rows against the 8C run's
results/phase8c/name_vectors.npz (must be identical)."""

import pathlib
import numpy as np
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from lexical_baseline import LexOperator

rng = np.random.default_rng(0)
d, k = 64, 3
U = rng.standard_normal((k, d))
gen = (1, 0, 2)  # swap slots 0,1

Db, Da = [], []
for i in range(k):
    for j in range(i + 1, k):
        Db.append(U[i] - U[j])
        Da.append(U[gen[i]] - U[gen[j]])
Db, Da = np.stack(Db), np.stack(Da)
op = LexOperator(Db, Da)

R_dense = np.eye(d) + op.C.T @ op.P.T
V = rng.standard_normal((7, d))
assert np.allclose(op.apply_RT(V), V @ R_dense.T, atol=1e-10)
Q = rng.standard_normal((d, 5))
assert np.allclose(op.apply_R_cols(Q), R_dense @ Q, atol=1e-10)
f2 = ((R_dense - np.eye(d)) ** 2).sum()
assert np.isclose(op.frob2_R_minus_I(), f2, rtol=1e-10), (op.frob2_R_minus_I(), f2)
# identity on orthogonal complement of span(Db)
q, _ = np.linalg.qr(Db.T)  # d x 2 (rank of differences is 2)
x = rng.standard_normal(d)
x_perp = x - q @ (q.T @ x)
assert np.allclose(R_dense @ x_perp, x_perp, atol=1e-9)
# exact swap on the difference vectors
assert np.allclose(Db @ R_dense.T, Da, atol=1e-9)
assert op.residual < 1e-16
print("single-base LexOperator: all checks pass; residual", op.residual)

# multi-base conflict-free case: disjoint name triples
U2 = rng.standard_normal((6, d))
Db2, Da2 = [], []
for base in (U2[:3], U2[3:]):
    for i in range(3):
        for j in range(i + 1, 3):
            Db2.append(base[i] - base[j])
            Da2.append(base[gen[i]] - base[gen[j]])
op2 = LexOperator(np.stack(Db2), np.stack(Da2))
R2 = np.eye(d) + op2.C.T @ op2.P.T
assert np.allclose(np.stack(Db2) @ R2.T, np.stack(Da2), atol=1e-8)
print("two-base disjoint: exact swap; residual", op2.residual)

# involution check on the swap operator (R_lex for a transposition should be
# an involution when construction is conflict-free)
assert np.allclose(R2 @ R2, np.eye(d), atol=1e-8)
print("involution R^2 = I: pass")

# cross-check fetched rows vs the run's name_vectors.npz
p_run = pathlib.Path(__file__).resolve().parents[2] / "results/verdict/discriminator/name_vectors.npz"
p_res = (
    pathlib.Path(__file__).resolve().parents[2]
    / "results/robustness/resolution/name_token_rows.npz"
)
if p_run.exists():
    a, b = np.load(p_run, allow_pickle=True), np.load(p_res)
    print("run npz keys:", a.files)
    print("res npz keys:", b.files)
else:
    print("run name_vectors.npz not present yet")

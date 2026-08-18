"""Phase 8C — fidelity check: the factored (dual-form) implementations in
phase8c_lib are EXACTLY the frozen formulations of
src/shared/discriminator.py (ridge fit, pair error, activation-weighted
group laws, support mass, eff-rank), verified on synthetic data BEFORE any
real fitting. Run: .venv/bin/python phase8c_fidelity.py"""

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from shared import discriminator as fz  # frozen code
import activation_discriminator as lib

rng = np.random.default_rng(0)
n, d = 60, 40
lam = 1e-2
X = rng.standard_normal((n, d))
Y = rng.standard_normal((n, d))

# 1. ridge fit: dense frozen vs dual reconstruction
R_frozen = fz.ridge_fit(X, Y, lam)
dr = lib.DualRidge(X, Y, lam)
R_dual = dr.apply_R_cols(np.eye(d))
err = np.abs(R_frozen - R_dual).max()
print(f"ridge R dense-vs-dual max abs diff: {err:.3e}")
assert err < 1e-8

# apply_RT
V = rng.standard_normal((17, d))
e2 = np.abs(V @ R_frozen.T - dr.apply_RT(V)).max()
print(f"apply_RT diff: {e2:.3e}")
assert e2 < 1e-8

# traces / frobenius
tr_err = abs(np.trace(R_frozen) - dr.trace_R())
fr_err = abs((R_frozen**2).sum() - dr.frob2_R())
fi_err = abs(((R_frozen - np.eye(d)) ** 2).sum() - dr.frob2_R_minus_I())
print(f"trace diff {tr_err:.3e}, frob2 diff {fr_err:.3e}, frob2(R-I) diff {fi_err:.3e}")
assert tr_err < 1e-8 and fr_err < 1e-6 and fi_err < 1e-6

# 2. pair error (frozen formula, dense) vs lib on a mock cell
X2 = rng.standard_normal((n, d))
Y2 = rng.standard_normal((n, d))
dr2 = lib.DualRidge(X2, Y2, lam)
P = dr2.apply_RT(X2)
err_lib = ((P - Y2) ** 2).sum() / ((Y2 - X2) ** 2).sum()
R2 = fz.ridge_fit(X2, Y2, lam)
err_fz = ((X2 @ R2.T - Y2) ** 2).sum() / ((Y2 - X2) ** 2).sum()
print(f"pair_error diff: {abs(err_lib - err_fz):.3e}")
assert abs(err_lib - err_fz) < 1e-10

# 3. group-law metrics: frozen dense formulas vs factored chain
H = rng.standard_normal((33, d))
Ra, Rb = R_frozen, R2
dra, drb = dr, dr2
hn = (H**2).sum()
inv_fz = np.mean([((H @ (R.T @ R.T - np.eye(d))) ** 2).sum() / hn for R in (Ra, Rb)])
A, B = Ra @ Rb @ Ra, Rb @ Ra @ Rb
braid_den = 0.5 * (((H @ A.T) ** 2).sum() + ((H @ B.T) ** 2).sum()) + 1e-12 * hn
braid_fz = ((H @ (A - B).T) ** 2).sum() / braid_den
nontriv_fz = np.mean([((H @ (R - np.eye(d)).T) ** 2).sum() / hn for R in (Ra, Rb)])
noncomm_fz = ((H @ (Ra @ Rb - Rb @ Ra).T) ** 2).sum() / hn


def ap(V, DR):
    return DR.apply_RT(V)


inv_lib = np.mean([((ap(ap(H, DR), DR) - H) ** 2).sum() / hn for DR in (dra, drb)])
HA = ap(ap(ap(H, dra), drb), dra)
HB = ap(ap(ap(H, drb), dra), drb)
braid_lib = ((HA - HB) ** 2).sum() / (
    0.5 * ((HA**2).sum() + (HB**2).sum()) + 1e-12 * hn
)
nontriv_lib = np.mean([((ap(H, DR) - H) ** 2).sum() / hn for DR in (dra, drb)])
HC = ap(ap(H, drb), dra) - ap(ap(H, dra), drb)
noncomm_lib = (HC**2).sum() / hn
for name, a_, b_ in (
    ("inv", inv_fz, inv_lib),
    ("braid", braid_fz, braid_lib),
    ("nontriv", nontriv_fz, nontriv_lib),
    ("noncomm", noncomm_fz, noncomm_lib),
):
    print(f"law {name}: frozen={a_:.10f} lib={b_:.10f} diff={abs(a_-b_):.3e}")
    assert abs(a_ - b_) < 1e-8
# NOTE frozen involution is H @ (R^T R^T - I) = H R^T R^T - H = ap(ap(H,R),R) - H. exact.
# frozen noncommute: H @ (R12@R23 - R23@R12).T = H (R23.T R12.T - R12.T R23.T)
#   = ap(ap(H, R23-first?) ... verify orientation explicitly:
nc_check = np.abs(
    H @ (Ra @ Rb - Rb @ Ra).T - (ap(ap(H, drb), dra) - ap(ap(H, dra), drb))
).max()
print(f"noncommute orientation check: {nc_check:.3e}")
assert nc_check < 1e-8
br_check = np.abs(H @ A.T - ap(ap(ap(H, dra), drb), dra)).max()
print(f"braid orientation check (A=R12R23R12): {br_check:.3e}")
assert br_check < 1e-8

# 4. support mass identity: ||(R-I)q||^2/||R-I||_F^2 equals frozen SVD-weighted sum
U3 = rng.standard_normal((3, d))
diffs = (U3[1:] - U3[:1]).T
q, _ = np.linalg.qr(
    diffs + 1e-12 * np.random.default_rng(0).standard_normal(diffs.shape)
)
_, S, Vt = np.linalg.svd(Ra - np.eye(d))
w = S**2 / (S**2).sum()
mass_fz = (w * ((Vt @ q) ** 2).sum(axis=1)).sum()
mass_lib = ((dr.apply_R_cols(q) - q) ** 2).sum() / dr.frob2_R_minus_I()
print(
    f"support mass: frozen={mass_fz:.10f} lib={mass_lib:.10f} diff={abs(mass_fz-mass_lib):.3e}"
)
assert abs(mass_fz - mass_lib) < 1e-8

# 5. eff-rank of (R - I): frozen entropy-of-SVD vs low-rank eigen route
_, S2, _ = np.linalg.svd(Ra - np.eye(d))
w2 = S2**2 / (S2**2).sum()
p2 = w2 / w2.sum()
er_fz = float(np.exp(-(p2 * np.log(p2 + 1e-30)).sum()))
er_lib = lib.eff_rank_R_minus_I(dr)
print(f"eff_rank: frozen={er_fz:.6f} lib={er_lib:.6f} diff={abs(er_fz-er_lib):.3e}")
assert abs(er_fz - er_lib) < 1e-4

# 6. underdetermined regime (n < d), the 8C case
Xu = rng.standard_normal((30, 200))
Yu = rng.standard_normal((30, 200))
Ru = fz.ridge_fit(Xu, Yu, lam)
dru = lib.DualRidge(Xu, Yu, lam)
eu = np.abs(Ru - dru.apply_R_cols(np.eye(200))).max()
print(f"underdetermined dense-vs-dual: {eu:.3e}")
assert eu < 1e-8

print("FIDELITY: ALL CHECKS PASS — factored implementation == frozen formulation")

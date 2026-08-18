"""Phase 8B — spectrum diagnostic: representation-family characterisation of
fitted generators R_12, R_23 restricted to the identified role subspace.

Item 3:
  - report the spectrum of the fitted R_12, R_23 restricted to the identified
    role subspace;
  - estimate the dimension of the g-invariant subspace within it
    (invariant direction present -> permutation-rep-like / slot-vector style;
     absent -> standard-rep-like);
  - validate on Stage 1 synthetics: S-role vs S-shared must separate.

Also runs the numerical companion to the 8B item-1 derivation: exhibits
rho(g) explicitly for S-shared, verifies the homomorphism property over all
of S_3, and verifies the trivial(1) + standard(k-1) decomposition.

Protocol: all diagnostic thresholds (role-span rank cut, invariant-dim
tolerance) are chosen on the CALIBRATION split distributions only, then
frozen and confirmed once on the test split. Frozen Stage 1 config
(lambda, per-run layer) is reused unchanged from calibration/thresholds.json.
No entry of the frozen thresholds file is modified.
"""

import json
import os
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]  # repository root
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("DV3_RESULTS", str(ROOT / "results"))

from shared import discriminator as disc  # noqa: E402
from synth.model import Frame, perms, compose  # noqa: E402
from synth.organisms import ORGANISMS  # noqa: E402

SNRS = (10, 3, 1)
N_SEEDS = 10
ORGS = ("S-role", "S-shared", "S-retrieval", "S-position")

# Diagnostic parameters — chosen from the calibration-split distributions in
# this script's 'cal' pass, frozen before the single 'test' pass (see report).
RANK_REL_CUT = 0.10  # role-span rank: sigma_i >= RANK_REL_CUT * sigma_1
TAU_INV = 0.30  # invariant dim: singular values of stack(A_g - I) < TAU_INV


# ---------------------------------------------------------------- item 1 check
def homomorphism_check(frame, seed=0):
    """Explicit rho(g) for S-shared on span{w_i}; verify homomorphism and the
    trivial + standard decomposition, numerically."""
    org = ORGANISMS["S-shared"](frame, snr=10, seed=seed)
    w = org.w  # (k, ROLE_DIM), rows w_i
    k = org.k
    assert np.linalg.matrix_rank(w) == k, "w_i not linearly independent"

    # rho(g) on span{w_i}: linear extension of w_i -> w_{g(i)}.
    # In coordinates: for x = c @ w (c row of coefficients),
    # rho(g) x = sum_i c_i w_{g(i)} = (P_g c) @ w with (P_g)[g(i), i] = 1.
    def P(g):
        M = np.zeros((k, k))
        for i in range(k):
            M[g[i], i] = 1.0
        return M

    # rho(g) as a d x d map supported on span{w}: rho = w^+ ... easier to test
    # everything in w-coordinates, where rho(g) IS P(g).
    all_g = perms(k)
    hom_defect = max(
        np.abs(P(compose(a, b)) - P(a) @ P(b)).max() for a in all_g for b in all_g
    )
    # decomposition: ones-vector coordinate direction is fixed by every P(g)
    ones = np.ones(k) / np.sqrt(k)
    inv_defect = max(np.abs(P(g) @ ones - ones).max() for g in all_g)
    # standard rep: mean-zero coordinate subspace, invariant under every P(g),
    # and contains NO further invariant vector (common fixed space of the
    # generators restricted there is 0)
    q, _ = np.linalg.qr(np.hstack([ones[:, None], np.eye(k)[:, : k - 1]]))
    B0 = q[:, 1:]  # basis of mean-zero subspace
    leak = max(np.abs(ones @ P(g) @ B0).max() for g in all_g)  # block off-diag
    gens = org.generators
    K = np.vstack([B0.T @ P(g) @ B0 - np.eye(k - 1) for g in gens])
    s_min = np.linalg.svd(K, compute_uv=False).min()
    return dict(
        k=k,
        w_rank=int(np.linalg.matrix_rank(w)),
        homomorphism_max_defect=float(hom_defect),
        invariant_direction_defect=float(inv_defect),
        block_offdiag_leak=float(leak),
        standard_block_no_fixed_vector_smin=float(s_min),
    )


# ------------------------------------------------------------ the diagnostic
def role_subspace(U, rank_rel_cut=RANK_REL_CUT):
    """U: (k, d) role-conditional mean vectors (ground-truth role blocks mapped
    to state space in Stage 1; slot-conditional activation means minus the
    known content background where available). Returns orthonormal basis B
    (d, r), singular values, and the chosen rank r."""
    _, S, Vt = np.linalg.svd(U, full_matrices=False)
    r = int((S >= rank_rel_cut * max(S[0], 1e-30)).sum())
    return Vt[:r].T, S, r


def spectrum_diagnostic(system, Rs, layer, rank_rel_cut=RANK_REL_CUT, tau_inv=TAU_INV):
    """Spectrum of each fitted generator restricted to the identified role
    subspace + estimated g-invariant subspace dimension within it."""
    eps = [system.orbit(system.bases("P", "fit", "cal")[0])[system.all_perms[0]]]
    U = system.u_vectors(eps, layer)[0]  # (k, d)
    B, S, r = role_subspace(U, rank_rel_cut)
    # scale check: does the role span carry signal at all (controls should not)
    H = system.states(
        [
            ep
            for b in system.bases("P", "fit", "cal")[:8]
            for ep in system.orbit(b).values()
        ],
        layer,
    )
    signal = float(S[0] / (np.linalg.norm(H, axis=1).mean() + 1e-30))
    out = dict(
        role_span_dim=r,
        u_singular_values=[float(x) for x in S],
        role_span_signal=signal,
    )
    if r == 0:
        out.update(invariant_dim=None, spectra={}, inv_singulars=[])
        return out
    As = {a: B.T @ R @ B for a, R in Rs.items()}
    spectra = {}
    for a, A in As.items():
        ev = np.linalg.eigvals(A)
        order = np.argsort(-ev.real)
        spectra[str(a)] = [[float(ev[i].real), float(ev[i].imag)] for i in order]
    K = np.vstack([A - np.eye(r) for A in As.values()])
    s_inv = np.linalg.svd(K, compute_uv=False)
    inv_dim = int((s_inv < tau_inv).sum())
    out.update(
        invariant_dim=inv_dim, spectra=spectra, inv_singulars=[float(x) for x in s_inv]
    )
    return out


# ------------------------------------------------------------------ execution
def main():
    cfg = json.load(open(ROOT / "results/instrument/calibration/thresholds.json"))["stage1"]
    lam = cfg["lambda"]
    frame = Frame()

    results = dict(
        config=dict(
            lam=lam, rank_rel_cut=RANK_REL_CUT, tau_inv=TAU_INV, n_seeds=N_SEEDS
        ),
        homomorphism=homomorphism_check(frame),
        runs={},
    )
    print("item-1 numerical companion:", json.dumps(results["homomorphism"]))

    for phase in ("cal", "test"):
        for name in ORGS:
            for snr in SNRS:
                layer = cfg["runs"][f"{name}@{snr}"]["layer"]
                for seed in range(N_SEEDS):
                    system = ORGANISMS[name](frame, snr, seed=seed)
                    s = disc.build_splits(system, phase)
                    Rs = disc.fit_maps(system, layer, s["fit"], lam)
                    d = spectrum_diagnostic(system, Rs, layer)
                    key = f"{phase}/{name}@{snr}/seed{seed}"
                    results["runs"][key] = d
                    print(
                        f"{key}: span_dim={d['role_span_dim']} "
                        f"inv_dim={d['invariant_dim']} "
                        f"signal={d['role_span_signal']:.3f} "
                        f"inv_sv={['%.3f' % x for x in d['inv_singulars']]}"
                    )

    outp = ROOT / "results/instrument/spectrum/spectrum_validation.json"
    outp.parent.mkdir(parents=True, exist_ok=True)
    json.dump(results, open(outp, "w"), indent=2)
    print("wrote", outp)

    # separation summary (the validation criterion)
    print("\n=== separation summary (test phase) ===")
    for name in ORGS:
        for snr in SNRS:
            keys = [f"test/{name}@{snr}/seed{s}" for s in range(N_SEEDS)]
            dims = [
                (
                    results["runs"][k]["role_span_dim"],
                    results["runs"][k]["invariant_dim"],
                )
                for k in keys
            ]
            uniq = sorted(set(dims))
            print(f"{name}@{snr}: (span_dim, inv_dim) -> {uniq} " f"[{N_SEEDS} seeds]")


if __name__ == "__main__":
    main()

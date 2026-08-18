"""Track B discriminator — synthetic sanity check at d=128 (instrument gate).

Mirrors the 8C fidelity requirement: before any real fit, the exact pipeline
(fit_generators -> metrics -> frozen thresholds -> frozen verdict logic) must
return the correct verdict on planted ground truth at d=128:

  S-role-like:      h(g.x) = rho(g) h(x) + noise, rho = permutation rep of S_3
                    acting on a random 3-dim subspace  -> expect H_role
  S-retrieval-like: h(x) base-specific random states, no shared operator
                    -> expect H_retrieval

Uses the same pairing arithmetic (base-major, PERMS3 order), the same null
modes, the same _quantile_thresholds, and the same evaluate_conditions /
verdict_and_score as the real run. C5/C6 are not exercisable on synthetic
states (no model to decode with); the check covers C1/C2/C3/C4 — the
conditions that carried every verdict in this programme. Verdict logic is
applied to the covered conditions only.
"""

import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import trained_discriminator as dl  # noqa: E402
from shared.discriminator import compose  # noqa: E402
from shared.calibrate import _quantile_thresholds  # noqa: E402

D, K, SNR = 128, 3, 3.0
PERMS = dl.PERMS3
rng = np.random.default_rng(0)


def make_states(kind, n_bases, path_seed):
    """(n_bases*6, d) states, base-major, PERMS3 order."""
    r = np.random.default_rng(path_seed)
    # shared role frame: rho(g) permutes 3 orthonormal directions
    Q, _ = np.linalg.qr(np.random.default_rng(99).standard_normal((D, K)))
    states = np.zeros((n_bases * 6, D), dtype=np.float64)
    for b in range(n_bases):
        content = r.standard_normal(D) * 0.5  # orbit-constant content
        coef = r.standard_normal(K) + 2.0  # role coefficients
        for gi, g in enumerate(PERMS):
            if kind == "role":
                # rho(g) x: coefficient of direction i becomes coef[g^{-1}(i)]
                ginv = {g[j]: j for j in range(K)}
                vec = sum(coef[ginv[i]] * Q[:, i] for i in range(K))
            else:
                vec = r.standard_normal(D) * np.linalg.norm(coef) / np.sqrt(K)
            noise = r.standard_normal(D) / SNR * 0.3
            states[b * 6 + gi] = content + vec + noise
    return states


def run(kind):
    n_cal, n_test = 60, 60
    sP = make_states(kind, n_cal + n_test, path_seed=1)
    sG = make_states(kind, n_cal + n_test, path_seed=2)  # same rho, other path
    sT = make_states(kind, n_cal + n_test, path_seed=3)  # "transfer vocab"
    bases = np.arange(n_cal + n_test)
    cal, test = bases[0::2], bases[1::2]

    def xy(states):
        return lambda bb, a: (
            states[[dl.ep_index(b, g) for b in bb for g in PERMS]],
            states[[dl.ep_index(b, compose(a, g)) for b in bb for g in PERMS]],
        )

    lam = 1e-3
    # nulls on cal -> thresholds (frozen quantile logic)
    nulls = {}
    for mode, seeds in (("shuffled", range(10)), ("identity", [0])):
        for ns in seeds:
            Rn = dl.fit_generators(
                xy(sP), cal, lam, null_mode=mode, null_seed=2000 * ns
            )
            m = dict(
                content_transfer_err=dl.pair_error_mat(Rn, xy(sT), cal),
                crosspath_err=dl.pair_error_mat(Rn, xy(sG), cal),
            )
            H = sT[[dl.ep_index(b, g) for b in cal for g in PERMS]]
            m.update(dl.group_law_metrics_mat(Rn, H))
            src = mode
            for metric, v in m.items():
                need = dict(
                    content_transfer_err="shuffled",
                    crosspath_err="shuffled",
                    law_inv_defect="shuffled",
                    law_braid_defect="shuffled",
                    nontriv="identity",
                    noncommute="identity",
                )[metric]
                if src == need:
                    nulls.setdefault(metric, []).append(float(v))
    taus = _quantile_thresholds(nulls, fpr=0.05)

    Rs = dl.fit_generators(xy(sP), test, lam)
    m = dict(
        content_transfer_err=dl.pair_error_mat(Rs, xy(sT), test),
        crosspath_err=dl.pair_error_mat(Rs, xy(sG), test),
    )
    H = sT[[dl.ep_index(b, g) for b in test for g in PERMS]]
    m.update(dl.group_law_metrics_mat(Rs, H))

    status = {}
    for cond, keys in (
        ("C1", ["content_transfer_err"]),
        ("C2", ["crosspath_err"]),
        ("C3", ["law_inv_defect", "law_braid_defect"]),
        ("C4", ["nontriv", "noncommute"]),
    ):
        ok = all(
            (
                (m[k] < taus[k]["tau"])
                if taus[k]["dir"] == "lt"
                else (m[k] > taus[k]["tau"])
            )
            for k in keys
        )
        status[cond] = "pass" if ok else "fail"
    verdict = "H_retrieval" if "fail" in status.values() else "H_role"
    print(f"{kind:>10}: verdict={verdict} conds={status}")
    print(
        f"           C1={m['content_transfer_err']:.4f} (tau {taus['content_transfer_err']['tau']:.4f})"
        f" C2={m['crosspath_err']:.4f} (tau {taus['crosspath_err']['tau']:.4f})"
        f" inv={m['law_inv_defect']:.2e} nontriv={m['nontriv']:.4f}"
    )
    return verdict


if __name__ == "__main__":
    v1 = run("role")
    v2 = run("retrieval")
    ok = v1 == "H_role" and v2 == "H_retrieval"
    print("SANITY:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)

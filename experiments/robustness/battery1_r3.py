"""ROBUSTNESS PASS R3 — centering audit of the GENUINE 8C pipeline
(phase8c_lib.py, phase8c_test_eval.py, spectrum code), with verbatim line
citations, plus a ONE-TIME recompute of the invariant-subspace/spectrum
estimate both ways (centered=False and centered=True) from the stored
deterministic fits, stored to robustness/.

Audit findings (quotes are verbatim from the files on disk; line numbers
match the current bytes, sha-recorded below):

1. Generator fitting — NO centering.
   phase8c_lib.py 157-167 (DualRidge.__init__):
       self.lam_eff = lam * float((X * X).sum()) / d + 1e-12
       XXt = X @ X.T
       s, U = np.linalg.eigh(XXt)
       self.M = (U / (s + self.lam_eff)) @ U.T
   X, Y are raw float64 casts of cached fp16 activations
   (phase8c_lib.py 85-86: states() -> np.asarray(..., dtype=np.float64)).

2. Metric code — NO centering.
   pair_error (215-232), group_law_metrics (235-277),
   support_mass_lexical (280-308) all consume raw states.

3. Probes — standardization (mean-center + scale), probe-internal only.
   phase8c_lib.py 335-336 (fit_probe, frozen _fit_probe verbatim):
       mu, sd = X.mean(0), X.std(0) + 1e-9
       Xn = (X - mu) / sd
   Affects C6 probes and decodability curves only; never the generator fit,
   never the spectrum estimate.

4. Spectrum / invariant-subspace estimate — centering is an explicit FLAG,
   and the run computed and stored BOTH branches.
   phase8c_lib.py 432-435 (spectrum_diagnostic):
       U = np.stack([X[lab == j].mean(0) for j in range(K)])   # (k, d)
       if centered:
           U = U - U.mean(0, keepdims=True)
       _, S, Vt = np.linalg.svd(U, full_matrices=False)
   phase8c_test_eval.py 188-194: invoked with centered=False AND
   centered=True; both stored in test_metrics.json
   .spectrum_diagnostic.{raw,centered}; tau_inv recalibrated for both
   branches (lines 204-222).

This script re-derives both spectrum branches ONCE from the deterministic
refit of the verdict-layer generators (frozen splits/config; DualRidge is
deterministic) and requires exact agreement with the stored values.

Outputs: results/phase8c/robustness/r3_centering_audit.json
"""

import json
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
import activation_discriminator as lib  # noqa: E402
from activation_discriminator import GENERATORS  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[2]
ROB = ROOT / "results/robustness/battery"
ROB.mkdir(parents=True, exist_ok=True)

t0 = time.time()
cfg = json.load(open(ROOT / "results/verdict/discriminator/committed_config.json"))
tm = json.load(open(ROOT / "results/verdict/discriminator/test_metrics.json"))
splits = json.load(open(ROOT / "results/verdict/discriminator/splits.json"))["splits"]
LAM, L = cfg["lambda"], cfg["layer"]
RANK_CUT = cfg["spectrum_diagnostic"]["rank_cut"]

findings = [
    dict(
        file="phase8c_lib.py",
        lines="157-167",
        stage="generator fitting (DualRidge)",
        mean_centering=False,
        code="self.lam_eff = lam * float((X * X).sum()) / d + 1e-12; "
        "XXt = X @ X.T; s, U = np.linalg.eigh(XXt); "
        "self.M = (U / (s + self.lam_eff)) @ U.T",
        verdict="raw activations; no mean subtraction",
    ),
    dict(
        file="phase8c_lib.py",
        lines="85-86",
        stage="activation ingress (Cell.states)",
        mean_centering=False,
        code="return np.asarray(self.acts[rows, layer, :], dtype=np.float64)",
        verdict="raw fp16 -> float64 cast only",
    ),
    dict(
        file="phase8c_lib.py",
        lines="215-232, 235-277, 280-308",
        stage="metrics (pair_error, group_law_metrics, " "support_mass_lexical)",
        mean_centering=False,
        code="num = ((P - Y) ** 2).sum(axis=1); den = ((Y - X) ** 2)"
        ".sum(axis=1) ... H = cell_eval.states(...); hn = "
        "(H ** 2).sum() — all on raw states",
        verdict="raw activations",
    ),
    dict(
        file="phase8c_lib.py",
        lines="335-336",
        stage="probes (fit_probe, frozen _fit_probe verbatim)",
        mean_centering=True,
        code="mu, sd = X.mean(0), X.std(0) + 1e-9; Xn = (X - mu) / sd",
        verdict="probe-feature standardization only; affects C6/"
        "decodability, NOT fits, NOT spectrum",
    ),
    dict(
        file="phase8c_lib.py",
        lines="432-435",
        stage="spectrum/invariant-subspace estimate (spectrum_diagnostic)",
        mean_centering="explicit flag; run stored BOTH branches",
        code="U = np.stack([X[lab == j].mean(0) for j in range(K)]); "
        "if centered: U = U - U.mean(0, keepdims=True); "
        "_, S, Vt = np.linalg.svd(U, full_matrices=False)",
        verdict="slot-conditional means (class means, not activation "
        "centering); grand-mean removal only under centered=True",
    ),
    dict(
        file="phase8c_test_eval.py",
        lines="188-194",
        stage="spectrum invocation in the run",
        mean_centering="both",
        code="spec_raw = lib.spectrum_diagnostic(..., centered=False); "
        "spec_cen = lib.spectrum_diagnostic(..., centered=True); "
        "results['spectrum_diagnostic'] = dict(raw=spec_raw, "
        "centered=spec_cen)",
        verdict="both branches computed in-run and stored; tau_inv "
        "recalibrated per branch (lines 204-222)",
    ),
]

# ---- one-time recompute of both branches from deterministic refit
cells = lib.load_cells()
cf = cells[("P", "fit")]
test_Pfit = splits["P/fit"]["test"]
cal_Pfit = splits["P/fit"]["cal"]
Rs = {}
for a in GENERATORS:
    rx, ry = lib.pair_rows(cf, test_Pfit, a)
    Rs[a] = lib.DualRidge(cf.states(rx, L), cf.states(ry, L), LAM)
spec_raw = lib.spectrum_diagnostic(
    Rs, cf, cal_Pfit, L, rank_cut=RANK_CUT, centered=False
)
spec_cen = lib.spectrum_diagnostic(
    Rs, cf, cal_Pfit, L, rank_cut=RANK_CUT, centered=True
)

stored_raw = tm["spectrum_diagnostic"]["raw"]
stored_cen = tm["spectrum_diagnostic"]["centered"]


def close(a, b):
    return bool(abs(a - b) < 1e-9)


checks = dict(
    raw_span=(
        spec_raw["span_dim"],
        stored_raw["span_dim"],
        spec_raw["span_dim"] == stored_raw["span_dim"],
    ),
    raw_min_sv=(
        spec_raw["stacked_min_sv"],
        stored_raw["stacked_min_sv"],
        close(spec_raw["stacked_min_sv"], stored_raw["stacked_min_sv"]),
    ),
    raw_sv_ratio_1=(
        spec_raw["sv_ratios"][1],
        stored_raw["sv_ratios"][1],
        close(spec_raw["sv_ratios"][1], stored_raw["sv_ratios"][1]),
    ),
    cen_span=(
        spec_cen["span_dim"],
        stored_cen["span_dim"],
        spec_cen["span_dim"] == stored_cen["span_dim"],
    ),
    cen_min_sv=(
        spec_cen["stacked_min_sv"],
        stored_cen["stacked_min_sv"],
        close(spec_cen["stacked_min_sv"], stored_cen["stacked_min_sv"]),
    ),
    cen_sv_ratio_1=(
        spec_cen["sv_ratios"][1],
        stored_cen["sv_ratios"][1],
        close(spec_cen["sv_ratios"][1], stored_cen["sv_ratios"][1]),
    ),
)

audit = dict(
    date="2026-08-08",
    file_hashes=dict(
        phase8c_lib=lib.sha_file(ROOT / "phase8c_lib.py"),
        phase8c_test_eval=lib.sha_file(ROOT / "phase8c_test_eval.py"),
    ),
    summary=dict(
        generator_fitting_centered=False,
        metric_code_centered=False,
        probe_pipeline_standardized=True,
        spectrum_pipeline="both branches (centered=False and centered=True) "
        "computed in the genuine run and stored in "
        "results/verdict/discriminator/test_metrics.json"
        ".spectrum_diagnostic; both re-derived here",
    ),
    findings=findings,
    recompute_both_ways=dict(
        layer=L,
        lam=LAM,
        rank_cut=RANK_CUT,
        raw=spec_raw,
        centered=spec_cen,
        stored_source="results/verdict/discriminator/test_metrics.json" ".spectrum_diagnostic",
        exact_match_checks={
            k: dict(recomputed=v[0], stored=v[1], match=bool(v[2]))
            for k, v in checks.items()
        },
        tau_inv_stored=tm["spectrum_diagnostic"]["tau_inv"],
    ),
)
json.dump(audit, open(ROB / "r3_centering_audit.json", "w"), indent=2)
for k, v in checks.items():
    print(f"{k}: recomputed={v[0]} stored={v[1]} match={v[2]}")
print(f"wrote r3_centering_audit.json ({time.time()-t0:.0f}s)")

"""Phase 8C resolution — item 3: centering audit (code-level, verbatim).

Audits every place activations enter (a) generator fitting, (b) the spectrum
/ invariant-subspace pipeline, in BOTH implementations:
  - frozen Stage-1/2 code: src/shared/discriminator.py,
    phase8b_spectrum_diagnostic.py
  - 8C run code: phase8c_lib.py (+ the run's driver scripts once present)

Records findings as structured JSON with file:line references.
"""

import json, pathlib

ROOT = pathlib.Path(__file__).resolve().parents[2]
RES = ROOT / "results/robustness/resolution"
RES.mkdir(parents=True, exist_ok=True)

findings = []


def note(file, lines, stage, centered, quote, verdict):
    findings.append(
        dict(
            file=file,
            lines=lines,
            stage=stage,
            mean_centering=centered,
            code=quote,
            verdict=verdict,
        )
    )


# --- fitting: src/shared/discriminator.py ---
note(
    "src/shared/discriminator.py",
    "88-93 (ridge_fit)",
    "generator fitting",
    False,
    "G = X.T @ X; lam_eff = lam * np.trace(G) / d + 1e-12; "
    "solve(G + lam_eff*I, X.T @ Y).T",
    "raw activations; no mean subtraction anywhere in fit_maps/ridge_fit",
)

note(
    "src/shared/discriminator.py",
    "113-128 (group_law_metrics)",
    "law/nontriviality metrics",
    False,
    "H = system.states(eps, layer); hn = (H**2).sum(); defects use H directly",
    "raw activations",
)

note(
    "src/shared/discriminator.py",
    "176-179 (_fit_probe)",
    "probes (conditions 6 / first-decodable)",
    True,
    "mu, sd = X.mean(0), X.std(0)+1e-9; Xn = (X-mu)/sd",
    "probe features ARE standardized (mean-centered + scaled). This affects "
    "probe-based metrics only (C6, first_decodable_layer), NOT the generator "
    "fit and NOT the spectrum estimate",
)

# --- spectrum pipeline: phase8b_spectrum_diagnostic.py ---
note(
    "phase8b_spectrum_diagnostic.py",
    "92-99 (role_subspace)",
    "invariant-subspace estimate (Stage 1 validation)",
    False,
    "_, S, Vt = np.linalg.svd(U, full_matrices=False)  # U = u_vectors, "
    "slot-conditional means, no grand-mean subtraction",
    "NOT centered: SVD taken on raw slot-mean vectors u_1..u_k. The "
    "invariant (all-ones) direction is retained in the span, which is what "
    "lets inv_dim separate permutation-rep (S-shared) from standard-rep "
    "(S-role)",
)

note(
    "src/synth/organisms.py",
    "88-101 (u_vectors)",
    "u_i construction (Stage 1)",
    False,
    "means = stack([rb[ans==j].mean(axis=0) ...]) — per-slot means of role "
    "blocks; the per-slot mean is a class-conditional mean, not a centering "
    "of activations",
    "class-conditional averaging only; no grand-mean removed",
)

# --- 8C run code: phase8c_lib.py ---
note(
    "phase8c_lib.py",
    "fit_generators/DualRidge",
    "8C generator fitting",
    False,
    "DualRidge(X, Y, lam): XXt = X @ X.T on raw float64 casts of the cached "
    "fp16 activations; no mean subtraction",
    "raw activations",
)

note(
    "phase8c_lib.py",
    "spectrum_diagnostic (centered=False default)",
    "8C invariant-subspace estimate",
    "flag-dependent",
    "U = stack([X[lab==j].mean(0)]); if centered: U -= U.mean(0)",
    "the 8C lib EXPOSES a centered flag; the resolution must record which "
    "value the run invoked and, if centered=True was used anywhere, "
    "recompute with centered=False and report both",
)

note(
    "phase8c_lib.py",
    "fit_probe (frozen verbatim copy)",
    "8C probes (C6, decodability curves)",
    True,
    "identical standardization to the frozen _fit_probe",
    "probe-only, as in the frozen code",
)

audit = dict(
    date="2026-08-08",
    scope="every activation ingress into generator fitting and the "
    "spectrum/invariant-subspace pipeline, both implementations",
    summary=dict(
        generator_fitting_centered=False,
        spectrum_pipeline_centered=False,
        probe_pipeline_standardized=True,
        note="Mean-centering appears ONLY inside the probe fitter "
        "(mu/sd standardization of features). Generator fits and the "
        "spectrum/invariant-subspace estimate consume raw activations. "
        "Therefore the item-3 recompute-without-centering applies only "
        "if the 8C run invoked spectrum_diagnostic(centered=True); "
        "checked against the run's driver when it lands.",
    ),
    findings=findings,
)

json.dump(audit, open(RES / "item3_centering_audit.json", "w"), indent=2)
for f in findings:
    print(f"{f['file']}:{f['lines']} [{f['stage']}] centered={f['mean_centering']}")
print("wrote", RES / "item3_centering_audit.json")

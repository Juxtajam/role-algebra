"""Calibration: choose lambda and every numerical threshold at a fixed
false-positive rate (target 0.05) on the calibration split only, then FREEZE
to results/calibration/thresholds.json before any test run (spec, "Verdict
and thresholds"). Overwriting a frozen config is logged as a protocol
violation.

Nulls: shuffled-pair fits, identity-pair fits, and the full pipeline run on
S-retrieval. Stage 2 has no synthetic S-retrieval analog, so its thresholds
use the null sources that exist there (shuffled/identity), stated in the
report.
"""

import time

import numpy as np

from shared import discriminator as disc
from shared import progress
from shared.progress import log

FPR = 0.05
THRESHOLDS_PATH = "calibration/thresholds.json"


def _quantile_thresholds(null_values, fpr=FPR):
    """null_values: {metric: [values]}. lt-metrics pass below the fpr-quantile
    of their null; gt-metrics pass above the (1-fpr)-quantile."""
    out = {}
    for metric, (direction, _) in disc.METRICS.items():
        vals = [v for v in null_values.get(metric, []) if v is not None]
        if not vals:
            continue
        q = fpr if direction == "lt" else 1.0 - fpr
        if metric in disc.LENIENT_QUANTILE:
            q = 1.0 - q
        out[metric] = dict(
            dir=direction, tau=float(np.quantile(vals, q)), null_n=len(vals)
        )
    return out


def _collect(null_values, m, source):
    for metric, (_, sources) in disc.METRICS.items():
        if source in sources and m.get(metric) is not None:
            null_values.setdefault(metric, []).append(m[metric])


def select_layer(system, lam):
    """Verdict/patch layer, frozen from the calibration split: the layer with
    the best calibration content-transfer error, capped at the first layer
    where the answer is linearly decodable (spec, condition 5)."""
    errs = []
    for layer in range(system.n_layers):
        s = disc.build_splits(system, "cal")
        Rs = disc.fit_maps(system, layer, s["fit"], lam)
        errs.append(disc.pair_error(system, Rs, layer, s["transfer"]))
    first_dec = disc.first_decodable_layer(system, system.bases("P", "fit", "cal"))
    layer = min(int(np.argmin(errs)), first_dec)
    return layer, first_dec, [float(e) for e in errs]


def calibrate_stage1(make_system, snrs=(10, 3, 1), null_reps=12, force=False):
    """make_system(name, snr) -> organism. Freezes lambda, thresholds and the
    per-run layer choices for every Stage 1 organism run."""
    if progress.exists(THRESHOLDS_PATH):
        cfg = progress.load_json(THRESHOLDS_PATH)
        if "stage1" in cfg and not force:
            log(
                "calibration: thresholds.json already frozen — loading (pass force=True to redo; that is a protocol violation)"
            )
            return cfg["stage1"]
        if "stage1" in cfg and force:
            log("PROTOCOL VIOLATION: re-calibrating over a frozen thresholds.json")
    cfg = (
        progress.load_json(THRESHOLDS_PATH) if progress.exists(THRESHOLDS_PATH) else {}
    )

    # -- lambda: fixed grid, selected on the calibration split of S-role@SNR3
    lam_scores = []
    for lam in disc.LAM_GRID:
        sys_cal = make_system("S-role", 3)
        errs = []
        for layer in range(sys_cal.n_layers):
            s = disc.build_splits(sys_cal, "cal")
            Rs = disc.fit_maps(sys_cal, layer, s["fit"], lam)
            errs.append(disc.pair_error(sys_cal, Rs, layer, s["transfer"]))
        lam_scores.append(min(errs))
        log(f"calibration: lambda={lam:g} best-layer cal transfer err={min(errs):.4f}")
    lam = disc.LAM_GRID[int(np.argmin(lam_scores))]
    log(f"calibration: frozen lambda={lam:g}")

    # -- nulls (calibration split only)
    null_values = {}
    for rep in range(null_reps):
        for snr in snrs:
            sys_role = make_system("S-role", snr)
            for mode in ("shuffled", "identity"):
                for layer in range(sys_role.n_layers):
                    m = disc.all_metrics(
                        sys_role,
                        layer,
                        "cal",
                        lam,
                        null_mode=mode,
                        null_seed=1000 * rep + layer,
                    )
                    _collect(null_values, m, mode)
        log(f"calibration: null rep {rep + 1}/{null_reps} (shuffled+identity) done")
    for rep in range(max(4, null_reps // 2)):
        for snr in snrs:
            sys_ret = make_system("S-retrieval", snr)
            for layer in range(sys_ret.n_layers):
                m = disc.all_metrics(sys_ret, layer, "cal", lam)
                _collect(null_values, m, "retrieval")
        log(f"calibration: S-retrieval null rep {rep + 1} done")

    thresholds = _quantile_thresholds(null_values)
    for metric, t in thresholds.items():
        log(
            f"calibration: threshold {metric}: {t['dir']} {t['tau']:.4f} (n_null={t['null_n']})"
        )

    # -- per-run frozen layer choices
    runs = {}
    for name in ("S-role", "S-shared", "S-retrieval", "S-position"):
        for snr in snrs:
            system = make_system(name, snr)
            layer, first_dec, errs = select_layer(system, lam)
            runs[f"{name}@{snr}"] = dict(
                layer=layer, first_decodable=first_dec, cal_transfer_err_by_layer=errs
            )
            log(
                f"calibration: {name}@{snr} layer={layer} (first decodable={first_dec})"
            )

    cfg["stage1"] = dict(
        **{"lambda": lam},
        fpr_target=FPR,
        thresholds=thresholds,
        runs=runs,
        frozen_at=time.strftime("%Y-%m-%d %H:%M:%S"),
    )
    progress.save_json(THRESHOLDS_PATH, cfg)
    log("calibration: Stage 1 config FROZEN")
    return cfg["stage1"]


def calibrate_system(system, key, null_reps=10, force=False):
    """Stage 2: freeze thresholds + layer for one trained organism x seed from
    its calibration split, before its test split is touched."""
    cfg = (
        progress.load_json(THRESHOLDS_PATH) if progress.exists(THRESHOLDS_PATH) else {}
    )
    stage2 = cfg.setdefault("stage2", {})
    support_key = (
        "support_mass_lex"
        if getattr(system, "support_test", "role") == "lexical"
        else "support_mass"
    )
    if key in stage2 and not force:
        if support_key in stage2[key]["thresholds"]:
            log(f"calibration[{key}]: already frozen — loading")
            return stage2[key]
        log(
            f"calibration[{key}]: frozen entry predates the {support_key} test — "
            f"re-calibrating before any test evaluation (declared, not silent)"
        )
    if key in stage2 and force:
        log(f"PROTOCOL VIOLATION: re-calibrating frozen stage2 config for {key}")

    lam_scores = []
    for lam in disc.LAM_GRID:
        s = disc.build_splits(system, "cal")
        errs = []
        for layer in range(system.n_layers):
            Rs = disc.fit_maps(system, layer, s["fit"], lam)
            errs.append(disc.pair_error(system, Rs, layer, s["transfer"]))
        lam_scores.append(min(errs))
    lam = disc.LAM_GRID[int(np.argmin(lam_scores))]

    null_values = {}
    for rep in range(null_reps):
        for mode in ("shuffled", "identity"):
            for layer in range(system.n_layers):
                m = disc.all_metrics(
                    system,
                    layer,
                    "cal",
                    lam,
                    null_mode=mode,
                    null_seed=2000 * rep + layer,
                )
                _collect(null_values, m, mode)
        log(f"calibration[{key}]: null rep {rep + 1}/{null_reps} done")
    thresholds = _quantile_thresholds(null_values)

    layer, first_dec, errs = select_layer(system, lam)
    stage2[key] = dict(
        **{"lambda": lam},
        fpr_target=FPR,
        thresholds=thresholds,
        layer=layer,
        first_decodable=first_dec,
        cal_transfer_err_by_layer=errs,
        frozen_at=time.strftime("%Y-%m-%d %H:%M:%S"),
    )
    progress.save_json(THRESHOLDS_PATH, cfg)
    log(f"calibration[{key}]: FROZEN (lambda={lam:g}, layer={layer})")
    return stage2[key]

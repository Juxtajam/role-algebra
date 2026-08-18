"""Phase 8C supplementary robustness — R3-samepath refit.

Fit generators on the G-PATH CALIBRATION split (G/fit cal bases,
splits.json), evaluate pair_error on the G-PATH TEST split (G/fit test
bases) — identical machinery to the genuine 8C run: phase8c_lib.py
dual-form ridge (frozen scale-free lam_eff), frozen lambda grid
(committed_config.json .lam_grid) with the frozen committed lambda = 1.0
as the primary column, the frozen 12-layer reporting set, matched-regime
nulls (shuffled x10, frozen seed rule 2000*rep+layer, + identity;
identical sample count and lambda as the samepath real fit), frozen pooled
quantile threshold logic (crosspath_err rule: lt, shuffled source,
FPR 0.05; phase8c_freeze_config.py lines 42-69).

Side-by-side target: the genuine C2 (fit-P/eval-G) stored values,
results/phase8c/test_metrics.json .per_layer[L].{crosspath_err,
crosspath_ci} (fit on P/fit TEST bases per phase8c_test_eval.py line 84).

DECLARED regime notes (raw facts, no interpretation):
  * G/fit cal has 145 bases -> 870 pairs/generator (the P-path fits used
    150 bases -> 900); the samepath nulls use the identical 870 to match
    the samepath real fit.
  * The genuine C2 fit uses P/fit TEST bases; the samepath fit uses G/fit
    CAL bases per the issuing instruction (keeps the G/fit TEST eval bases
    disjoint from the fit bases).
  * The genuine cal-split nulls were evaluated on cal-split bases; here
    the samepath nulls are fit on G/fit CAL and evaluated on G/fit TEST
    (the same eval target as the samepath real fit; cal-eval would be
    in-sample for a samepath null).

Bootstrap CI: phase8c_test_eval.py boot_ratio verbatim (seed lib.BOOT_SEED,
10k resamples over bases). CPU/numpy float64. Writes
results/phase8c/robustness/r3_samepath_refit.json AS COMPUTED.
"""

import hashlib
import json
import time

import numpy as np

import activation_discriminator as lib
from activation_discriminator import GENERATORS

t0 = time.time()

cfg_txt = (lib.OUT / "committed_config.json").read_text()
h = hashlib.sha256(cfg_txt.encode()).hexdigest()
assert h == (lib.OUT / "committed_config.sha256").read_text().strip()
cfg = json.loads(cfg_txt)
LAYER_SET = cfg["reporting_layer_set"]
LAM = cfg["lambda"]
LAM_GRID = cfg["lam_grid"]
FPR = cfg["fpr_target"]
print(
    f"config verified sha256={h[:16]}...; lambda={LAM:g} "
    f"grid={LAM_GRID} layers={LAYER_SET}"
)

splits = json.load(open(lib.OUT / "splits.json"))["splits"]
cal_g = splits["G/fit"]["cal"]
test_g = splits["G/fit"]["test"]
cg = lib.Cell("G", "fit")
tm = json.load(open(lib.OUT / "test_metrics.json"))


def boot_ratio(nums, dens, seed=lib.BOOT_SEED, n_boot=lib.N_BOOT):
    """phase8c_test_eval.py lines 63-71 verbatim."""
    nums, dens = np.asarray(nums), np.asarray(dens)
    nb = nums.shape[1]
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, nb, size=(n_boot, nb))
    stats = np.mean(
        nums[:, idx].sum(axis=2) / np.maximum(dens[:, idx].sum(axis=2), 1e-12), axis=0
    )
    return [float(np.percentile(stats, q)) for q in (2.5, 97.5)]


results = dict(
    config_sha256=h,
    what=(
        "R3-samepath: generators fit on G/fit CAL bases, pair_error "
        "evaluated on G/fit TEST bases; frozen dual-form ridge, frozen "
        "lambda/grid/layers; matched-regime nulls + frozen pooled "
        "quantile rule; side-by-side with stored C2 fit-P/eval-G"
    ),
    fit_bases="G/fit cal (145 bases, 870 pairs/generator) — splits.json",
    eval_bases="G/fit test (145 bases, 870 pairs/generator) — splits.json",
    regime_notes=[
        "870 pairs/generator here vs 900 in the P-path fits (G/fit "
        "strict-orbit universe is 290 bases); nulls matched to 870",
        "genuine C2 fits on P/fit TEST bases (phase8c_test_eval.py line "
        "84); samepath fits on G/fit CAL bases per issuing instruction",
        "samepath nulls fit on G/fit CAL, evaluated on G/fit TEST (same "
        "eval target as the samepath real fit)",
    ],
    lam_primary=LAM,
    lam_grid=LAM_GRID,
    null_rule=(
        "shuffled x10 (seed 2000*rep+layer) + identity, fit on "
        "G/fit cal at frozen lambda; crosspath_err threshold rule "
        "(lt, shuffled source, FPR 0.05, pooled over the 12 frozen "
        "layers) per phase8c_freeze_config.py lines 42-69"
    ),
    per_layer={},
    null_values={"shuffled": {}, "identity": {}},
)

null_pool_shuffled = []
for layer in LAYER_SET:
    tl = time.time()
    lay = {}
    # ---- real samepath fits across the frozen lambda grid ----
    grid_out = {}
    for lam in LAM_GRID:
        Rs = lib.fit_generators(cg, cal_g, layer, lam)
        err, pb = lib.pair_error(Rs, cg, test_g, layer, per_base=True)
        e = dict(err=float(err))
        if lam == LAM:
            e["ci"] = boot_ratio([p["num"] for p in pb], [p["den"] for p in pb])
            e["conditioning"] = {
                lib.GEN_NAMES[GENERATORS.index(a)]: dict(
                    n_pairs=Rs[a].n,
                    lam_eff=Rs[a].lam_eff,
                    cond_regularised=Rs[a].cond_eff,
                )
                for a in GENERATORS
            }
        grid_out[f"{lam:g}"] = e
    lay["samepath_fitG_evalG"] = grid_out[f"{LAM:g}"]
    lay["samepath_lambda_grid"] = {k: v["err"] for k, v in grid_out.items()}
    # ---- stored C2 fit-P/eval-G (quoted, with path) ----
    stored = tm["per_layer"][str(layer)]
    lay["stored_C2_fitP_evalG"] = dict(
        err=stored["crosspath_err"],
        ci=stored["crosspath_ci"],
        path=(
            f"results/verdict/discriminator/test_metrics.json .per_layer.{layer}"
            ".crosspath_err/.crosspath_ci"
        ),
    )
    # ---- matched-regime nulls at frozen lambda ----
    shuf = []
    for rep in range(10):
        seed = 2000 * rep + layer
        Rn = lib.fit_generators(
            cg, cal_g, layer, LAM, null_mode="shuffled", null_seed=seed
        )
        shuf.append(float(lib.pair_error(Rn, cg, test_g, layer)))
    Ri = lib.fit_generators(cg, cal_g, layer, LAM, null_mode="identity")
    ident = float(lib.pair_error(Ri, cg, test_g, layer))
    results["null_values"]["shuffled"][str(layer)] = shuf
    results["null_values"]["identity"][str(layer)] = ident
    null_pool_shuffled.extend(shuf)
    lay["null_shuffled_min"] = float(min(shuf))
    lay["null_identity"] = ident
    results["per_layer"][str(layer)] = lay
    print(
        f"L{layer}: samepath={lay['samepath_fitG_evalG']['err']:.4f} "
        f"storedC2={stored['crosspath_err']:.4f} "
        f"null_shuf_min={min(shuf):.4f} ident={ident:.4f} "
        f"({time.time()-tl:.0f}s)",
        flush=True,
    )

tau = float(np.quantile(null_pool_shuffled, FPR))  # lt, not LENIENT
tau_bonf = float(np.quantile(null_pool_shuffled, FPR / len(LAYER_SET)))
results["threshold_samepath"] = dict(
    dir="lt",
    tau=tau,
    tau_bonferroni=tau_bonf,
    null_n=len(null_pool_shuffled),
    rule=(
        "0.05 quantile (lt) of pooled shuffled samepath nulls over the "
        "12 frozen layers — frozen crosspath_err quantile logic applied "
        "to the samepath regime"
    ),
)
results["stored_C2_threshold"] = dict(
    tau=cfg["thresholds"]["crosspath_err"]["tau"],
    tau_bonferroni=cfg["thresholds_bonferroni"]["crosspath_err"]["tau"],
    path=(
        "results/verdict/discriminator/committed_config.json .thresholds.crosspath_err"
        " / .thresholds_bonferroni.crosspath_err (P-regime; quoted for "
        "reference, not applied to the samepath values)"
    ),
)
results["wall_s"] = round(time.time() - t0, 1)
p = lib.OUT / "robustness" / "r3_samepath_refit.json"
json.dump(results, open(p, "w"), indent=2)
print(
    f"samepath tau (pooled shuffled, lt) = {tau:.6f} "
    f"(bonf {tau_bonf:.6f}, n={len(null_pool_shuffled)})"
)
print(f"wrote {p} ({time.time()-t0:.0f}s)")

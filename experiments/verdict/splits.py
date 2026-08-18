"""Phase 8C — Step 1: SPLITS. Defined over BASE PROBLEMS, per cell, before
any fitting, curve, or threshold work. Documented in the committed config.

Rules (frozen here):
  * Universe per cell = the 300 disc bases, FILTERED to strict-orbit-correct
    bases (all 6 episodes answered correctly in the 8A-final behavioural
    labelling, preds_disc_*.json) — the pre-registered 8C filtering use of
    those labels (phase8a_final_gate.md section 8).
  * Deterministic split: rng = default_rng((20260807, path=='G',
    vocab=='transfer', 8)) permutes the strict-ok base list; calibration =
    first floor(n/2), test = next floor(n/2); any odd remainder base is
    DROPPED (declared) so calibration and test have identical base counts —
    required for the matched-regime null calibration ( instruction A).
  * The test half is not read by any calibration/threshold code.
"""

import json

import numpy as np

import activation_discriminator as lib

OUT = lib.OUT
OUT.mkdir(parents=True, exist_ok=True)

cells = lib.load_cells()
splits = {}
for (p, v), cell in cells.items():
    ok = cell.strict_ok
    rng = np.random.default_rng((20260807, p == "G", v == "transfer", 8))
    perm = rng.permutation(len(ok))
    half = len(ok) // 2
    cal = sorted(int(ok[i]) for i in perm[:half])
    test = sorted(int(ok[i]) for i in perm[half : 2 * half])
    dropped = [int(ok[i]) for i in perm[2 * half :]]
    splits[f"{p}/{v}"] = dict(
        n_bases_total=cell.n_bases,
        n_strict_ok=len(ok),
        n_cal=len(cal),
        n_test=len(test),
        dropped_odd=dropped,
        cal=cal,
        test=test,
    )
    print(
        f"{p}/{v}: strict-ok {len(ok)}/300 -> cal {len(cal)} / test "
        f"{len(test)} / dropped {dropped}"
    )

doc = dict(
    phase="8C",
    defined_before="any decodability curve, fit, lambda/layer selection, or threshold work",
    universe="disc_{path}_{vocab} base problems (300 per cell)",
    filter="strict-orbit-correct bases only (preds_disc_*.json, 8A-final session labels)",
    split_rule=(
        "default_rng((20260807, path=='G', vocab=='transfer', 8)) "
        "permutation; cal = first floor(n/2), test = next floor(n/2), "
        "odd remainder dropped (declared) for matched cal/test counts"
    ),
    splits=splits,
)
json.dump(doc, open(OUT / "splits.json", "w"), indent=2)
print("written", OUT / "splits.json")

# R5 — overlap: behaviourally flagged bases vs per-base condition outliers

Sources: results/phase8c/test_per_base.json, results/phase8c_resolution/item5_behavioural_bases.json, results/phase8c/splits.json. Rule: outlier if per-base pooled-generator err > Q3 + 1.5*IQR (declared before reading values).

| cell (condition) | n test bases | outlier thr | outlier bases (err) | flagged bases in cell | flagged in test split | overlap |
|---|---|---|---|---|---|---|
| C1_content_transfer__P/transfer_test | 150 | 2.098 | 237(3.236); 76(2.591); 101(2.172); 231(2.149); 297(2.146) | none |  | EMPTY |
| C2_crosspath__G/fit_test | 145 | 10.71 | 196(14.820); 53(13.830); 197(12.548); 226(11.874); 81(11.564); 199(10.874) | [16, 44, 60, 68, 78, 211, 232, 259, 296] |  | EMPTY |
| C3_law_inv__P/transfer_test | 150 | 0.0469 | 233(0.057); 95(0.051); 122(0.050); 240(0.049); 250(0.048); 244(0.048) | none |  | EMPTY |

Flagged-base disposition (strict-orbit filter, phase8c_splits.py lines 5-8, excludes every flagged disc base from cal/test):

| disc cell | flagged | in cal | in test | dropped-odd |
|---|---|---|---|---|
| disc_P_fit | none |  |  |  |
| disc_P_transfer | none |  |  |  |
| disc_G_fit | [16, 44, 60, 68, 78, 211, 232, 259, 296] |  |  |  |
| disc_G_transfer | [38, 96, 160, 164, 243] |  |  |  |

Base 57: gate_G_fit (all-6-wrong, orbit-CONSISTENT slot pattern [2, 2, 2, 2, 2, 2]); gate-set base, not in any disc split, no per-base condition record exists for it.

Artifacts: results/phase8c/robustness/base_overlap_table.json (+ this file); mirrored to dv3-results:phase8c/robustness/.

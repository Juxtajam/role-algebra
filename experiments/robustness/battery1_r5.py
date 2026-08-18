"""ROBUSTNESS PASS R5 — overlap table: behaviourally flagged bases
(base 57 / inconsistent-G-orbit bases, results/phase8c_resolution/
item5_behavioural_bases.json) vs per-base condition outliers in the stored
verdict-layer decompositions (results/phase8c/test_per_base.json).

Per-base statistic (from the stored decompositions, no refitting):
  err_b = sum_g num_{g,b} / sum_g den_{g,b}  (both generators pooled)
for content_transfer (P/transfer TEST bases) and crosspath (G/fit TEST
bases); law defects analogous with hn_b denominators.
Outlier rule (declared here, before values are read): err_b > Q3 + 1.5*IQR
of the cell's per-base distribution; the top-5 bases by value are listed
regardless.

Overlap logic: the split universe was FILTERED to strict-orbit-correct bases
(phase8c_splits.py lines 5-8), so every behaviourally flagged disc base
(n_errors >= 1) is excluded from cal/test by construction; this script
verifies that emptiness from the stored bytes rather than asserting it.
Base 57 is a GATE_G_fit base (gate set) — gate bases are not in the disc
splits at all; recorded in the table.

Output: results/phase8c/robustness/base_overlap_table.json
        results/phase8c/robustness/base_overlap_table.md
"""

import json
import pathlib

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
ROB = ROOT / "results/robustness/battery"
ROB.mkdir(parents=True, exist_ok=True)

beh = json.load(
    open(ROOT / "results/robustness/resolution/" "item5_behavioural_bases.json")
)
pb = json.load(open(ROOT / "results/verdict/discriminator/test_per_base.json"))
splits = json.load(open(ROOT / "results/verdict/discriminator/splits.json"))["splits"]

flagged = {cell: sorted(int(b) for b in d["flagged"]) for cell, d in beh.items()}


def per_base_err(entries, test_bases):
    num = np.sum([e["num"] for e in entries], axis=0)
    den = np.sum([e["den"] for e in entries], axis=0)
    err = num / np.maximum(den, 1e-12)
    assert len(err) == len(test_bases)
    return dict(zip(test_bases, err.astype(float)))


def outliers(errs):
    v = np.array(list(errs.values()))
    q1, q3 = np.percentile(v, 25), np.percentile(v, 75)
    thr = float(q3 + 1.5 * (q3 - q1))
    out = sorted([b for b, e in errs.items() if e > thr], key=lambda b: -errs[b])
    top5 = sorted(errs, key=lambda b: -errs[b])[:5]
    return thr, out, top5


table = dict(
    outlier_rule="err_b > Q3 + 1.5*IQR of per-base pooled-generator "
    "num/den ratio at the verdict layer (L61); top-5 listed "
    "regardless",
    sources=dict(
        per_base="results/verdict/discriminator/test_per_base.json",
        behavioural="results/robustness/resolution/item5_behavioural_bases.json",
        splits="results/verdict/discriminator/splits.json",
    ),
    cells={},
)

# ---- C1: content_transfer on P/transfer test
errs = per_base_err(pb["content_transfer"], splits["P/transfer"]["test"])
thr, outl, top5 = outliers(errs)
table["cells"]["C1_content_transfer__P/transfer_test"] = dict(
    n_test_bases=len(errs),
    outlier_threshold=thr,
    outlier_bases={str(b): errs[b] for b in outl},
    top5={str(b): errs[b] for b in top5},
    flagged_bases_in_cell=flagged["disc_P_transfer"],
    flagged_in_test_split=sorted(
        set(flagged["disc_P_transfer"]) & set(splits["P/transfer"]["test"])
    ),
    overlap_flagged_x_outliers=sorted(set(flagged["disc_P_transfer"]) & set(outl)),
)

# ---- C2: crosspath on G/fit test
errs_g = per_base_err(pb["crosspath"], splits["G/fit"]["test"])
thr_g, outl_g, top5_g = outliers(errs_g)
table["cells"]["C2_crosspath__G/fit_test"] = dict(
    n_test_bases=len(errs_g),
    outlier_threshold=thr_g,
    outlier_bases={str(b): errs_g[b] for b in outl_g},
    top5={str(b): errs_g[b] for b in top5_g},
    flagged_bases_in_cell=flagged["disc_G_fit"],
    flagged_in_test_split=sorted(
        set(flagged["disc_G_fit"]) & set(splits["G/fit"]["test"])
    ),
    flagged_in_cal_split=sorted(
        set(flagged["disc_G_fit"]) & set(splits["G/fit"]["cal"])
    ),
    overlap_flagged_x_outliers=sorted(set(flagged["disc_G_fit"]) & set(outl_g)),
)

# ---- C3: involution defect on P/transfer test (num=inv, den=hn)
inv_num = np.sum([pb["laws"]["inv"][i] for i in range(2)], axis=0)
hn = np.array(pb["laws"]["hn"]) * 2.0
errs_l = dict(
    zip(splits["P/transfer"]["test"], (inv_num / np.maximum(hn, 1e-12)).astype(float))
)
thr_l, outl_l, top5_l = outliers(errs_l)
table["cells"]["C3_law_inv__P/transfer_test"] = dict(
    n_test_bases=len(errs_l),
    outlier_threshold=thr_l,
    outlier_bases={str(b): errs_l[b] for b in outl_l},
    top5={str(b): errs_l[b] for b in top5_l},
    flagged_bases_in_cell=flagged["disc_P_transfer"],
    overlap_flagged_x_outliers=sorted(set(flagged["disc_P_transfer"]) & set(outl_l)),
)

# ---- flagged-base disposition (all disc cells)
dispo = {}
for cell_key, split_key in (
    ("disc_P_fit", "P/fit"),
    ("disc_P_transfer", "P/transfer"),
    ("disc_G_fit", "G/fit"),
    ("disc_G_transfer", "G/transfer"),
):
    fl = flagged[cell_key]
    dispo[cell_key] = dict(
        flagged=fl,
        in_cal=sorted(set(fl) & set(splits[split_key]["cal"])),
        in_test=sorted(set(fl) & set(splits[split_key]["test"])),
        dropped_odd=sorted(set(fl) & set(splits[split_key]["dropped_odd"])),
    )
table["flagged_base_disposition"] = dispo
table["base_57_note"] = dict(
    cell="gate_G_fit",
    record=beh["gate_G_fit"]["flagged"].get("57"),
    disposition="gate-set base; gate bases are not part of the disc splits "
    "and produce no per-base condition records in "
    "test_per_base.json",
)

json.dump(table, open(ROB / "base_overlap_table.json", "w"), indent=2)

# ---- markdown table
lines = [
    "# R5 — overlap: behaviourally flagged bases vs per-base condition " "outliers",
    "",
    "Sources: results/phase8c/test_per_base.json, "
    "results/robustness/resolution/item5_behavioural_bases.json, "
    "results/verdict/discriminator/splits.json. Rule: outlier if per-base pooled-"
    "generator err > Q3 + 1.5*IQR (declared before reading values).",
    "",
    "| cell (condition) | n test bases | outlier thr | outlier bases "
    "(err) | flagged bases in cell | flagged in test split | overlap |",
    "|---|---|---|---|---|---|---|",
]
for key, c in table["cells"].items():
    ob = "; ".join(f"{b}({v:.3f})" for b, v in c["outlier_bases"].items()) or "none"
    lines.append(
        f"| {key} | {c['n_test_bases']} | {c['outlier_threshold']:.4g} "
        f"| {ob} | {c['flagged_bases_in_cell'] or 'none'} "
        f"| {c.get('flagged_in_test_split', [])} "
        f"| {c['overlap_flagged_x_outliers'] or 'EMPTY'} |"
    )
lines += [
    "",
    "Flagged-base disposition (strict-orbit filter, phase8c_splits.py "
    "lines 5-8, excludes every flagged disc base from cal/test):",
    "",
    "| disc cell | flagged | in cal | in test | dropped-odd |",
    "|---|---|---|---|---|",
]
for cell_key, d in dispo.items():
    lines.append(
        f"| {cell_key} | {d['flagged'] or 'none'} | {d['in_cal']} "
        f"| {d['in_test']} | {d['dropped_odd']} |"
    )
lines += [
    "",
    f"Base 57: {table['base_57_note']['cell']} "
    f"(all-6-wrong, orbit-CONSISTENT slot pattern "
    f"{table['base_57_note']['record']['pred_slots']}); gate-set base, "
    "not in any disc split, no per-base condition record exists for it.",
    "",
    "Artifacts: results/phase8c/robustness/base_overlap_table.json (+ this "
    "file); mirrored to dv3-results:phase8c/robustness/.",
]
(ROB / "base_overlap_table.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
print("\nwrote", ROB / "base_overlap_table.json", "and .md")

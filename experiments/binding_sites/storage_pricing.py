"""Phase 9 STAGE 1 (local prep) — Task 5: storage + session pricing.

Expected cache bytes = sum over episodes of (positions_per_episode) x
(n_layers in the pre-registered set) x d_model x 2 bytes (fp16), for BOTH
sets (frozen + joint). Position counts read from position_manifest.json,
layer count from layer_set.json. Rates: $0.09/GB-month volume storage,
$5.00/h for 2x A100-80GB (8A-final rate, modal.com/pricing 2026-08-07).

Session-hours estimate anchored to measured 8A-final timings
(results/phase8a_final/session_meta.json + gate_results.json):
model load 906 s; greedy gate generation ~45 s per 600-episode cell;
caching forward passes ~17 min for 9600 episodes (batch 8).

Writes results/phase9/storage_pricing.json.
"""

import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[2]
P9 = ROOT / "results/binding_sites"

D_MODEL = 8192
BYTES_FP16 = 2
USD_PER_GB_MONTH = 0.09
USD_PER_HOUR = 5.00  # 2x A100-80GB @ $2.50/h each

pm = json.load(open(P9 / "position_manifest.json"))
ls = json.load(open(P9 / "layer_set.json"))
n_layers = ls["count"]

sets = {}
total_bytes = 0
for set_name, s in pm["sets"].items():
    b = s["positions"] * n_layers * D_MODEL * BYTES_FP16
    sets[set_name] = dict(
        episodes=s["episodes"],
        positions=s["positions"],
        bytes=b,
        gb=round(b / 1e9, 3),
        gib=round(b / 2**30, 3),
    )
    total_bytes += b

gb = total_bytes / 1e9
usd_month = gb * USD_PER_GB_MONTH

# ---- session estimate (9A: joint gate first, then caching both sets) ----
model_load_s = 906  # measured, 8A-final
gate_cells = 4  # joint set, 900 eps per cell
gate_s = gate_cells * 900 / 600 * 45  # measured 45 s / 600-ep cell
n_eps_total = pm["total_episodes"]  # 13200 (9600 frozen + 3600 joint)
cache_s = n_eps_total / 9600 * 17 * 60  # measured ~17 min / 9600 eps
overhead_s = 15 * 60  # hash verify, IO, checksums,
# volume commits, re-read verify
est_s = model_load_s + gate_s + cache_s + overhead_s
est_h = est_s / 3600
# buffered estimate for the launch decision (x1.5)
est_h_buf = est_h * 1.5

rec = dict(
    phase="9-stage1-prep",
    inputs=dict(
        position_manifest="results/binding_sites/position_manifest.json",
        layer_set="results/binding_sites/layer_set.json",
    ),
    d_model=D_MODEL,
    act_dtype="float16",
    bytes_per_value=BYTES_FP16,
    n_layers_cached=n_layers,
    formula="positions x layers x 8192 x 2 bytes",
    sets=sets,
    total_bytes=total_bytes,
    total_gb=round(gb, 3),
    total_gib=round(total_bytes / 2**30, 3),
    usd_per_gb_month=USD_PER_GB_MONTH,
    storage_usd_per_month=round(usd_month, 2),
    comparison_8c=dict(
        note=(
            "8A-final answer-position cache: 9600 eps x 80 layers x 8192 "
            "x 2 = 12.6 GB; this cache trades all-80-layers for the "
            "22-layer pre-registered set across 10-14x the positions"
        ),
        bytes_8a_final=9600 * 80 * 8192 * 2,
    ),
    session_estimate=dict(
        gpu="A100-80GB:2",
        usd_per_hour=USD_PER_HOUR,
        anchors="results/verdict/answer_position/session_meta.json (load 906 s), "
        "results/verdict/answer_position/gate_results.json (~45 s/600-ep "
        "gate cell), 8A-final caching ~17 min/9600 eps",
        components_s=dict(
            model_load=round(model_load_s),
            joint_gate=round(gate_s),
            caching_both_sets=round(cache_s),
            overhead=round(overhead_s),
        ),
        est_hours=round(est_h, 2),
        est_usd=round(est_h * USD_PER_HOUR, 2),
        est_hours_buffered_1_5x=round(est_h_buf, 2),
        est_usd_buffered_1_5x=round(est_h_buf * USD_PER_HOUR, 2),
    ),
)
json.dump(rec, open(P9 / "storage_pricing.json", "w"), indent=2)
print(
    json.dumps(
        {k: rec[k] for k in ("total_gb", "storage_usd_per_month", "session_estimate")},
        indent=2,
    )
)
print("written:", P9 / "storage_pricing.json")

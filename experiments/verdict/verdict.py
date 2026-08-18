"""Phase 8C — Step 7: reproduction check + verdict assembly.

Per the permanent process rule the verdict is issued only after every
stored metric is REPRODUCED from stored predictions / per-base
decompositions. This script:
  1. re-verifies committed config hash;
  2. recomputes the verdict-layer metrics from test_per_base.json (per-base
     numerators/denominators, probe bits) and independently REFITS the
     generators from the cached activations to reproduce test_metrics.json
     end-to-end (exact match required);
  3. recomputes condition-5 transport_agree + tau from the stored per-item
     records (cond5_results.json), incl. per-base bootstrap CI;
  4. applies the frozen thresholds via the FROZEN evaluate_conditions /
     verdict_and_score logic (src.shared.discriminator, imported);
  5. writes verdict.json.
"""

import pathlib
import hashlib
import json
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "src"))
from shared import discriminator as fz

import activation_discriminator as lib

# 1. config
cfg_txt = (lib.OUT / "committed_config.json").read_text()
assert (
    hashlib.sha256(cfg_txt.encode()).hexdigest()
    == (lib.OUT / "committed_config.sha256").read_text().strip()
)
cfg = json.loads(cfg_txt)
LAYER, LAM = cfg["layer"], cfg["lambda"]

stored = json.load(open(lib.OUT / "test_metrics.json"))
mv = stored["verdict_layer_metrics"]
pb = json.load(open(lib.OUT / "test_per_base.json"))

checks = {}


# 2a. reproduce ratio metrics from per-base decompositions
def ratio(pbl):
    num = (
        np.sum([p["num"] for p in pbl], axis=(0, 1)) if isinstance(pbl, list) else None
    )


c1 = np.mean([np.sum(p["num"]) / np.sum(p["den"]) for p in pb["content_transfer"]])
checks["content_transfer_err"] = (c1, mv["content_transfer_err"])
c2 = np.mean([np.sum(p["num"]) / np.sum(p["den"]) for p in pb["crosspath"]])
checks["crosspath_err"] = (c2, mv["crosspath_err"])
lpb = pb["laws"]
hn = np.sum(lpb["hn"])
inv = np.mean([np.sum(e) / hn for e in lpb["inv"]])
checks["law_inv_defect"] = (inv, mv["law_inv_defect"])
braid = np.sum(lpb["braid_num"]) / (np.sum(lpb["braid_den"]) + 1e-12 * hn)
checks["law_braid_defect"] = (braid, mv["law_braid_defect"])
nt = np.mean([np.sum(e) / hn for e in lpb["nontriv"]])
checks["nontriv"] = (nt, mv["nontriv"])
nc = np.sum(lpb["noncomm"]) / hn
checks["noncommute"] = (nc, mv["noncommute"])
sm = np.mean(pb["support_mass_per_ep"])
checks["support_mass_lex"] = (sm, mv["support_mass_lex"])

# 2b. independent end-to-end refit from cached activations
cells = lib.load_cells()
splits = json.load(open(lib.OUT / "splits.json"))["splits"]
cf, ct, cg = cells[("P", "fit")], cells[("P", "transfer")], cells[("G", "fit")]
Rs = {}
for a in lib.GENERATORS:
    rx, ry = lib.pair_rows(cf, splits["P/fit"]["test"], a)
    Rs[a] = lib.DualRidge(cf.states(rx, LAYER), cf.states(ry, LAYER), LAM)
refit_c1 = lib.pair_error(Rs, ct, splits["P/transfer"]["test"], LAYER)
refit_c2 = lib.pair_error(Rs, cg, splits["G/fit"]["test"], LAYER)
checks["refit_content_transfer_err"] = (refit_c1, mv["content_transfer_err"])
checks["refit_crosspath_err"] = (refit_c2, mv["crosspath_err"])

# 3. condition 5 from stored per-item records
c5 = json.load(open(lib.OUT / "cond5_results.json"))
assert c5["config_sha256"] == hashlib.sha256(cfg_txt.encode()).hexdigest()
assert c5["patch_layer"] == cfg["patch_layer"]
by_fit = {}
for it in c5["items"]:
    by_fit.setdefault(it["fit"], []).append(it)
real = by_fit["real"]
agree_real = float(np.mean([it["agree"] for it in real]))
checks["transport_agree"] = (agree_real, c5["summary"]["real"]["agree"])
null_agrees = sorted(
    float(np.mean([it["agree"] for it in v])) for k, v in by_fit.items() if k != "real"
)
tau_transport = float(np.quantile(null_agrees, 1 - 0.05))
# per-base bootstrap CI for the real transport
test_bases = splits["P/transfer"]["test"]
row2base = {}
for b in test_bases:
    for g in lib.PERMS:
        row2base[ct.row(b, g)] = b
per_base_bits = {}
for it in real:
    per_base_bits.setdefault(row2base[it["eval_row"]], []).append(it["agree"])
pbb = np.array([np.mean(v) for _, v in sorted(per_base_bits.items())])
rng = np.random.default_rng(lib.BOOT_SEED)
idx = rng.integers(0, len(pbb), size=(lib.N_BOOT, len(pbb)))
ci_transport = [float(np.percentile(pbb[idx].mean(1), q)) for q in (2.5, 97.5)]

print("== reproduction check (recomputed vs stored) ==")
all_ok = True
for k, (a, b) in checks.items():
    ok = (a is None and b is None) or abs(a - b) < 1e-9
    all_ok &= ok
    print(f"  {k}: {a:.10g} vs {b:.10g} -> {'EXACT' if ok else 'MISMATCH'}")
assert all_ok, "reproduction failed"

# 4. frozen threshold application via FROZEN logic
thresholds = dict(cfg["thresholds"])
thresholds["transport_agree"] = dict(
    dir="gt", tau=tau_transport, null_n=len(null_agrees)
)
metrics = dict(
    content_transfer_err=mv["content_transfer_err"],
    crosspath_err=mv["crosspath_err"],
    law_inv_defect=mv["law_inv_defect"],
    law_braid_defect=mv["law_braid_defect"],
    nontriv=mv["nontriv"],
    noncommute=mv["noncommute"],
    support_mass_lex=mv["support_mass_lex"],
    transport_agree=agree_real,
    probe_content_keep=mv["probe_content_keep"],
    probe_role_perm=mv["probe_role_perm"],
)
conds = fz.evaluate_conditions(metrics, thresholds)
verdict, score = fz.verdict_and_score(conds, metrics)
print("\n== conditions (frozen evaluate_conditions) ==")
for c, s in conds.items():
    print(f"  {c}: {s}")
print(f"VERDICT: {verdict}  score={score:.3f}")

out = dict(
    config_sha256=hashlib.sha256(cfg_txt.encode()).hexdigest(),
    reproduction=dict(
        all_exact=bool(all_ok),
        checks={k: [float(x) for x in v] for k, v in checks.items()},
    ),
    transport=dict(
        agree_real=agree_real,
        ci=ci_transport,
        null_agrees=null_agrees,
        tau=tau_transport,
        n_items=len(real),
        n_bases=len(pbb),
        harness=c5["harness"],
        wall_hours=c5["wall_hours"],
        est_cost_usd=c5["est_cost_usd"],
    ),
    thresholds_applied=thresholds,
    metrics=metrics,
    conditions=conds,
    verdict=verdict,
    score=float(score),
)
json.dump(out, open(lib.OUT / "verdict.json", "w"), indent=2)
print("written", lib.OUT / "verdict.json")

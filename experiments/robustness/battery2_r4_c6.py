"""Phase 8C supplementary robustness — R4-c6: reproduce the stored 0.298
content-probe accuracy at the verdict layer (L61) from stored bytes.

Two-part reproduction:
  (a) quote the stored values (committed_config.json
      .probes.availability_by_layer.61.content_acc and test_metrics.json
      .verdict_layer_metrics.probe_content_acc) with paths;
  (b) deterministic refit: the content probe per the exact stored recipe
      (phase8c_test_eval.py lines 124-141, content part: 12-way mark probe,
      lib.fit_probe, trained on ALL 150 P/fit CAL bases at layer 61,
      availability measured on the SECOND HALF of the P/transfer CAL bases)
      from the checksummed cached activations. lib.fit_probe is
      deterministic (fixed init, fixed iteration count, no rng), so the
      refit must reproduce the stored accuracy exactly.

CPU only. Writes results/phase8c/robustness/r4_c6_reproduction.json.
"""

import hashlib
import json
import time

import numpy as np

import activation_discriminator as lib
from activation_discriminator import PERMS

t0 = time.time()

# ---- config hash gate (same assertion path as phase8c_test_eval.py) ----
cfg_txt = (lib.OUT / "committed_config.json").read_text()
h = hashlib.sha256(cfg_txt.encode()).hexdigest()
h_rec = (lib.OUT / "committed_config.sha256").read_text().strip()
assert h == h_rec, "committed config hash mismatch"
cfg = json.loads(cfg_txt)
LAYER = cfg["layer"]
assert LAYER == 61

# ---- (a) stored values, with paths ----
stored_cfg_acc = cfg["probes"]["availability_by_layer"]["61"]["content_acc"]
tm = json.load(open(lib.OUT / "test_metrics.json"))
stored_tm_acc = tm["verdict_layer_metrics"]["probe_content_acc"]
stored_role_acc = tm["verdict_layer_metrics"]["probe_role_acc"]

# ---- (b) deterministic refit from cached activations ----
cf = lib.Cell("P", "fit")
ct = lib.Cell("P", "transfer")
splits = json.load(open(lib.OUT / "splits.json"))["splits"]
cal_pf = splits["P/fit"]["cal"]
cal_pt = splits["P/transfer"]["cal"]

tr_rows = np.array([cf.row(b, g) for b in cal_pf for g in PERMS])
pc = lib.fit_probe(cf.states(tr_rows, LAYER), lib.content_labels(cf, tr_rows), 12)
half = len(cal_pt) // 2
role_av = cal_pt[half:]
av_rows = np.array([ct.row(b, g) for b in role_av for g in PERMS])
Xav = ct.states(av_rows, LAYER)
hits = lib.probe_pred(pc, Xav) == lib.content_labels(ct, av_rows)
refit_acc = float(np.mean(hits))
n_correct, n_total = int(hits.sum()), int(len(hits))

out = dict(
    what=(
        "R4-c6: reproduce stored content-probe accuracy at verdict layer "
        "61 from stored bytes (deterministic refit per "
        "phase8c_test_eval.py lines 124-141, content part)"
    ),
    config_sha256=h,
    layer=LAYER,
    stored=dict(
        committed_config_content_acc=stored_cfg_acc,
        committed_config_path=(
            "results/verdict/discriminator/committed_config.json "
            ".probes.availability_by_layer.61.content_acc"
        ),
        test_metrics_content_acc=stored_tm_acc,
        test_metrics_path=(
            "results/verdict/discriminator/test_metrics.json "
            ".verdict_layer_metrics.probe_content_acc"
        ),
        test_metrics_role_acc=stored_role_acc,
        availability_rule=cfg["probes"]["availability_rule"],
    ),
    refit=dict(
        content_acc=refit_acc,
        n_correct=n_correct,
        n_total=n_total,
        train="P/fit CAL bases (150) x 6 perms = 900 episodes, layer 61",
        eval=(
            "second half of P/transfer CAL bases (75) x 6 perms = 450 "
            "episodes, layer 61"
        ),
        probe="lib.fit_probe (frozen _fit_probe verbatim; deterministic)",
        acts_files=dict(
            disc_P_fit="results/verdict/answer_position/acts/disc_P_fit.npy",
            disc_P_transfer="results/verdict/answer_position/acts/disc_P_transfer.npy",
            checksums="results/verdict/answer_position/acts/checksums.json",
        ),
    ),
    match=dict(
        refit_eq_committed=bool(refit_acc == stored_cfg_acc),
        refit_eq_test_metrics=bool(refit_acc == stored_tm_acc),
        abs_diff_committed=abs(refit_acc - stored_cfg_acc),
        abs_diff_test_metrics=abs(refit_acc - stored_tm_acc),
    ),
    wall_s=round(time.time() - t0, 1),
)
p = lib.OUT / "robustness" / "r4_c6_reproduction.json"
json.dump(out, open(p, "w"), indent=2)
print(f"config verified sha256={h[:16]}...")
print(f"stored committed_config content_acc = {stored_cfg_acc!r}")
print(f"stored test_metrics  content_acc = {stored_tm_acc!r}")
print(f"refit content_acc = {refit_acc!r}  ({n_correct}/{n_total})")
print(f"refit == committed_config: {refit_acc == stored_cfg_acc}")
print(f"refit == test_metrics:     {refit_acc == stored_tm_acc}")
print(f"wrote {p} ({time.time()-t0:.0f}s)")

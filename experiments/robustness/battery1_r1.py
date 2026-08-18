"""ROBUSTNESS PASS R1 — Condition-4 exact accounting + registered-variant
recompute, executed against the GENUINE Phase 8C artifacts.

WHAT RAN in 8C (code of record):
  - statistic: phase8c_lib.support_mass_lexical (lines 280-308) — fraction of
    (R - I) row-space mass inside the EPISODE-LOCAL difference subspace
    span{u_i - u_1} of the episode's k=3 name vectors, mass averaged over the
    first 48 eval episodes and the two generators.
  - u vectors: phase8c_test_eval.py lines 40-41:
        nv = np.load(lib.OUT / "name_vectors.npz", allow_pickle=True)
        name_vec = {n: v for n, v in zip(nv["names"], nv["readout"])}
    i.e. READOUT rows (lm_head.weight) of the name tokens, NOT embed_tokens.
  - direction: lt (committed_config.json thresholds.support_mass_lex.dir);
  - null/threshold: shuffled(10/layer, seed 2000*rep+layer) + identity(1/layer)
    cal-split fits evaluated on P/transfer CAL, pooled over the frozen
    12-layer set (132 values), LENIENT_QUANTILE flip -> tau = 0.95-quantile
    (phase8c_freeze_config.py lines 42-69).

REGISTERED definition: "nontriviality with the
corrected difference-subspace support test, direction lt"; frozen Stage-2
comment (src/shared/discriminator.py lines 33-38): "u_i are the episode's
name EMBEDDINGS". The run used the readout (unembedding) rows. Structure and
direction match the registered definition; the "which embedding" reading is
ambiguous (embed_tokens vs lm_head). This script therefore:

  (a) RE-DERIVES the stored tau from the stored null values
      (results/phase8c/null_values_per_layer.json) — byte-level check;
  (b) recomputes the READOUT-span variant (real: test-split fits evaluated on
      P/transfer TEST; nulls: cal-split fits evaluated on P/transfer CAL,
      frozen seed rule) and requires EXACT match to the stored per-layer
      values — validates the machinery;
  (c) runs the EMBED-TOKENS-span variant ONCE on the same deterministic fits
      at the frozen 12-layer set, with matched nulls and the identical frozen
      threshold rule — the registered-variant recompute authorised by the
      resolution instruction. No other variant is computed.

Fidelity note: null fits reuse the X-derived pieces (M, MX, lam_eff) of the
real fit at the same layer/split — mathematically identical to refitting
because DualRidge's M = (XX^T + lam_eff I)^-1 and lam_eff depend on X only;
Y enters the metrics linearly through apply_* (phase8c_lib.py lines 157-176).

Outputs (written incrementally, per layer, as computed):
  results/phase8c/robustness/r1_c4_accounting.json
  results/phase8c/robustness/r1_registered_c4.json
"""

import copy
import json
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
import activation_discriminator as lib  # noqa: E402
from activation_discriminator import GENERATORS  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[2]
ROB = ROOT / "results/robustness/battery"
ROB.mkdir(parents=True, exist_ok=True)

FPR = 0.05
N_SHUF = 10

t0 = time.time()

cfg = json.load(open(ROOT / "results/verdict/discriminator/committed_config.json"))
tm = json.load(open(ROOT / "results/verdict/discriminator/test_metrics.json"))
nulls_stored = json.load(open(ROOT / "results/verdict/discriminator/null_values_per_layer.json"))
splits = json.load(open(ROOT / "results/verdict/discriminator/splits.json"))["splits"]
LAM = cfg["lambda"]
LAYERS = cfg["reporting_layer_set"]
L_STAR = cfg["layer"]

# ---------------------------------------------------------------- part (a)
# accounting: stored C4 values + re-derived tau from stored nulls
stored_null_sml = []
for layer in LAYERS:
    ln = nulls_stored[str(layer)]
    stored_null_sml += [m["support_mass_lex"] for m in ln["shuffled"]]
    stored_null_sml += [m["support_mass_lex"] for m in ln["identity"]]
q = 1.0 - FPR  # lt + LENIENT_QUANTILE flip (phase8c_freeze_config.py 64-66)
tau_rederived = float(np.quantile(stored_null_sml, q))
tau_stored = cfg["thresholds"]["support_mass_lex"]["tau"]

acct = dict(
    what_ran=dict(
        statistic="phase8c_lib.support_mass_lexical (lines 280-308): "
        "||(R-I)q_b||_F^2 / ||R-I||_F^2, q_b = orthonormal basis of "
        "episode-local span{u_i - u_1}, mean over first 48 eval "
        "episodes and both generators",
        u_vectors="READOUT rows (lm_head.weight) of the episode's 3 name "
        "tokens — phase8c_test_eval.py lines 40-41 load "
        "name_vectors.npz key 'readout'",
        direction="lt (committed_config.json .thresholds.support_mass_lex.dir)",
        null="shuffled x10 (seed 2000*rep+layer) + identity, cal-split fits "
        "evaluated on P/transfer CAL, pooled over 12 frozen layers "
        "(132 values), LENIENT_QUANTILE -> tau = 0.95 quantile "
        "(phase8c_freeze_config.py lines 42-69, 198-208)",
    ),
    registered_definition="'nontriviality with "
    "the corrected difference-subspace support test, direction lt'; "
    "src/shared/discriminator.py lines 33-38: u_i = the episode's "
    "name embeddings, lt, LENIENT",
    match_statement="structure (episode-local name-difference span), "
    "direction (lt), and null/threshold rule (LENIENT 1-FPR quantile) "
    "all match the registered definition; the run instantiated 'name "
    "embedding' as the lm_head READOUT row (the vector a lexical-swap "
    "readout artifact acts on). The embed_tokens reading is recomputed "
    "in r1_registered_c4.json to close the ambiguity.",
    stored_values=dict(
        source="results/verdict/discriminator/test_metrics.json",
        verdict_layer=dict(
            layer=L_STAR,
            support_mass_lex=tm["verdict_layer_metrics"]["support_mass_lex"],
            support_mass_ci=tm["verdict_layer_metrics"]["support_mass_ci"],
            nontriv=tm["verdict_layer_metrics"]["nontriv"],
            nontriv_ci=tm["verdict_layer_metrics"]["nontriv_ci"],
            noncommute=tm["verdict_layer_metrics"]["noncommute"],
            noncommute_ci=tm["verdict_layer_metrics"]["noncommute_ci"],
        ),
        per_layer_support_mass_lex={
            l: tm["per_layer"][l]["support_mass_lex"] for l in tm["per_layer"]
        },
        thresholds=dict(
            support_mass_lex=cfg["thresholds"]["support_mass_lex"],
            nontriv=cfg["thresholds"]["nontriv"],
            noncommute=cfg["thresholds"]["noncommute"],
        ),
        thresholds_bonferroni=dict(
            support_mass_lex=cfg["thresholds_bonferroni"]["support_mass_lex"],
            nontriv=cfg["thresholds_bonferroni"]["nontriv"],
            noncommute=cfg["thresholds_bonferroni"]["noncommute"],
        ),
    ),
    tau_rederivation=dict(
        n_stored_null_values=len(stored_null_sml),
        quantile=q,
        tau_rederived=tau_rederived,
        tau_stored=tau_stored,
        exact_match=bool(abs(tau_rederived - tau_stored) < 1e-15),
        source="results/verdict/discriminator/null_values_per_layer.json",
    ),
    noncommute_tau_note=dict(
        stored_tau=cfg["thresholds"]["noncommute"]["tau"],
        report_text_value="~4.7e-9 (phase8c_discriminator_report.md table)",
        note="report table quotes ~4.7e-9; committed_config.json and "
        "verdict.json both store tau=0.0 (identity-null noncommute is "
        "exactly 0). Flagged as a report-prose discrepancy; the stored "
        "config governs. gt-0 comparison unaffected.",
    ),
)
json.dump(acct, open(ROB / "r1_c4_accounting.json", "w"), indent=2)
print(
    "wrote r1_c4_accounting.json; tau rederived",
    tau_rederived,
    "stored",
    tau_stored,
    "match",
    acct["tau_rederivation"]["exact_match"],
)

# ---------------------------------------------------------------- part (b,c)
cells = lib.load_cells()
cf, ct = cells[("P", "fit")], cells[("P", "transfer")]
cal_Pfit, cal_Ptr = splits["P/fit"]["cal"], splits["P/transfer"]["cal"]
test_Pfit, test_Ptr = splits["P/fit"]["test"], splits["P/transfer"]["test"]

nv = np.load(ROOT / "results/verdict/discriminator/name_vectors.npz", allow_pickle=True)
name_vec_readout = {n: v for n, v in zip(nv["names"], nv["readout"])}
name_vec_embed = {n: v for n, v in zip(nv["names"], nv["embed"])}


def dual_with_Y(base, Y):
    """Identical math to DualRidge(X, Y, lam): M, MX, lam_eff depend on X
    only (phase8c_lib.py lines 157-167); Y enters apply_* linearly."""
    r = copy.copy(base)
    r.Y = Y
    return r


out = dict(
    definition="registered C4 support test on the stored deterministic fits; "
    "variant u = embed_tokens rows (name-embedding reading); "
    "readout variant recomputed alongside as exact-match check",
    lam=LAM,
    layers=list(LAYERS),
    fpr=FPR,
    threshold_rule="pooled nulls over 12 frozen layers, LENIENT 0.95 "
    "quantile (frozen rule, phase8c_freeze_config.py)",
    real_fit="P/fit TEST bases, eval P/transfer TEST (as the stored run)",
    null_fit="P/fit CAL bases, eval P/transfer CAL, shuffled seeds "
    "2000*rep+layer x10 + identity (as the stored run)",
    per_layer={},
    fidelity={},
)

for layer in LAYERS:
    tl = time.time()
    # ---- real fits on TEST split (deterministic reconstruction)
    Rs_real = {}
    for a in GENERATORS:
        rx, ry = lib.pair_rows(cf, test_Pfit, a)
        Rs_real[a] = lib.DualRidge(cf.states(rx, layer), cf.states(ry, layer), LAM)
    real_ro = lib.support_mass_lexical(Rs_real, ct, test_Ptr, name_vec_readout)
    real_em = lib.support_mass_lexical(Rs_real, ct, test_Ptr, name_vec_embed)
    stored_ro = tm["per_layer"][str(layer)]["support_mass_lex"]

    # ---- null fits on CAL split (frozen seed rule), Y-swap on shared X
    base_fits, Y_real, X_cal = {}, {}, {}
    for a in GENERATORS:
        rx, ry = lib.pair_rows(cf, cal_Pfit, a)
        X = cf.states(rx, layer)
        Y = cf.states(ry, layer)
        base_fits[a] = lib.DualRidge(X, Y, LAM)
        Y_real[a], X_cal[a] = Y, X
    null_ro, null_em = [], []
    for rep in range(N_SHUF):
        rng = np.random.default_rng(2000 * rep + layer)
        Rs_n = {}
        for a in GENERATORS:  # rng draws sequential across generators (frozen)
            Rs_n[a] = dual_with_Y(
                base_fits[a], Y_real[a][rng.permutation(len(Y_real[a]))]
            )
        null_ro.append(lib.support_mass_lexical(Rs_n, ct, cal_Ptr, name_vec_readout))
        null_em.append(lib.support_mass_lexical(Rs_n, ct, cal_Ptr, name_vec_embed))
    Rs_id = {a: dual_with_Y(base_fits[a], X_cal[a]) for a in GENERATORS}
    null_ro.append(lib.support_mass_lexical(Rs_id, ct, cal_Ptr, name_vec_readout))
    null_em.append(lib.support_mass_lexical(Rs_id, ct, cal_Ptr, name_vec_embed))

    stored_layer_nulls = [
        m["support_mass_lex"] for m in nulls_stored[str(layer)]["shuffled"]
    ] + [m["support_mass_lex"] for m in nulls_stored[str(layer)]["identity"]]
    fid = dict(
        real_readout_recomputed=real_ro,
        real_readout_stored=stored_ro,
        real_exact=bool(abs(real_ro - stored_ro) < 1e-12),
        null_readout_max_abs_dev=float(
            np.max(np.abs(np.array(null_ro) - np.array(stored_layer_nulls)))
        ),
    )
    out["per_layer"][str(layer)] = dict(
        embed_real=real_em,
        embed_nulls=null_em,
        readout_real=real_ro,
        readout_nulls=null_ro,
    )
    out["fidelity"][str(layer)] = fid
    json.dump(out, open(ROB / "r1_registered_c4.json", "w"), indent=2)
    print(
        f"layer {layer}: embed real={real_em:.8e} "
        f"nulls[{min(null_em):.3e},{max(null_em):.3e}] | readout "
        f"real={real_ro:.8e} (stored {stored_ro:.8e}, "
        f"exact={fid['real_exact']}, null dev {fid['null_readout_max_abs_dev']:.1e}) "
        f"({time.time()-tl:.0f}s)",
        flush=True,
    )

# ---- pooled thresholds, frozen rule
pool_em = [v for l in LAYERS for v in out["per_layer"][str(l)]["embed_nulls"]]
pool_ro = [v for l in LAYERS for v in out["per_layer"][str(l)]["readout_nulls"]]
tau_em = float(np.quantile(pool_em, 1 - FPR))
tau_em_bonf = float(np.quantile(pool_em, 1 - FPR / len(LAYERS)))
tau_ro_check = float(np.quantile(pool_ro, 1 - FPR))
out["thresholds"] = dict(
    embed=dict(dir="lt", tau=tau_em, tau_bonferroni=tau_em_bonf, null_n=len(pool_em)),
    readout_recheck=dict(
        tau=tau_ro_check,
        stored_tau=tau_stored,
        exact=bool(abs(tau_ro_check - tau_stored) < 1e-12),
    ),
)
out["embed_per_layer_lt"] = {
    str(l): dict(
        real=out["per_layer"][str(l)]["embed_real"],
        lt_tau=bool(out["per_layer"][str(l)]["embed_real"] < tau_em),
        lt_tau_bonf=bool(out["per_layer"][str(l)]["embed_real"] < tau_em_bonf),
    )
    for l in LAYERS
}
json.dump(out, open(ROB / "r1_registered_c4.json", "w"), indent=2)
print(
    "embed tau:",
    tau_em,
    "bonf:",
    tau_em_bonf,
    "| readout tau recheck:",
    tau_ro_check,
    "stored:",
    tau_stored,
)
print(f"wrote r1_registered_c4.json ({time.time()-t0:.0f}s total)")

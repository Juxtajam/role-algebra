"""Phase 8C — Steps 4-6 (CPU part): SINGLE EVALUATION on the untouched test
split with the frozen committed config. Conditions 1, 2, 3, 4, 6 + the 8B
spectrum diagnostic + conditioning report + per-layer descriptive table
(Bonferroni thresholds, stated in advance). Condition 5 is produced by the
pre-committed GPU procedure (phase8c_cond5_*.py); this script writes the
patch set inputs for it and stores every metric WITH per-base decompositions
and stored predictions so the report can reproduce all numbers.

Verifies the committed config hash before touching any test base.
"""

import hashlib
import json
import time

import numpy as np

import activation_discriminator as lib
from activation_discriminator import GENERATORS, GEN_NAMES, PERMS

t0 = time.time()

# ---------------- frozen config (hash-verified) ----------------
cfg_txt = (lib.OUT / "committed_config.json").read_text()
h = hashlib.sha256(cfg_txt.encode()).hexdigest()
h_rec = (lib.OUT / "committed_config.sha256").read_text().strip()
assert h == h_rec, "committed config hash mismatch"
cfg = json.loads(cfg_txt)
LAYER = cfg["layer"]
LAM = cfg["lambda"]
LAYER_SET = cfg["reporting_layer_set"]
print(f"config verified sha256={h[:16]}...; layer={LAYER} lambda={LAM:g}")

cells = lib.load_cells()
splits = json.load(open(lib.OUT / "splits.json"))["splits"]
cal = {k: v["cal"] for k, v in splits.items()}
test = {k: v["test"] for k, v in splits.items()}
cf, ct = cells[("P", "fit")], cells[("P", "transfer")]
cg, cgt = cells[("G", "fit")], cells[("G", "transfer")]

nv = np.load(lib.OUT / "name_vectors.npz", allow_pickle=True)
name_vec = {n: v for n, v in zip(nv["names"], nv["readout"])}

results = dict(config_sha256=h, layer=LAYER, **{"lambda": LAM})


def fit_real(layer, bases):
    Rs = {}
    conditioning = {}
    for a in GENERATORS:
        rx, ry = lib.pair_rows(cf, bases, a)
        X, Y = cf.states(rx, layer), cf.states(ry, layer)
        R = lib.DualRidge(X, Y, LAM)
        Rs[a] = R
        s = R.eig_s
        conditioning[GEN_NAMES[GENERATORS.index(a)]] = dict(
            n_pairs=len(rx),
            d=lib.D_MODEL,
            lam=LAM,
            lam_eff=R.lam_eff,
            gram_eig_max=float(s[-1]),
            gram_eig_min=float(s[0]),
            cond_regularised=R.cond_eff,
            cond_unregularised_gram=float(s[-1] / max(s[0], 1e-300)),
        )
    return Rs, conditioning


def boot_ratio(nums, dens, seed=lib.BOOT_SEED, n_boot=lib.N_BOOT):
    """Bootstrap CI over base problems for sum(num)/sum(den) statistics.
    nums/dens: (n_gen, n_bases). Statistic = mean over generators."""
    nums, dens = np.asarray(nums), np.asarray(dens)
    nb = nums.shape[1]
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, nb, size=(n_boot, nb))
    stats = np.mean(
        nums[:, idx].sum(axis=2) / np.maximum(dens[:, idx].sum(axis=2), 1e-12), axis=0
    )
    return [float(np.percentile(stats, q)) for q in (2.5, 97.5)]


def boot_mean(bits_per_base, seed=lib.BOOT_SEED, n_boot=lib.N_BOOT):
    arr = np.asarray(bits_per_base, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(arr), size=(n_boot, len(arr)))
    stats = arr[idx].mean(axis=1)
    return [float(np.percentile(stats, q)) for q in (2.5, 97.5)]


def metrics_at_layer(layer, store_preds=False):
    """Full battery at one layer, TEST split, per-base decompositions."""
    Rs, conditioning = fit_real(layer, test["P/fit"])
    m = dict(conditioning=conditioning)
    pb_all = {}

    # ---- C1: content transfer on the disjoint vocabulary (P/transfer test)
    err, pb = lib.pair_error(Rs, ct, test["P/transfer"], layer, per_base=True)
    m["content_transfer_err"] = err
    m["content_transfer_ci"] = boot_ratio(
        [p["num"] for p in pb], [p["den"] for p in pb]
    )
    pb_all["content_transfer"] = pb

    # ---- C2: cross-path, fit-on-P / eval-on-G (G/fit test; G/transfer desc)
    err, pb = lib.pair_error(Rs, cg, test["G/fit"], layer, per_base=True)
    m["crosspath_err"] = err
    m["crosspath_ci"] = boot_ratio([p["num"] for p in pb], [p["den"] for p in pb])
    pb_all["crosspath"] = pb
    err2, pb2 = lib.pair_error(Rs, cgt, test["G/transfer"], layer, per_base=True)
    m["crosspath_transfer_err_desc"] = err2
    m["crosspath_transfer_ci_desc"] = boot_ratio(
        [p["num"] for p in pb2], [p["den"] for p in pb2]
    )

    # ---- C3 + C4 numerators: activation-weighted laws on P/transfer test
    laws, lpb = lib.group_law_metrics(Rs, ct, test["P/transfer"], layer, per_base=True)
    m.update(laws)
    m["law_inv_ci"] = boot_ratio(lpb["inv"], [lpb["hn"]] * 2)
    m["law_braid_ci"] = boot_ratio([lpb["braid_num"]], [lpb["braid_den"]])
    m["nontriv_ci"] = boot_ratio(lpb["nontriv"], [lpb["hn"]] * 2)
    m["noncommute_ci"] = boot_ratio([lpb["noncomm"]], [lpb["hn"]])
    pb_all["laws"] = lpb

    # ---- C4 support test (corrected difference-subspace, direction lt)
    mass, per = lib.support_mass_lexical(
        Rs, ct, test["P/transfer"], name_vec, per_ep=True
    )
    m["support_mass_lex"] = mass
    m["support_mass_ci"] = boot_mean(np.mean(per, axis=0))
    pb_all["support_mass_per_ep"] = per

    # ---- C6: probes per frozen config (trained on CAL data only)
    tr_names = lib.vocab_names(ct)
    tr_rows_pf = np.array([cf.row(b, g) for b in cal["P/fit"] for g in PERMS])
    pc = lib.fit_probe(
        cf.states(tr_rows_pf, layer), lib.content_labels(cf, tr_rows_pf), 12
    )
    half = len(cal["P/transfer"]) // 2
    role_tr = cal["P/transfer"][:half]
    role_av = cal["P/transfer"][half:]
    rp_rows = np.array([ct.row(b, g) for b in role_tr for g in PERMS])
    pr = lib.fit_probe(
        ct.states(rp_rows, layer), lib.name_labels(ct, rp_rows, tr_names), 12
    )
    av_rows = np.array([ct.row(b, g) for b in role_av for g in PERMS])
    Xav = ct.states(av_rows, layer)
    cand_av = lib.candidate_matrix(ct, av_rows, tr_names)
    slot_av = np.array(
        [list(ct.recs[i]["base"]["names"]).index(ct.recs[i]["answer"]) for i in av_rows]
    )
    c_acc = float(np.mean(lib.probe_pred(pc, Xav) == lib.content_labels(ct, av_rows)))
    r_acc = float(np.mean(lib.episode_local_pred(pr, Xav, cand_av) == slot_av))
    available = c_acc >= 0.9 and r_acc >= 0.9
    m["probe_content_acc"] = c_acc
    m["probe_role_acc"] = r_acc
    m["probes_available"] = bool(available)
    ev_rows = np.array([ct.row(b, g) for b in test["P/transfer"] for g in PERMS])
    Xev = ct.states(ev_rows, layer)
    cand_ev = lib.candidate_matrix(ct, ev_rows, tr_names)
    base_c = lib.probe_pred(pc, Xev)
    base_r = lib.episode_local_pred(pr, Xev, cand_ev)
    keeps, perms_ = [], []
    for a in GENERATORS:
        XevR = Rs[a].apply_RT(Xev)
        keeps.append((lib.probe_pred(pc, XevR) == base_c).astype(int))
        perms_.append(
            (lib.episode_local_pred(pr, XevR, cand_ev) == np.array(a)[base_r]).astype(
                int
            )
        )
    if available:
        m["probe_content_keep"] = float(np.mean(keeps))
        m["probe_role_perm"] = float(np.mean(perms_))
        nb = len(test["P/transfer"])
        m["probe_content_keep_ci"] = boot_mean(
            np.mean(keeps, axis=0).reshape(nb, 6).mean(1)
        )
        m["probe_role_perm_ci"] = boot_mean(
            np.mean(perms_, axis=0).reshape(nb, 6).mean(1)
        )
    else:
        m["probe_content_keep"] = None
        m["probe_role_perm"] = None
    pb_all["probe_keep_bits"] = [k.tolist() for k in keeps]
    pb_all["probe_perm_bits"] = [p_.tolist() for p_ in perms_]

    # ---- diagnostics
    m["eff_rank_R_minus_I_diag"] = float(
        np.mean([lib.eff_rank_R_minus_I(Rs[a]) for a in GENERATORS])
    )
    return m, Rs, pb_all


# ================= verdict layer (single evaluation) =================
print(f"== TEST evaluation at frozen layer {LAYER} ==")
m_verdict, Rs_verdict, pb_verdict = metrics_at_layer(LAYER, store_preds=True)
for k in (
    "content_transfer_err",
    "crosspath_err",
    "law_inv_defect",
    "law_braid_defect",
    "nontriv",
    "noncommute",
    "support_mass_lex",
    "probe_content_acc",
    "probe_role_acc",
    "probe_content_keep",
    "probe_role_perm",
    "eff_rank_R_minus_I_diag",
):
    print(f"  {k} = {m_verdict[k]}")
results["verdict_layer_metrics"] = m_verdict

# ---- 8B spectrum diagnostic (conditional; u from CAL split, frozen rule)
spec_raw = lib.spectrum_diagnostic(
    Rs_verdict,
    cf,
    cal["P/fit"],
    LAYER,
    rank_cut=cfg["spectrum_diagnostic"]["rank_cut"],
    centered=False,
)
spec_cen = lib.spectrum_diagnostic(
    Rs_verdict,
    cf,
    cal["P/fit"],
    LAYER,
    rank_cut=cfg["spectrum_diagnostic"]["rank_cut"],
    centered=True,
)
results["spectrum_diagnostic"] = dict(raw=spec_raw, centered=spec_cen)
print(
    "spectrum raw: span",
    spec_raw["span_dim"],
    "min_sv",
    round(spec_raw["stacked_min_sv"], 4),
    "sv_ratios",
    [round(x, 4) for x in spec_raw["sv_ratios"]],
)
print(
    "spectrum centered: span",
    spec_cen["span_dim"],
    "min_sv",
    round(spec_cen["stacked_min_sv"], 4),
    "sv_ratios",
    [round(x, 4) for x in spec_cen["sv_ratios"]],
)

# tau_inv recalibration per frozen rule: stacked-min-sv over the shuffled
# null fits at the verdict layer (CAL split), 0.05 quantile
null_min_svs_raw, null_min_svs_cen = [], []
for rep in range(10):
    seed = 2000 * rep + LAYER
    Rs_null = lib.fit_generators(
        cf, cal["P/fit"], LAYER, LAM, null_mode="shuffled", null_seed=seed
    )
    sr = lib.spectrum_diagnostic(
        Rs_null,
        cf,
        cal["P/fit"],
        LAYER,
        rank_cut=cfg["spectrum_diagnostic"]["rank_cut"],
        centered=False,
    )
    sc = lib.spectrum_diagnostic(
        Rs_null,
        cf,
        cal["P/fit"],
        LAYER,
        rank_cut=cfg["spectrum_diagnostic"]["rank_cut"],
        centered=True,
    )
    null_min_svs_raw.append(sr["stacked_min_sv"])
    null_min_svs_cen.append(sc["stacked_min_sv"])
tau_inv_raw = float(np.quantile(null_min_svs_raw, 0.05))
tau_inv_cen = float(np.quantile(null_min_svs_cen, 0.05))
results["spectrum_diagnostic"]["tau_inv"] = dict(
    raw=tau_inv_raw,
    centered=tau_inv_cen,
    null_min_svs_raw=null_min_svs_raw,
    null_min_svs_cen=null_min_svs_cen,
    rule=cfg["spectrum_diagnostic"]["tau_inv_rule"],
)
print(
    f"tau_inv (5% of shuffled-null min-sv): raw={tau_inv_raw:.4f} "
    f"centered={tau_inv_cen:.4f}"
)

# ================= per-layer descriptive table =================
results["per_layer"] = {}
for layer in LAYER_SET:
    tl = time.time()
    if layer == LAYER:
        m = {k: v for k, v in m_verdict.items()}
    else:
        m, _, _ = metrics_at_layer(layer)
    results["per_layer"][str(layer)] = {
        k: m.get(k)
        for k in (
            "content_transfer_err",
            "content_transfer_ci",
            "crosspath_err",
            "crosspath_ci",
            "crosspath_transfer_err_desc",
            "law_inv_defect",
            "law_braid_defect",
            "nontriv",
            "noncommute",
            "support_mass_lex",
            "probe_content_acc",
            "probe_role_acc",
            "probes_available",
            "probe_content_keep",
            "probe_role_perm",
            "eff_rank_R_minus_I_diag",
            "conditioning",
        )
    }
    print(
        f"layer {layer}: C1={m['content_transfer_err']:.4f} "
        f"C2={m['crosspath_err']:.4f} inv={m['law_inv_defect']:.4f} "
        f"({time.time()-tl:.0f}s)"
    )

json.dump(results, open(lib.OUT / "test_metrics.json", "w"), indent=2)
json.dump(
    {k: v for k, v in pb_verdict.items()}, open(lib.OUT / "test_per_base.json", "w")
)
print(f"saved test_metrics.json + test_per_base.json ({time.time()-t0:.0f}s)")

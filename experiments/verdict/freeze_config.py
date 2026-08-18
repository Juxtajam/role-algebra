"""Phase 8C — Step 3 ( instruction A): threshold recalibration against
matched-regime nulls on the CALIBRATION split only, then FREEZE the
committed config (write + sha256 + record) BEFORE the test split is touched.

Reads calibration_curves.json (Step 2 outputs). Frozen selection rules
(src/shared/calibrate.py, applied verbatim):
  * lambda*  = grid value minimising min-over-layers cal transfer error
  * layer L* = min(argmin-over-layers cal transfer err at lambda*,
                   first_decodable(P/fit cal))          [select_layer]
  * thresholds = FPR-0.05 quantiles of pooled null values
    (_quantile_thresholds verbatim, incl. LENIENT_QUANTILE flips)

MATCHED REGIME (stated instruction A): every null fit uses the SAME
pair count as the real test fits (cal split and test split have identical
base counts by the split construction: 150 P-fit bases -> 900 pairs per
generator), the SAME frozen lambda*, and the SAME layer set (the frozen
reporting layer set) as the real test-phase fits. Null modes: shuffled
(10 reps, frozen seed rule 2000*rep+layer) + identity (deterministic here,
1 value per layer — declared). No S-retrieval organism exists for a
pretrained LM; per the Stage-2 precedent the null sources that exist
(shuffled/identity) are used and this is declared in the report.

transport_agree (C5) thresholds cannot be computed without the GPU; the
PROCEDURE is frozen here (null patch sets, seeds, quantile rule) and the
numeric tau is filled by the pre-committed procedure from cal-split null
transports in the single condition-5 session, before the real test
transport values are unblinded to the verdict logic.
"""

import hashlib
import json
import time

import numpy as np

import activation_discriminator as lib
from activation_discriminator import GENERATORS, PERMS

FPR = 0.05
NULL_REPS_SHUF = 10

# frozen quantile logic (calibrate._quantile_thresholds verbatim)
METRICS = {
    "content_transfer_err": ("lt", ("shuffled",)),
    "crosspath_err": ("lt", ("shuffled",)),
    "law_inv_defect": ("lt", ("shuffled",)),
    "law_braid_defect": ("lt", ("shuffled",)),
    "nontriv": ("gt", ("identity",)),
    "noncommute": ("gt", ("identity",)),
    "support_mass_lex": ("lt", ("shuffled", "identity")),
    "transport_agree": ("gt", ("shuffled", "identity")),  # GPU, procedure frozen here
    "probe_content_keep": ("gt", ("shuffled",)),
    "probe_role_perm": ("gt", ("shuffled",)),
}
LENIENT = {"law_braid_defect", "support_mass_lex"}


def quantile_thresholds(null_values, fpr=FPR, m_correction=1):
    out = {}
    eff_fpr = fpr / m_correction
    for metric, (direction, _) in METRICS.items():
        vals = [v for v in null_values.get(metric, []) if v is not None]
        if not vals:
            continue
        q = eff_fpr if direction == "lt" else 1.0 - eff_fpr
        if metric in LENIENT:
            q = 1.0 - q
        out[metric] = dict(
            dir=direction, tau=float(np.quantile(vals, q)), null_n=len(vals)
        )
    return out


def main():
    t0 = time.time()
    cells = lib.load_cells()
    splits = json.load(open(lib.OUT / "splits.json"))["splits"]
    cal = {k: v["cal"] for k, v in splits.items()}
    curves = json.load(open(lib.OUT / "calibration_curves.json"))

    # ---------- frozen selections from the cal curves ----------
    real_err = np.array(curves["real_transfer_err"])  # (80, 6)
    lam_scores = real_err.min(axis=0)  # min over layers
    li_star = int(np.argmin(lam_scores))
    lam = lib.LAM_GRID[li_star]
    first_dec = curves["decodability"]["P/fit"]["first_decodable"]
    argmin_layer = int(np.argmin(real_err[:, li_star]))
    L_star = min(argmin_layer, first_dec)
    # reporting layer set (frozen): strided + the committed layers
    layer_set = sorted(set(list(range(0, 80, 8)) + [79, L_star, first_dec]))
    m = len(layer_set)
    print(
        f"lambda*={lam:g} (min cal err per lam: "
        f"{[round(float(x),4) for x in lam_scores]})"
    )
    print(
        f"argmin layer={argmin_layer} first_decodable(P/fit)={first_dec} "
        f"-> L*={L_star}; reporting set ({m}): {layer_set}"
    )

    cf, ct, cg = cells[("P", "fit")], cells[("P", "transfer")], cells[("G", "fit")]
    bases_fit, bases_tr, bases_g = cal["P/fit"], cal["P/transfer"], cal["G/fit"]
    n_pairs = len(bases_fit) * 6

    # ---------- frozen probes (calibration split only) ----------
    # content probe: 12-way mark probe on P/fit cal (marks shared across
    # vocabularies -> transfers to P/transfer). Frozen convention.
    # role probe: episode-local 12-way name probe. DECLARED ADAPTATION: the
    # frozen Stage-1 role probe trained on P/fit transfers only because slot
    # labels were vocabulary-independent there; here the answer classes are
    # vocabulary tokens, so the role probe trains on the FIRST HALF of the
    # P/transfer CAL bases (calibration data only), availability is measured
    # on the second half, and test evaluation uses the untouched test bases.
    tr_names = lib.vocab_names(ct)
    half = len(bases_tr) // 2
    role_tr_bases, role_av_bases = bases_tr[:half], bases_tr[half:]
    probes = {}
    null_values = {}
    per_layer_null = {}

    for layer in layer_set:
        tl = time.time()
        # -- real activations at this layer
        pairs_fit = {a: lib.pair_rows(cf, bases_fit, a) for a in GENERATORS}
        Xg = {}
        for a in GENERATORS:
            rx, ry = pairs_fit[a]
            Xg[a] = (cf.states(rx, layer), cf.states(ry, layer))
        ev_rows_t = np.array([ct.row(b, g) for b in bases_tr for g in PERMS])
        Ht = ct.states(ev_rows_t, layer)
        pairs_tr = {a: lib.pair_rows(ct, bases_tr, a) for a in GENERATORS}
        TRX = {
            a: (ct.states(pairs_tr[a][0], layer), ct.states(pairs_tr[a][1], layer))
            for a in GENERATORS
        }
        pairs_g = {a: lib.pair_rows(cg, bases_g, a) for a in GENERATORS}
        GX = {
            a: (cg.states(pairs_g[a][0], layer), cg.states(pairs_g[a][1], layer))
            for a in GENERATORS
        }

        # -- probes at this layer (depend on real data only; cached)
        tr_rows_pf = np.array([cf.row(b, g) for b in bases_fit for g in PERMS])
        Xpf = cf.states(tr_rows_pf, layer)
        c_tr = lib.content_labels(cf, tr_rows_pf)
        pc = lib.fit_probe(Xpf, c_tr, 12)
        rp_rows = np.array([ct.row(b, g) for b in role_tr_bases for g in PERMS])
        Xrp = ct.states(rp_rows, layer)
        y_rp = lib.name_labels(ct, rp_rows, tr_names)
        pr = lib.fit_probe(Xrp, y_rp, 12)
        av_rows = np.array([ct.row(b, g) for b in role_av_bases for g in PERMS])
        Xav = ct.states(av_rows, layer)
        cand_av = lib.candidate_matrix(ct, av_rows, tr_names)
        slot_av = np.array(
            [
                list(ct.recs[i]["base"]["names"]).index(ct.recs[i]["answer"])
                for i in av_rows
            ]
        )
        c_av = lib.content_labels(ct, av_rows)
        c_acc = float(np.mean(lib.probe_pred(pc, Xav) == c_av))
        r_acc = float(np.mean(lib.episode_local_pred(pr, Xav, cand_av) == slot_av))
        probes[layer] = dict(
            content_acc=c_acc,
            role_acc=r_acc,
            available=bool(c_acc >= 0.9 and r_acc >= 0.9),
        )

        # name readout vectors for support_mass_lex
        nv = np.load(lib.OUT / "name_vectors.npz", allow_pickle=True)
        name_vec = {n: v for n, v in zip(nv["names"], nv["readout"])}

        def suite(null_mode, null_seed):
            """all_metrics equivalent on cal split with matched-regime fit."""
            rng = np.random.default_rng(null_seed)
            Rs = {}
            for a in GENERATORS:
                X, Y = Xg[a]
                if null_mode == "identity":
                    Yn = X
                else:
                    Yn = Y[rng.permutation(len(Y))] if null_mode == "shuffled" else Y
                Rs[a] = lib.DualRidge(X, Yn, lam)
            m_ = {}
            # C1 pair error on P/transfer cal
            errs = []
            for a in GENERATORS:
                Vx, Vy = TRX[a]
                P = Rs[a].apply_RT(Vx)
                errs.append(((P - Vy) ** 2).sum() / max(((Vy - Vx) ** 2).sum(), 1e-12))
            m_["content_transfer_err"] = float(np.mean(errs))
            # C2 crosspath on G/fit cal
            errs = []
            for a in GENERATORS:
                Gx, Gy = GX[a]
                P = Rs[a].apply_RT(Gx)
                errs.append(((P - Gy) ** 2).sum() / max(((Gy - Gx) ** 2).sum(), 1e-12))
            m_["crosspath_err"] = float(np.mean(errs))
            m_.update(lib.group_law_metrics(Rs, ct, bases_tr, layer))
            m_["support_mass_lex"] = lib.support_mass_lexical(
                Rs, ct, bases_tr, name_vec
            )
            if probes[layer]["available"]:
                XavR = {a: Rs[a].apply_RT(Xav) for a in GENERATORS}
                base_c = lib.probe_pred(pc, Xav)
                base_r = lib.episode_local_pred(pr, Xav, cand_av)
                keep = [
                    float(np.mean(lib.probe_pred(pc, XavR[a]) == base_c))
                    for a in GENERATORS
                ]
                perm = [
                    float(
                        np.mean(
                            lib.episode_local_pred(pr, XavR[a], cand_av)
                            == np.array(a)[base_r]
                        )
                    )
                    for a in GENERATORS
                ]
                m_["probe_content_keep"] = float(np.mean(keep))
                m_["probe_role_perm"] = float(np.mean(perm))
            return m_

        layer_nulls = {"shuffled": [], "identity": []}
        for rep in range(NULL_REPS_SHUF):
            m_ = suite("shuffled", 2000 * rep + layer)
            layer_nulls["shuffled"].append(m_)
        layer_nulls["identity"].append(suite("identity", 0))
        per_layer_null[str(layer)] = layer_nulls
        for mode, runs in layer_nulls.items():
            for m_ in runs:
                for metric, (_, sources) in METRICS.items():
                    if mode in sources and m_.get(metric) is not None:
                        null_values.setdefault(metric, []).append(m_[metric])
        print(
            f"layer {layer}: probes c={c_acc:.3f} r={r_acc:.3f} "
            f"avail={probes[layer]['available']} nulls done "
            f"({time.time()-tl:.0f}s)"
        )

    thresholds = quantile_thresholds(null_values)
    thresholds_bonf = quantile_thresholds(null_values, m_correction=m)
    for k, t in thresholds.items():
        print(f"threshold {k}: {t['dir']} {t['tau']:.4f} (n={t['null_n']})")

    cfg = dict(
        phase="8C",
        frozen_at=time.strftime("%Y-%m-%d %H:%M:%S"),
        content_hash="84f2e54d85d6e8aa4c1474b608bef5ab69babe54353ef0ef2702d9f6ed38baef",
        model="Qwen/Qwen2.5-72B-Instruct",
        revision="495f39366efef23836d0cfae4fbe635880d2be31",
        splits_file="splits.json",
        **{"lambda": lam},
        lam_grid=lib.LAM_GRID,
        lambda_rule="argmin over grid of min-over-80-layers cal transfer err (frozen calibrate_system rule)",
        layer=L_star,
        patch_layer=L_star,
        first_decodable_P_fit=first_dec,
        decodability_all_cells={
            k: v["first_decodable"] for k, v in curves["decodability"].items()
        },
        layer_rule="min(argmin cal transfer err at lambda*, first_decodable(P/fit cal)) — frozen select_layer",
        reporting_layer_set=layer_set,
        multiple_comparison_rule=(
            f"The verdict is a SINGLE evaluation at the frozen layer {L_star} "
            f"at FPR 0.05 (no cross-layer selection -> no correction needed "
            f"for the verdict). The descriptive per-layer reporting table "
            f"over the frozen {m}-layer set uses Bonferroni-corrected "
            f"thresholds at FPR 0.05/{m} (thresholds_bonferroni), stated in "
            f"advance."
        ),
        matched_regime=dict(
            n_fit_pairs_per_generator=n_pairs,
            n_fit_bases=len(bases_fit),
            d_model=lib.D_MODEL,
            statement=(
                "Null fits use identical sample count (900 pairs/generator "
                "= 150 P/fit bases x 6 perms; cal and test splits have "
                "identical base counts by construction), identical frozen "
                "lambda (grid identical to the real-fit grid), and the "
                "identical frozen layer set as the real test-phase fits. "
                "Thresholds at any other sample-to-dimension ratio are void "
                "and none are used."
            ),
            null_modes=dict(
                shuffled=dict(reps=NULL_REPS_SHUF, seed_rule="2000*rep+layer (frozen)"),
                identity=dict(reps=1, note="deterministic on cached activations"),
            ),
            no_retrieval_null="declared: no S-retrieval analog exists for a pretrained LM (Stage-2 precedent)",
        ),
        fpr_target=FPR,
        thresholds=thresholds,
        thresholds_bonferroni=thresholds_bonf,
        probes=dict(
            content="12-way mark probe, trained P/fit cal (all 150 bases), frozen _fit_probe",
            role=(
                "episode-local 12-way name probe, trained on first half of "
                "P/transfer cal bases, availability on second half; "
                "DECLARED adaptation (vocabulary-dependent answer classes)"
            ),
            availability_by_layer=probes,
            availability_rule="both probes >= 0.9 held-out at the verdict layer, else C6 unavailable",
        ),
        condition5=dict(
            procedure=(
                "patch-and-continue at the frozen patch layer: replace "
                "resid_post[patch_layer] at the answer position (last prompt "
                "token) of episode y=a.g's forward pass with R_a x_L(g), "
                "cast bf16; continue; greedy decode (8A rendering/decoding "
                "conventions verbatim); transport_agree = mean(pred_patch == "
                "pred_nat) with pred_nat = stored 8A-final natural preds "
                "(declared reuse; deterministic greedy, identical model)"
            ),
            real=dict(
                fit="P/fit TEST bases",
                eval="P/transfer TEST bases",
                perms_per_base=2,
                note="frozen all_perms[:2]",
            ),
            nulls=dict(
                fit="P/fit CAL bases",
                eval="P/transfer CAL bases",
                shuffled_seeds=[2000 * r + L_star for r in range(NULL_REPS_SHUF)],
                identity=True,
            ),
            threshold_rule=(
                "tau(transport_agree) = (1-0.05) quantile of the "
                "pooled cal-split null transports (frozen "
                "quantile logic, gt), computed by this "
                "pre-committed rule before real transports feed "
                "the verdict"
            ),
            gpu="A100-80GB:2 @ $5.00/h, single session, model loads once",
        ),
        spectrum_diagnostic=dict(
            role="conditional on H_role; never gates the verdict",
            u_vectors="slot-conditional activation means, P/fit CAL bases, verdict layer",
            rank_cut=0.10,
            rank_cut_basis="frozen 8B value, applied to CENTERED u spectrum (declared: raw spectrum is dominated by the grand activation mean in a pretrained LM)",
            tau_inv_rule="0.05 quantile of stacked-min-sv over the shuffled-null fits at the verdict layer (recalibrated per 8B section 2.5)",
            report="raw-span and centered-span inv-dims both reported; caveat: the u-sum direction conflates content mean with a trivial summand in a pretrained LM",
        ),
        verdict_space=dict(
            H_role="all conditions pass and C1/C2/C5 available (frozen verdict_and_score)",
            H_retrieval="any condition fails (in particular C1 or C2 failing cleanly)",
            inconclusive="mandatory condition unavailable (probe unavailability, conditioning failure) — blocking condition named",
        ),
        evaluation="single evaluation on the test split; no layer/lambda/threshold selection after test results are seen",
    )
    txt = json.dumps(cfg, indent=2, sort_keys=True)
    p = lib.OUT / "committed_config.json"
    p.write_text(txt)
    h = hashlib.sha256(txt.encode()).hexdigest()
    (lib.OUT / "committed_config.sha256").write_text(h + "\n")
    json.dump(
        per_layer_null, open(lib.OUT / "null_values_per_layer.json", "w"), indent=2
    )
    print(f"CONFIG FROZEN: {p}")
    print(f"CONFIG SHA256: {h}")
    print(f"total {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()

"""Track B discriminator driver — freeze / cache / calibrate / test / compare.

Implements discriminator_design.md (pre-registered 2026-08-09) at k=3, two
position classes, under deviations D18/D19.

Stages (run in order; each writes its artifacts before the next):
  freeze      write disc_committed_config.json + sha  (rules only, no data)
  cache SEED ARM   build eval sets, predictions, activations for one organism
  calibrate   lambda*, verdict layers, frozen thresholds from cal splits
              of the gated organisms (writes disc_thresholds.json)
  test        single evaluation per organism on test bases; verdicts
  compare     arm-level rates, Fisher exact, Mann-Whitney (exact, permutation)

Usage: .venv/bin/python experiments/phase10_trackB_discriminator.py <stage> [...]
"""

import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import trained_discriminator as dl  # noqa: E402
import trained_organism_data as od  # noqa: E402

sys.path.insert(0, str(ROOT / "experiments"))
from large_discriminator_paths import (  # noqa: E402
    OUT,
    CFG_PATH,
    CFG_SHA,
    sha256_file,
    verify_cfg,
    org_dir,
    load_set,
    cal_test_split,
    xy_fn,
    states_at,
    gated_organisms,
)

# Eval-set seeds: fresh family, disjoint from every training-time eval seed
# (9000/9500/9600/9700/9800/9850/9900 families used by finetune_modal.py).
SET_SEEDS = {
    "disc_P_fit": 7100,
    "disc_P_transfer": 7200,
    "disc_G_fit": 7300,
}
N_BASES = 600
# Amendment 1 (disc_committed_config_amendment1.json, declared before any
# fit): larger universes for the thin-filtered sets.
N_BASES_AMENDED = {"disc_P_transfer": 1800, "disc_G_fit": 6000}

CONFIG = dict(
    design="phase10/trackB/discriminator_design.md",
    deviations=["D18", "D19"],
    k=3,
    d_model=128,
    n_layers=8,
    arms=["R", "C"],
    position_classes=["answer", "entity_mention_carry"],
    organism_gate=(
        "FORMED_final per D18: pure Q_P held-out episode acc >= 0.5 "
        "at the final checkpoint; discriminator runs only on gated "
        "(seed, arm) pairs; stop rule 1.3(c): no discriminator at "
        "all unless some arm has >= 3 gated checkpoints"
    ),
    eval_sets={
        k: dict(
            seed=v,
            n_bases=N_BASES,
            qtok={"disc_P_fit": "Q_P", "disc_P_transfer": "Q_P", "disc_G_fit": "Q_G"}[
                k
            ],
            vocab={
                "disc_P_fit": "fit",
                "disc_P_transfer": "transfer",
                "disc_G_fit": "fit",
            }[k],
        )
        for k, v in SET_SEEDS.items()
    },
    joint_sets="same bases/seeds as disc_P_fit and disc_P_transfer, joint=True "
    "(C7 DESCRIPTIVE only per D19-iii)",
    split_rule="strict-orbit-correct filter per set, then even index -> cal, "
    "odd index -> test, on the filtered universe (design 5.2)",
    fit_split="generators fitted on disc_P_fit TEST bases (8C precedent); "
    "cal split used only for lambda/layer/thresholds/probe training",
    lambda_grid=dl.LAM_GRID,
    lambda_rule="per (arm, position class): argmin over grid of mean-over-"
    "gated-seeds cal transfer error at the best layer",
    layer_rule="per (arm, position class): min(argmin mean cal transfer err "
    "over layers, max first-decodable over gated seeds) — 8C rule",
    fpr=0.05,
    n_shuffled_nulls=10,
    null_seeds=list(range(10)),
    null_sources=dict(
        content_transfer_err="shuffled",
        crosspath_err="shuffled",
        law_inv_defect="shuffled",
        law_braid_defect="shuffled",
        nontriv="identity",
        noncommute="identity",
        support_mass_lex="shuffled",
        transport_agree="shuffled+identity 0.95q pre-committed",
        probe_content_keep="shuffled",
        probe_role_perm="shuffled",
    ),
    threshold_pooling="null values pooled across gated seeds per (arm, "
    "position, verdict layer); frozen quantile logic from "
    "src/shared/calibrate.py",
    verdict_logic="frozen src evaluate_conditions + verdict_and_score",
    combined_verdict="H_role per seed requires H_role at answer AND carry",
    c5_per_base=2,
    spectrum=dict(rank_cut=0.10, tau_inv_quantile=0.05),
    bootstrap=dict(n=10000, seed=20260807),
    arm_comparison=dict(
        primary="Fisher exact one-tailed on H_role_both " "rates (R > C)",
        secondary="Mann-Whitney U one-tailed "
        "on answer content_transfer_err (R < C), exact by "
        "permutation",
    ),
)


def freeze():
    assert not CFG_PATH.exists(), "config already frozen; refusing to overwrite"
    json.dump(CONFIG, open(CFG_PATH, "w"), indent=1, sort_keys=True)
    h = sha256_file(CFG_PATH)
    CFG_SHA.write_text(h + "  disc_committed_config.json\n")
    print(f"FROZEN disc_committed_config.json sha256={h}")


def cache(seed, arm):
    verify_cfg()
    ck = ROOT / f"results/trained_organisms/large/finetune/seed{seed}/{arm}/ckpt.pt"
    model = dl.load_model(seed, arm, ck)
    d = org_dir(seed, arm)
    manifest = dict(
        seed=seed, arm=arm, ckpt=str(ck), ckpt_sha256=sha256_file(ck), sets={}
    )
    for name, scfg in CONFIG["eval_sets"].items():
        qtok = getattr(od, scfg["qtok"])
        n_b = N_BASES_AMENDED.get(name, scfg["n_bases"])  # amendment 1
        ev = od.build_eval_orbits(
            arm, scfg["vocab"], n_b, seed=scfg["seed"] + seed, qtok=qtok
        )
        preds = dl.predictions(model, ev)
        ok = dl.strict_ok_bases(preds, ev)
        acts, carr = dl.capture_states(model, ev)
        np.savez_compressed(
            d / f"{name}.npz",
            acts=acts,
            carr=carr,
            preds=preds,
            strict_ok=ok,
            tokens=ev["tokens"],
            answers=ev["answers"],
            answer_pos=ev["answer_pos"],
            candidates=ev["candidates"],
            slots=ev["slots"],
            carry_pos=ev["carry_pos"],
        )
        manifest["sets"][name] = dict(
            n_bases=n_b,
            strict_ok=int(len(ok)),
            episode_acc=float((preds == ev["answers"]).mean()),
        )
        # joint variants for the two P sets (C7 descriptive)
        if name in ("disc_P_fit", "disc_P_transfer"):
            evj = od.build_eval_orbits(
                arm, scfg["vocab"], n_b, seed=scfg["seed"] + seed, qtok=qtok, joint=True
            )
            actsj, carrj = dl.capture_states(model, evj)
            predsj = dl.predictions(model, evj)
            np.savez_compressed(
                d / f"{name}_joint.npz",
                acts=actsj,
                carr=carrj,
                preds=predsj,
                answers=evj["answers"],
                candidates=evj["candidates"],
                carry_pos=evj["carry_pos"],
            )
            manifest["sets"][name + "_joint"] = dict(
                episode_acc=float((predsj == evj["answers"]).mean())
            )
    json.dump(manifest, open(d / "manifest.json", "w"), indent=1)
    print(
        f"cached seed{seed}/{arm}: "
        + " ".join(
            f"{k}:{v.get('strict_ok', '-')}/{v.get('n_bases', '-')}"
            for k, v in manifest["sets"].items()
        )
    )


def calibrate():
    verify_cfg()
    import os as _os

    exploratory = _os.environ.get("DISC_EXPLORATORY") == "1"  # D22
    gated = gated_organisms()
    print("gated organisms:", gated, "| EXPLORATORY" if exploratory else "")
    by_arm = {arm: [s for s, a in gated if a == arm] for arm in ("R", "C")}
    n_max = max(len(v) for v in by_arm.values())
    if not exploratory:
        assert n_max >= 3, (
            f"STOP rule 1.3(c): no arm has >=3 gated checkpoints "
            f"({ {k: len(v) for k, v in by_arm.items()} }) — "
            f"discriminator does not run"
        )

    out = dict(gated=[[s, a] for s, a in gated], exploratory=exploratory, cells={})
    for arm in ("R", "C"):
        for position in ("answer", "entity_mention_carry"):
            seeds = by_arm[arm]
            if not seeds:
                continue
            # ---- lambda + layer from pooled cal transfer curves -----------
            err_grid = np.zeros((len(dl.LAM_GRID), dl.N_LAYERS))
            fdecs = []
            for s in seeds:
                zf = load_set(s, arm, "disc_P_fit")
                zt = load_set(s, arm, "disc_P_transfer")
                zg_ = load_set(s, arm, "disc_G_fit")
                calf, _ = cal_test_split(zf["strict_ok"])
                calt, _ = cal_test_split(zt["strict_ok"])
                calg_, _ = cal_test_split(zg_["strict_ok"])
                # Amendment 2 calibration_fallback: transfer cal < 20 bases
                # -> lambda/layer curve uses the crosspath (G) cal error
                if len(calt) >= 20:
                    z_ev, cal_ev = zt, calt
                else:
                    z_ev, cal_ev = zg_, calg_
                for li in range(dl.N_LAYERS):
                    for gi, lam in enumerate(dl.LAM_GRID):
                        Rs = dl.fit_generators(xy_fn(zf, position, li), calf, lam)
                        err_grid[gi, li] += dl.pair_error_mat(
                            Rs, xy_fn(z_ev, position, li), cal_ev
                        )
                # first-decodable on answer states of P/fit cal (role slots)
                half = len(calf) // 2
                idx_tr = np.concatenate(
                    [[dl.ep_index(b, g) for g in dl.PERMS3] for b in calf[:half]]
                ).astype(int)
                idx_ev = np.concatenate(
                    [[dl.ep_index(b, g) for g in dl.PERMS3] for b in calf[half:]]
                ).astype(int)
                fdecs.append(
                    dl.first_decodable_layer_mat(
                        zf["acts"][idx_tr],
                        zf["slots"][idx_tr],
                        zf["acts"][idx_ev],
                        zf["slots"][idx_ev],
                    )
                    if position == "answer"
                    else 0
                )
            err_grid /= len(seeds)
            gi, li = np.unravel_index(err_grid.argmin(), err_grid.shape)
            lam = dl.LAM_GRID[int(gi)]
            layer = int(li)
            if position == "answer":
                layer = min(layer, int(max(fdecs)))
            # ---- nulls on cal split, pooled across seeds ------------------
            nulls = {}
            for s in seeds:
                zf = load_set(s, arm, "disc_P_fit")
                zt = load_set(s, arm, "disc_P_transfer")
                zg = load_set(s, arm, "disc_G_fit")
                calf, _ = cal_test_split(zf["strict_ok"])
                calt, _ = cal_test_split(zt["strict_ok"])
                calg, _ = cal_test_split(zg["strict_ok"])
                # Amendment 2: below-floor transfer cal -> P_fit cal fallback
                t_ok = len(calt) >= 20
                ze, cale = (zt, calt) if t_ok else (zf, calf)
                H = states_at(ze, position, cale, layer)
                model = dl.load_model(
                    s,
                    arm,
                    ROOT / f"results/trained_organisms/large/finetune/" f"seed{s}/{arm}/ckpt.pt",
                )
                evt = dict(
                    tokens=ze["tokens"],
                    answer_pos=ze["answer_pos"],
                    candidates=ze["candidates"],
                    carry_pos=ze["carry_pos"],
                    answers=ze["answers"],
                )
                for mode, nseeds in (
                    ("shuffled", CONFIG["null_seeds"]),
                    ("identity", [0]),
                ):
                    for ns in nseeds:
                        Rs = dl.fit_generators(
                            xy_fn(zf, position, layer),
                            calf,
                            lam,
                            null_mode=mode,
                            null_seed=2000 * ns + layer,
                        )
                        m = {}
                        if t_ok:
                            m["content_transfer_err"] = dl.pair_error_mat(
                                Rs, xy_fn(zt, position, layer), calt
                            )
                        if len(calg) >= 20:
                            m["crosspath_err"] = dl.pair_error_mat(
                                Rs, xy_fn(zg, position, layer), calg
                            )
                        m.update(dl.group_law_metrics_mat(Rs, H))
                        m["support_mass_lex"] = dl.support_mass_lex_mat(
                            Rs, model, ze, cale
                        )["mass"]
                        m.update(
                            dl.transport_agreement(
                                model,
                                Rs,
                                evt,
                                ze["acts"],
                                ze["carr"],
                                cale,
                                layer,
                                position,
                                per_base=CONFIG["c5_per_base"],
                            )
                        )
                        src = "shuffled" if mode == "shuffled" else "identity"
                        for metric, v in m.items():
                            need = CONFIG["null_sources"].get(metric, "")
                            if src in need:
                                nulls.setdefault(metric, []).append(float(v))
            from shared.calibrate import _quantile_thresholds

            thresholds = _quantile_thresholds(nulls, fpr=CONFIG["fpr"])
            # transport_agree: pre-committed 0.95q of all null transports
            ta = nulls.get("transport_agree", [])
            if ta:
                thresholds["transport_agree"] = dict(
                    dir="gt", tau=float(np.quantile(ta, 0.95)), null_n=len(ta)
                )
            out["cells"][f"{arm}/{position}"] = dict(
                seeds=seeds,
                lam=lam,
                layer=layer,
                first_decodable=[int(f) for f in fdecs],
                cal_err_grid=err_grid.tolist(),
                thresholds=thresholds,
                null_counts={k: len(v) for k, v in nulls.items()},
            )
            print(
                f"cell {arm}/{position}: lam={lam} layer={layer} "
                f"thresholds={ {k: round(v['tau'], 6) for k, v in thresholds.items()} }"
            )
    path = OUT / "disc_thresholds.json"
    json.dump(out, open(path, "w"), indent=1)
    h = sha256_file(path)
    (OUT / "disc_thresholds.sha256").write_text(h + "  disc_thresholds.json\n")
    print(f"FROZEN thresholds sha256={h}")


if __name__ == "__main__":
    stage = sys.argv[1]
    if stage == "freeze":
        freeze()
    elif stage == "cache":
        cache(int(sys.argv[2]), sys.argv[3])
    elif stage == "calibrate":
        calibrate()
    else:
        raise SystemExit(
            f"unknown stage {stage} (test/compare in " f"phase10_trackB_disc_test.py)"
        )

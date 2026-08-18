"""Track B discriminator — TEST stage (single evaluation) + arm comparison.

Runs only after: disc_committed_config.json frozen (sha verified),
disc_thresholds.json frozen (sha verified). Test bases are read here for the
first time. Per organism, per position class: fit on disc_P_fit TEST bases at
the frozen (lam, layer); evaluate C1/C2/C3/C4/C5/C6 against frozen
thresholds; C7 descriptive; frozen verdict logic; bootstrap CIs over base
problems (10k, seed 20260807) on C1/C2.

Usage:
  .venv/bin/python experiments/phase10_trackB_disc_test.py test
  .venv/bin/python experiments/phase10_trackB_disc_test.py compare
"""

import json
import pathlib
import sys
from itertools import combinations

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import trained_discriminator as dl  # noqa: E402
from shared.discriminator import evaluate_conditions, verdict_and_score  # noqa: E402
from large_discriminator_paths import (  # noqa: E402
    OUT,
    load_set,
    cal_test_split,
    xy_fn,
    states_at,
    sha256_file,
    verify_cfg,
)

PERMS3 = dl.PERMS3
BOOT_N, BOOT_SEED = 10000, 20260807


def per_base_ratio_components(Rs, get_XY, bases):
    """Per-generator, per-base numerator/denominator of the frozen
    pair_error, for bootstrap over base problems."""
    comps = {}
    for a, R in Rs.items():
        nums, dens = [], []
        for b in bases:
            X, Y = get_XY(np.array([b]), a)
            X, Y = X.astype(np.float64), Y.astype(np.float64)
            nums.append(((X @ R.T - Y) ** 2).sum())
            dens.append(max(((Y - X) ** 2).sum(), 1e-12))
        comps[a] = (np.array(nums), np.array(dens))
    return comps


def boot_ci(comps, n_bases):
    rng = np.random.default_rng(BOOT_SEED)
    idx = rng.integers(0, n_bases, size=(BOOT_N, n_bases))
    vals = np.zeros(BOOT_N)
    for a, (nums, dens) in comps.items():
        vals += nums[idx].sum(axis=1) / dens[idx].sum(axis=1)
    vals /= len(comps)
    return [float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))]


def content_labels(z, idx):
    """Queried property class (episode-level): the query-arg token at
    answer_pos - 1, minus PROP0. Valid for Q_P sets."""
    import trained_organism_data as od

    args = z["tokens"][idx, z["answer_pos"][idx] - 1]
    return (args - od.PROP0).astype(int)


def spectrum_diagnostic(Rs, zf, calf, layer, arm, position, lam):
    """8B carry-forward (design §7): slot-conditional means on the CAL split,
    rank cut 0.10, restricted generators, invariant-subspace dimension with
    tau_inv = 0.05q of shuffled-null stacked min-SVs. Only called when H_role
    fires; descriptive."""
    idx = np.concatenate([[dl.ep_index(b, g) for g in PERMS3] for b in calf]).astype(
        int
    )
    S = (
        zf["acts"][idx, layer]
        if position == "answer"
        else zf["carr"][idx][:, :, layer].reshape(-1, dl.D)
    )
    slots = (
        zf["slots"][idx]
        if position == "answer"
        else np.array([PERMS3[e % 6][s] for e in idx for s in range(3)])
    )
    u = np.stack([S[slots == j].mean(0) for j in range(3)])
    U, sv, Vt = np.linalg.svd(u, full_matrices=False)
    r = int((sv >= 0.10 * sv[0]).sum())
    B = Vt[:r].T
    A = {a: B.T @ Rs[a] @ B for a in Rs}
    eigs = {
        str(a): sorted(np.linalg.eigvals(Ag).tolist(), key=lambda z: -abs(z))
        for a, Ag in A.items()
    }
    stacked = np.concatenate([Ag - np.eye(r) for Ag in A.values()])
    min_sv = float(np.linalg.svd(stacked, compute_uv=False).min())
    null_min_svs = []
    for ns in range(10):
        Rn = dl.fit_generators(
            xy_fn(zf, position, layer),
            calf,
            lam,
            null_mode="shuffled",
            null_seed=2000 * ns + layer,
        )
        An = np.concatenate([B.T @ R @ B - np.eye(r) for R in Rn.values()])
        null_min_svs.append(float(np.linalg.svd(An, compute_uv=False).min()))
    tau_inv = float(np.quantile(null_min_svs, 0.05))
    return dict(
        span_dim=r,
        sv_ratios=[float(s / sv[0]) for s in sv],
        eigenvalues={
            k: [[float(z.real), float(z.imag)] for z in v] for k, v in eigs.items()
        },
        stacked_min_sv=min_sv,
        tau_inv=tau_inv,
        inv_dim=int(min_sv < tau_inv),
    )


def test():
    cfg_sha = verify_cfg()
    thr = json.load(open(OUT / "disc_thresholds.json"))
    thr_sha = sha256_file(OUT / "disc_thresholds.json")
    rec = (OUT / "disc_thresholds.sha256").read_text().split()[0]
    assert thr_sha == rec, "thresholds sha mismatch"
    gated = [tuple(x) for x in thr["gated"]]
    print(f"config {cfg_sha[:12]}… thresholds {thr_sha[:12]}… gated={gated}")

    results = {}
    for seed, arm in gated:
        zf = load_set(seed, arm, "disc_P_fit")
        zt = load_set(seed, arm, "disc_P_transfer")
        zg = load_set(seed, arm, "disc_G_fit")
        ztj = load_set(seed, arm, "disc_P_transfer_joint")
        calf, testf = cal_test_split(zf["strict_ok"])
        calt, testt = cal_test_split(zt["strict_ok"])
        calg, testg = cal_test_split(zg["strict_ok"])
        model = dl.load_model(
            seed,
            arm,
            ROOT / f"results/trained_organisms/large/finetune/" f"seed{seed}/{arm}/ckpt.pt",
        )
        # Amendment 2: min-universe floor + fallbacks
        FLOOR = 20
        t_ok = len(testt) >= FLOOR
        g_ok = len(testg) >= FLOOR
        # fallback eval source where the transfer universe is below floor
        ze, teste, eval_src = (
            (zt, testt, "P_transfer")
            if t_ok
            else (zf, testf, "P_fit_FALLBACK_in_sample")
        )
        org = {}
        for position in ("answer", "entity_mention_carry"):
            cell = thr["cells"][f"{arm}/{position}"]
            lam, layer = cell["lam"], cell["layer"]
            taus = cell["thresholds"]
            Rs = dl.fit_generators(xy_fn(zf, position, layer), testf, lam)
            m = {}
            if t_ok:
                comps1 = per_base_ratio_components(
                    Rs, xy_fn(zt, position, layer), testt
                )
                m["content_transfer_err"] = float(
                    np.mean([n.sum() / d.sum() for n, d in comps1.values()])
                )
                ci1 = boot_ci(comps1, len(testt))
            else:
                m["content_transfer_err"] = None
                ci1 = None
            if g_ok:
                comps2 = per_base_ratio_components(
                    Rs, xy_fn(zg, position, layer), testg
                )
                m["crosspath_err"] = float(
                    np.mean([n.sum() / d.sum() for n, d in comps2.values()])
                )
                ci2 = boot_ci(comps2, len(testg))
            else:
                m["crosspath_err"] = None
                ci2 = None
            H = states_at(ze, position, teste, layer)
            m.update(dl.group_law_metrics_mat(Rs, H))
            sup = dl.support_mass_lex_mat(Rs, model, ze, teste)
            m["support_mass_lex"] = sup["mass"]
            m["eff_rank"] = sup["eff_rank"]  # diagnostic only
            evt = dict(
                tokens=ze["tokens"],
                answer_pos=ze["answer_pos"],
                candidates=ze["candidates"],
                carry_pos=ze["carry_pos"],
            )
            m.update(
                dl.transport_agreement(
                    model,
                    Rs,
                    evt,
                    ze["acts"],
                    ze["carr"],
                    teste,
                    layer,
                    position,
                    per_base=2,
                )
            )
            # C6 probes: train on P/fit CAL, eval on P/transfer TEST
            # (amendment 2 fallback: eval on P/fit TEST where below floor)
            idx_tr = np.concatenate(
                [[dl.ep_index(b, g) for g in PERMS3] for b in calf]
            ).astype(int)
            idx_ev = np.concatenate(
                [[dl.ep_index(b, g) for g in PERMS3] for b in teste]
            ).astype(int)
            if position == "answer":
                Str, Sev = zf["acts"][idx_tr, layer], ze["acts"][idx_ev, layer]
                role_tr, role_ev = zf["slots"][idx_tr], ze["slots"][idx_ev]
                c_tr, c_ev = content_labels(zf, idx_tr), content_labels(ze, idx_ev)
            else:
                k = 3
                Str = zf["carr"][idx_tr][:, :, layer].reshape(-1, dl.D)
                Sev = ze["carr"][idx_ev][:, :, layer].reshape(-1, dl.D)
                gidx_tr = np.array(
                    [PERMS3[i % 6] for i in range(len(idx_tr)) for _ in range(k)]
                )

                # role label at carry slot s of episode g is g[s]
                def carry_roles(idx):
                    out = []
                    for e in idx:
                        g = PERMS3[e % 6]
                        out.extend([g[s] for s in range(k)])
                    return np.array(out)

                role_tr, role_ev = carry_roles(idx_tr), carry_roles(idx_ev)
                c_tr = np.repeat(content_labels(zf, idx_tr), k)
                c_ev = np.repeat(content_labels(ze, idx_ev), k)
            m.update(dl.probe_metrics_mat(Rs, Str, Sev, c_tr, c_ev, role_tr, role_ev))
            # C7 descriptive: joint transport error (not a condition)
            zj = ztj if t_ok else load_set(seed, arm, "disc_P_fit_joint")
            joint_err = dl.pair_error_mat(Rs, xy_fn(zj, position, layer), teste)
            conds = evaluate_conditions(m, taus)
            verdict, score = verdict_and_score(conds, m)
            spec = (
                spectrum_diagnostic(Rs, zf, calf, layer, arm, position, lam)
                if verdict == "H_role"
                else None
            )
            org[position] = dict(
                spectrum=spec,
                lam=lam,
                layer=layer,
                metrics=m,
                ci_content_transfer=ci1,
                ci_crosspath=ci2,
                joint_transport_err_descriptive=float(joint_err),
                conditions=conds,
                verdict=verdict,
                score=score,
                n_test_bases=dict(
                    P_fit=len(testf), P_transfer=len(testt), G_fit=len(testg)
                ),
                eval_source=eval_src,
            )
            c1s = (
                "n/a"
                if m["content_transfer_err"] is None
                else f"{m['content_transfer_err']:.4f}{ci1}"
            )
            c2s = (
                "n/a"
                if m["crosspath_err"] is None
                else f"{m['crosspath_err']:.4f}{ci2}"
            )
            print(
                f"seed{seed}/{arm} {position}: verdict={verdict} "
                f"C1={c1s} C2={c2s} "
                f"nontriv={m['nontriv']:.4g} inv={m['law_inv_defect']:.4g} "
                f"C5={m['transport_agree']:.3f} joint={joint_err:.4f} src={eval_src}"
            )
        both = (
            org["answer"]["verdict"] == "H_role"
            and org["entity_mention_carry"]["verdict"] == "H_role"
        )
        org["combined_verdict"] = (
            "H_role"
            if both
            else (
                "inconclusive"
                if "inconclusive"
                in (org["answer"]["verdict"], org["entity_mention_carry"]["verdict"])
                else "H_retrieval"
            )
        )
        results[f"{seed}/{arm}"] = org
        print(f"  seed{seed}/{arm} COMBINED: {org['combined_verdict']}")

    path = OUT / "disc_test_results.json"
    json.dump(
        dict(config_sha256=cfg_sha, thresholds_sha256=thr_sha, results=results),
        open(path, "w"),
        indent=1,
    )
    print(f"wrote {path}")


def fisher_one_tailed(a, b, c, d):
    """P(X >= a) for the 2x2 [[a,b],[c,d]] under the hypergeometric null
    (margins fixed): one-tailed, R-favouring."""
    from math import comb

    n, r1, c1 = a + b + c + d, a + b, a + c
    denom = comb(n, r1)
    p = 0.0
    for x in range(a, min(r1, c1) + 1):
        if r1 - x <= n - c1:
            p += comb(c1, x) * comb(n - c1, r1 - x) / denom
    return p


def mann_whitney_exact(x, y):
    """Exact one-tailed MWU: P(rank statistic as extreme as observed) under
    permutation, alternative x < y. Small n — full enumeration."""
    nx, pooled = len(x), np.array(list(x) + list(y))
    obs = sum(1 for xi in x for yj in y if xi < yj) + 0.5 * sum(
        1 for xi in x for yj in y if xi == yj
    )
    stats = []
    for c in combinations(range(len(pooled)), nx):
        xs = pooled[list(c)]
        ys = np.delete(pooled, list(c))
        u = sum(1 for xi in xs for yj in ys if xi < yj) + 0.5 * sum(
            1 for xi in xs for yj in ys if xi == yj
        )
        stats.append(u)
    stats = np.array(stats)
    return float((stats >= obs).mean())


def compare():
    res = json.load(open(OUT / "disc_test_results.json"))["results"]
    by_arm = {"R": [], "C": []}
    errs = {"R": [], "C": []}
    for key, org in res.items():
        seed, arm = key.split("/")
        by_arm[arm].append(org["combined_verdict"])
        errs[arm].append(org["answer"]["metrics"]["content_transfer_err"])
    nR, nC = len(by_arm["R"]), len(by_arm["C"])
    hR = sum(v == "H_role" for v in by_arm["R"])
    hC = sum(v == "H_role" for v in by_arm["C"])
    p_fisher = fisher_one_tailed(hR, nR - hR, hC, nC - hC)
    p_mwu = (
        mann_whitney_exact(errs["R"], errs["C"]) if errs["R"] and errs["C"] else None
    )
    out = dict(
        n_gated=dict(R=nR, C=nC),
        H_role_both=dict(R=hR, C=hC),
        verdicts=by_arm,
        answer_content_transfer_err=errs,
        fisher_one_tailed_p=p_fisher,
        mwu_one_tailed_p=p_mwu,
    )
    json.dump(out, open(OUT / "arm_comparison.json", "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    {"test": test, "compare": compare}[sys.argv[1]]()

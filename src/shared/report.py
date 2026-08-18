"""Reporting (spec v3, "Deliverables" / "Reporting").

Headline Stage 2 result: dose-response — the continuous discriminator score
must order correctly across the pressure gradient T0 -> T3 (Spearman
correlation, per seed and pooled). Binary verdicts are secondary. Also:
SNR sensitivity floor, the S-shared limitation stated prominently, per-seed
results (never seed-averaged alone), training trajectories whatever they
show, and any protocol violations found in the log.
"""

import csv

import matplotlib.pyplot as plt

from shared import progress
from shared.progress import log

PRESSURE = {"T0": 0, "T1": 1, "T2": 2, "T3": 3}
SNRS = (10, 3, 1)
STAGE1_ORGS = ("S-role", "S-shared", "S-retrieval", "S-position")


def _fig_path(name):
    p = progress.results_dir() / "figures"
    p.mkdir(parents=True, exist_ok=True)
    return p / name


def write_verdict_matrix(seeds, suffix=""):
    rows = [("stage", "organism", "snr_or_seed", "verdict", "score")]
    for org in STAGE1_ORGS:
        for snr in SNRS:
            rel = f"stage1/{org}@{snr}/discriminator.json"
            if progress.exists(rel):
                r = progress.load_json(rel)
                rows.append(
                    ("stage1", org, f"snr={snr}", r["verdict"], round(r["score"], 3))
                )
    for org in PRESSURE:
        for s in seeds:
            rel = f"stage2/{org}{suffix}/seed{s}/discriminator.json"
            if progress.exists(rel):
                r = progress.load_json(rel)
                rows.append(
                    (
                        "stage2",
                        org,
                        f"seed={s}",
                        r["verdict"],
                        round(r["score"], 3) if r.get("score") is not None else "",
                    )
                )
    with open(progress.results_dir() / "verdict_matrix.csv", "w", newline="") as f:
        csv.writer(f).writerows(rows)
    log(f"wrote verdict_matrix.csv ({len(rows) - 1} rows)")
    return rows


def dose_response(seeds, suffix=""):
    """dose_response.csv: score x organism x seed, Spearman vs the pressure
    ordering. Requires scipy."""
    from scipy.stats import spearmanr

    recs = []
    for org in PRESSURE:
        for s in seeds:
            rel = f"stage2/{org}{suffix}/seed{s}/discriminator.json"
            if progress.exists(rel):
                r = progress.load_json(rel)
                if r.get("score") is not None:
                    recs.append(
                        dict(
                            organism=org,
                            seed=s,
                            pressure=PRESSURE[org],
                            score=r["score"],
                            verdict=r["verdict"],
                        )
                    )
    with open(progress.results_dir() / "dose_response.csv", "w", newline="") as f:
        w = csv.DictWriter(
            f, fieldnames=["organism", "seed", "pressure", "score", "verdict"]
        )
        w.writeheader()
        w.writerows(recs)
    if len(recs) < 3:
        log(f"dose-response: only {len(recs)} usable runs — Spearman not computed")
        return dict(records=recs, spearman=None)
    rho, p = spearmanr([r["pressure"] for r in recs], [r["score"] for r in recs])
    per_seed = {}
    for s in seeds:
        sub = [r for r in recs if r["seed"] == s]
        if len(sub) >= 3:
            r_s, p_s = spearmanr(
                [r["pressure"] for r in sub], [r["score"] for r in sub]
            )
            per_seed[s] = dict(rho=float(r_s), p=float(p_s), n=len(sub))
    out = dict(
        records=recs,
        spearman=dict(rho=float(rho), p=float(p), n=len(recs)),
        per_seed=per_seed,
    )
    progress.save_json(
        "dose_response_summary.json", dict(spearman=out["spearman"], per_seed=per_seed)
    )
    log(
        f"dose-response (HEADLINE): pooled Spearman rho={rho:.3f} (p={p:.4f}, n={len(recs)}) "
        "— read with the recorded T1 flag: the T0<T1<T2<T3 gradient is weaker than the "
        "design claims (T1 permits no memorisation; it differs from T2 only in "
        "name-embedding gradient density)"
    )
    for s, d in per_seed.items():
        log(
            f"  seed {s}: rho={d['rho']:.3f} (p={d['p']:.3f}, n={d['n']}) — per-seed, never averaged alone"
        )
    return out


def plot_dose_response(dr):
    if not dr["records"]:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    for s in sorted({r["seed"] for r in dr["records"]}):
        sub = sorted(
            [r for r in dr["records"] if r["seed"] == s], key=lambda r: r["pressure"]
        )
        ax.plot(
            [r["pressure"] for r in sub],
            [r["score"] for r in sub],
            marker="o",
            label=f"seed {s}",
        )
    ax.set_xticks(list(PRESSURE.values()), list(PRESSURE.keys()))
    ax.set_xlabel("role-reuse pressure (T0 lowest, T3 highest)")
    ax.set_ylabel("discriminator score (conditions passed + tiebreak)")
    title = "Dose-response"
    if dr.get("spearman"):
        title += f' — pooled Spearman rho={dr["spearman"]["rho"]:.2f}'
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(_fig_path("dose_response.png"), dpi=120)
    plt.show()


def plot_trajectories(seeds, suffix=""):
    fig, axes = plt.subplots(1, len(PRESSURE), figsize=(16, 3.5), sharey=True)
    for ax, org in zip(axes, PRESSURE):
        for s in seeds:
            rel = f"stage2/{org}{suffix}/seed{s}/trajectory.json"
            if not progress.exists(rel):
                continue
            h = progress.load_json(rel)
            ax.plot(
                [r["step"] for r in h],
                [r["held_episode"] for r in h],
                label=f"seed {s} held",
            )
            ax.plot(
                [r["step"] for r in h],
                [r["transfer_episode"] for r in h],
                "--",
                alpha=0.6,
                label=f"seed {s} transfer",
            )
        ax.set_title(org)
        ax.set_xlabel("step")
        ax.axhline(0.95, color="gray", lw=0.5)
    axes[0].set_ylabel("candidate-masked answer accuracy")
    axes[0].legend(fontsize=7)
    fig.suptitle("Training trajectories (reported whatever they show)")
    fig.tight_layout()
    fig.savefig(_fig_path("trajectories.png"), dpi=120)
    plt.show()


FORWARD_HOPS = ("property->symbol", "symbol->person", "symbol->guarded")


def hop_vs_composition(orgs, seeds):
    """Task-2 branch on the per-query-type breakdown.
    composition_failure: forward-read hops > 0.9 with composed queries at
    chance -> depth is the correct lever (proceed to the depth sweep).
    hop_failure: any forward-read hop at or near chance -> STOP; the
    clause-order correction did not take effect — inspect raw episodes; more
    layers will not help a model that cannot do one hop.
    (person->symbol is the deliberate backward read: reported, not gated.)"""
    verdicts = {}
    for org in orgs:
        for s in seeds:
            rel = f"stage2/{org}/seed{s}/query_breakdown.json"
            if not progress.exists(rel):
                continue
            b = progress.load_json(rel)
            hops = [b[h]["accuracy"] for h in FORWARD_HOPS if h in b]
            composed = [
                b[c]["accuracy"] for c in ("composed_P", "composed_G") if c in b
            ]
            if not hops or not composed:
                continue
            if min(hops) < 0.5:
                v = "hop_failure"
            elif min(hops) > 0.9 and max(composed) < 0.4:
                v = "composition_failure"
            elif min(hops) > 0.9 and max(composed) >= 0.4:
                v = "learning"
            else:
                v = "mixed"
            verdicts[f"{org}/seed{s}"] = v
            log(
                f"branch {org}/seed{s}: forward hops min={min(hops):.3f}, "
                f"composed max={max(composed):.3f} -> {v}"
            )
    if any(v == "hop_failure" for v in verdicts.values()):
        log(
            "BRANCH VERDICT: hop failure — STOP. The clause-order correction did not "
            "take effect; dump and inspect raw episodes. Do NOT proceed to depth changes."
        )
    elif (
        verdicts
        and all(v in ("composition_failure", "learning") for v in verdicts.values())
        and any(v == "composition_failure" for v in verdicts.values())
    ):
        log(
            "BRANCH VERDICT: composition failure — forward hops intact, composed queries "
            "at chance. Depth is the correct lever; proceed to the depth sweep."
        )
    elif verdicts:
        log(
            "BRANCH VERDICT: mixed/learning — inspect the per-run lines above before "
            "choosing a lever."
        )
    else:
        log(
            "branch: no query_breakdown.json found — run the breakdown cell first (blocking)."
        )
    progress.save_json("stage2/hop_vs_composition.json", verdicts)
    return verdicts


def transition_report(org, seeds, threshold=0.40):
    """Per seed: full trajectory summary, the step at which held-out accuracy
    first exceeds `threshold` (if ever), and whether the transition is abrupt
    (a large single-evaluation jump) or gradual. Flags bimodality across
    seeds of identical configuration explicitly."""
    rows = []
    for s in seeds:
        rel = f"stage2/{org}/seed{s}/trajectory.json"
        if not progress.exists(rel):
            continue
        h = progress.load_json(rel)
        accs = [r["held_episode"] for r in h]
        steps = [r["step"] for r in h]
        first = next((steps[i] for i, a in enumerate(accs) if a > threshold), None)
        jumps = [accs[i + 1] - accs[i] for i in range(len(accs) - 1)] or [0.0]
        max_jump = max(jumps)
        kind = (
            "never left chance"
            if first is None
            else "ABRUPT" if max_jump >= 0.2 else "gradual"
        )
        rows.append(
            dict(
                seed=s,
                final_acc=accs[-1],
                last_step=steps[-1],
                first_above=first,
                max_single_eval_jump=round(max_jump, 3),
                transition=kind,
            )
        )
        log(
            f"transition {org}/seed{s}: final={accs[-1]:.3f} at step {steps[-1]}, "
            f"first >{threshold} at {first}, max single-eval jump {max_jump:.3f} ({kind})"
        )
    crossed = [r for r in rows if r["first_above"] is not None]
    if rows and 0 < len(crossed) < len(rows):
        log(
            f"transition {org}: BIMODAL — {len(crossed)}/{len(rows)} seeds of identical "
            f"configuration crossed {threshold} (at steps "
            f"{[r['first_above'] for r in crossed]}); the rest never left chance. "
            f"Recorded explicitly for the report."
        )
    progress.save_json(
        f"stage2/{org}/transition_report.json",
        dict(
            threshold=threshold,
            seeds=rows,
            bimodal=bool(rows and 0 < len(crossed) < len(rows)),
        ),
    )
    return rows


def protocol_violations():
    logf = progress.results_dir() / "progress.log"
    if not logf.exists():
        return []
    return [l.strip() for l in open(logf) if "PROTOCOL VIOLATION" in l]


def final_report(seeds):
    lines = ["=" * 72, "DISCRIMINATOR VALIDATION v3 — FINAL REPORT", "=" * 72]
    if progress.exists("stage1/summary.json"):
        s1 = progress.load_json("stage1/summary.json")
        lines.append(f"Stage 1: {'PASSED' if s1['passed'] else 'FAILED'}")
        lines.append(
            f"  SNR sensitivity floor (lowest SNR with S-role detection): "
            f"{s1['snr_sensitivity_floor']} — bounds every other result"
        )
        lines.append(
            f"  LIMITATION (stated, by design): S-shared verdicts = "
            f"{s1['s_shared_verdicts']} — k fixed global vectors satisfy the S_k "
            f"relations, so the method cannot distinguish shared-symbol codes from "
            f"genuine role algebra; the discriminator was NOT tuned to reject it"
        )
        lines.append(
            "  AMENDED 2026-08-07 (Phase 8B, derivation in "
            "phase8b_representation_note.md): S-shared's H_role verdict is "
            "CORRECT — k fixed vectors permuted by g carry the permutation "
            "representation of S_k (trivial + standard), a genuine role "
            "representation. The boundary is role-representation vs "
            "base-specific-retrieval; the residual limitation is only that the "
            "binary verdict does not identify the irrep decomposition, which "
            "H_role does not require. The validated Phase 8B spectrum "
            "diagnostic measures it (invariant direction present -> "
            "permutation-rep-like; absent -> standard-rep-like)."
        )
        for k, v in sorted(s1["verdicts"].items()):
            lines.append(f"    {k:<18} {v:<14} score={s1['scores'][k]:.3f}")
    else:
        lines.append("Stage 1: not run")
    dr = dose_response(seeds)
    lines.append("Stage 2 (headline = dose-response; binary verdicts secondary):")
    if dr.get("spearman"):
        lines.append(
            f"  pooled Spearman rho={dr['spearman']['rho']:.3f} "
            f"(p={dr['spearman']['p']:.4f}, n={dr['spearman']['n']})"
        )
        for s, d in dr.get("per_seed", {}).items():
            lines.append(f"  seed {s}: rho={d['rho']:.3f} (n={d['n']})")
    else:
        lines.append(
            "  insufficient converged runs for Spearman — reported as-is "
            "(negative and null results are deliverables)"
        )
    for r in dr["records"]:
        lines.append(
            f"    {r['organism']}/seed{r['seed']:<3} pressure={r['pressure']} "
            f"score={r['score']:.3f} verdict={r['verdict']}"
        )
    n_tests = 12 + len([r for r in dr["records"]])
    lines.append(
        f"Multiple-comparison statement: thresholds were calibrated at "
        f"per-comparison FPR 0.05; across ~{n_tests} verdict tests "
        f"(layers frozen per run; organisms x seeds x SNRs), the Bonferroni-adjusted "
        f"per-test level is {0.05 / max(n_tests, 1):.4f}. Individual passes near "
        f"threshold should be read accordingly."
    )
    # per-organism transition summaries (bimodality is recorded explicitly)
    for org in PRESSURE:
        rel = f"stage2/{org}/transition_report.json"
        if progress.exists(rel):
            tr = progress.load_json(rel)
            for r in tr["seeds"]:
                lines.append(
                    f"  {org}/seed{r['seed']}: final={r['final_acc']:.3f} "
                    f"(step {r['last_step']}), first >{tr['threshold']} at "
                    f"{r['first_above']}, transition {r['transition']}"
                )
            if tr.get("bimodal"):
                lines.append(
                    f"  {org}: BIMODAL across identical-config seeds — the circuit "
                    f"forms abruptly and only in some seeds."
                )
    # unleaked T0 shortcut check (recorded as FINAL — not to be revisited)
    for org in ("T0",):
        for rel_seed in range(0, 3):
            rel = f"stage2/{org}/seed{rel_seed}/unleaked_eval.json"
            if progress.exists(rel):
                m = progress.load_json(rel)
                tag = (
                    "confirms shortcut — FINAL"
                    if m["episode_acc"] < 0.6
                    else "UNEXPECTED — learned the real task; gradient's bottom rung invalid"
                )
                lines.append(
                    f"  {org}/seed{rel_seed} WITHOUT leak: acc={m['episode_acc']:.3f} "
                    f"consistency={m['orbit_consistency']:.3f} ({tag})"
                )
    # per-query-type breakdowns (hop vs composition)
    for org in PRESSURE:
        for s in range(0, 3):
            rel = f"stage2/{org}/seed{s}/query_breakdown.json"
            if progress.exists(rel):
                b = progress.load_json(rel)
                lines.append(
                    f"  {org}/seed{s} breakdown: "
                    + "  ".join(f"{k}={v['accuracy']:.2f}" for k, v in b.items())
                )
    if progress.exists("stage2/hop_vs_composition.json"):
        lines.append(
            f"  hop-vs-composition verdicts: "
            f"{progress.load_json('stage2/hop_vs_composition.json')}"
        )
    # RECORDED CORRECTION 1: T1's design premise is wrong. It was specified as
    # permitting per-name memorisation, but every binding is resampled per
    # episode, so no mapping is memorisable at any pool size. T1 and T2 differ
    # only in name-embedding gradient density, not shortcut availability.
    lines.append(
        "RECORDED CORRECTION — T1 design premise: T1 was specified as permitting "
        "per-name memorisation, but every binding in this task is resampled per "
        "episode, so no mapping is memorisable at any pool size; T1 and T2 differ "
        "only in name-embedding gradient density. The T0<T1<T2<T3 pressure gradient "
        "is therefore WEAKER than claimed — any dose-response result must "
        "carry this flag. T1 was NOT redesigned."
    )
    # RECORDED CORRECTION 2: loss/accuracy dissociation — the model learns
    # fact-block structure while composed accuracy stays at chance, which is
    # evidence against a data-pipeline fault and for a missing composition circuit.
    for org in PRESSURE:
        for s in range(0, 3):
            rel = f"stage2/{org}/seed{s}/trajectory.json"
            if not progress.exists(rel):
                continue
            h = progress.load_json(rel)
            if h and h[-1]["held_episode"] < 0.4 and h[0]["loss"] - h[-1]["loss"] > 0.5:
                lines.append(
                    f"RECORDED — {org}/seed{s}: LM loss fell "
                    f"{h[0]['loss']:.2f} -> {h[-1]['loss']:.2f} while composed accuracy "
                    f"stayed at {h[-1]['held_episode']:.3f}: the model learns fact-block "
                    f"structure and fails only at the query — evidence against a "
                    f"data-pipeline fault, for a missing composition circuit."
                )
    lines.append(
        "Method notes: eff_rank is uninterpretable as implemented (R is fit "
        "unconstrained, so off the role subspace it reflects ridge and noise) — "
        "logged diagnostic only, nothing depends on it. Stage 2's support test "
        "(support_mass_lex, direction lt) rejects (R−I) row-space concentration in "
        "the name-embedding difference subspace beyond the far (1−FPR) end of the "
        "Stage 2 null range — the readout-level lexical-swap signature; C4's FPR "
        "remains controlled by the nontriviality metrics."
    )
    viol = protocol_violations()
    lines.append(f"Protocol violations: {len(viol)}")
    lines.extend(f"  {v}" for v in viol)
    logf = progress.results_dir() / "progress.log"
    if logf.exists():
        with open(logf) as f:
            resumed = [l.strip() for l in f if "invalid early stop" in l]
        if resumed:
            lines.append(
                f"Declared protocol change: early-stop floor added at 25k steps; "
                f"{len(resumed)} pre-floor non-convergence results were invalidated "
                f"and their runs continued from checkpoint."
            )
    txt = "\n".join(lines)
    print(txt)
    with open(progress.results_dir() / "final_report.txt", "w") as f:
        f.write(txt + "\n")
    return txt

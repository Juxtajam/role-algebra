"""Phase 8A stratified gate recomputation (CPU-only,
2026-08-07).

Reuses the existing classification code path:
  - per-episode metadata fields written by phase8a_generate_tasks.episode()
    (answer, inner_name, third_name, qpos) — the fields the run-2
    error-category decomposition classified against:
        pred == answer      -> correct
        pred == inner_name  -> inner-clause (operator deletion)
        pred == third_name  -> opposite-endpoint
        else                -> other
  - orbit_metrics() imported from phase8a_modal (episode acc, strict-orbit,
    abstract-role orbit consistency), NOT reimplemented.
  - bootstrap CI methodology of run 1/2: 10,000 resamples over BASE problems,
    percentile 95% CI.

No episodes generated, no path P touched, no model, no GPU.
"""

import collections
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]  # repository root
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from gate_modal import GATE, orbit_metrics  # noqa: E402  (reused, not reimplemented)

TASKS = ROOT / "results/verdict/gate/tasks"  # run-2 task set (redesigned G)
PREDS = ROOT / "results/verdict/gate/preds"  # pulled from dv3-results:phase8a/
N_BOOT = 10_000
BOOT_SEED = 20260807  # deterministic; resampling unit = base problem (as run 1/2)


def load_cell(vocab):
    recs = [json.loads(l) for l in open(TASKS / f"gate_G_{vocab}.jsonl")]
    preds = json.load(open(PREDS / f"preds_gate_G_{vocab}.json"))
    assert len(recs) == len(preds) == 600, (len(recs), len(preds))
    return recs, preds


def classify(rec, pred):
    """Run-2 error-category decomposition, verbatim logic."""
    if pred == rec["answer"]:
        return "correct"
    if pred == rec["inner_name"]:
        return "inner_clause"
    if pred == rec["third_name"]:
        return "opposite_endpoint"
    return "other"


def queried_is_guards_subject(base):
    """Structural nesting condition, computed from the base's chain (not
    assumed from qpos): the queried sigil index is chain[qpos]; the guards
    clauses are chain[j] guards chain[j+1], j=0..K-2, so their SUBJECTS are
    chain[0..K-2]. True iff the queried sigil is the subject of some guards
    clause."""
    ch = base["chain"]
    subjects = set(ch[:-1])
    return ch[base["qpos"]] in subjects


def subset(recs, preds, base_ids):
    keep = [(r, p) for r, p in zip(recs, preds) if r["base_id"] in base_ids]
    return [r for r, _ in keep], [p for _, p in keep]


def boot_ci(recs, preds, n_boot=N_BOOT, seed=BOOT_SEED):
    """Bootstrap over base problems (run-2 methodology): resample bases with
    replacement, recompute episode acc and strict-orbit, percentile 95% CI."""
    bybase = collections.defaultdict(list)
    for r, p in zip(recs, preds):
        bybase[r["base_id"]].append((r, p))
    bases = sorted(bybase)
    ep_by_base = np.array(
        [np.mean([p == r["answer"] for r, p in bybase[b]]) for b in bases]
    )
    strict_by_base = np.array(
        [all(p == r["answer"] for r, p in bybase[b]) for b in bases], dtype=float
    )
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(bases), size=(n_boot, len(bases)))
    ep_stats = ep_by_base[idx].mean(axis=1)
    st_stats = strict_by_base[idx].mean(axis=1)
    return (
        [float(np.percentile(ep_stats, q)) for q in (2.5, 97.5)],
        [float(np.percentile(st_stats, q)) for q in (2.5, 97.5)],
    )


def cell_stats(recs, preds):
    acc, strict, cons = orbit_metrics(recs, preds)
    ep_ci, st_ci = boot_ci(recs, preds)
    n_bases = len({r["base_id"] for r in recs})
    return dict(
        episode_acc=acc,
        episode_acc_ci=ep_ci,
        strict_orbit=strict,
        strict_orbit_ci=st_ci,
        orbit_consistency=cons,
        n_bases=n_bases,
        n_eps=len(recs),
    )


def gate_eval(stats):
    comp = dict(
        episode_acc=stats["episode_acc"] >= GATE["episode_acc"],
        strict_orbit=stats["strict_orbit"] >= GATE["strict_orbit"],
        orbit_consistency=stats["orbit_consistency"] >= GATE["orbit_consistency"],
    )
    return comp, all(comp.values())


def main():
    out = {"gate_thresholds": GATE, "n_boot": N_BOOT, "boot_seed": BOOT_SEED}

    # ---------- step 1+verification: load, reproduce run-2 numbers ----------
    cells = {}
    print("=== verification: reproduce run-2 headline metrics + decomposition ===")
    for vocab in ("fit", "transfer"):
        recs, preds = load_cell(vocab)
        acc, strict, cons = orbit_metrics(recs, preds)
        cats = collections.Counter(classify(r, p) for r, p in zip(recs, preds))
        print(
            f"G/{vocab}: acc={acc:.4f} strict={strict:.4f} cons={cons:.4f} "
            f"cats={dict(cats)}"
        )
        cells[vocab] = (recs, preds)
        out.setdefault("run2_reproduction", {})[vocab] = dict(
            episode_acc=acc,
            strict_orbit=strict,
            orbit_consistency=cons,
            categories=dict(cats),
        )

    # expected from phase8a_gate_run2_report.md
    exp = {
        "fit": (
            0.9167,
            0.87,
            0.91,
            dict(correct=550, inner_clause=36, opposite_endpoint=10, other=4),
        ),
        "transfer": (
            0.8917,
            0.80,
            0.86,
            dict(correct=535, inner_clause=33, opposite_endpoint=30, other=2),
        ),
    }
    for vocab, (ea, so, oc, cats) in exp.items():
        recs, preds = cells[vocab]
        a, s, c = orbit_metrics(recs, preds)
        got = collections.Counter(classify(r, p) for r, p in zip(recs, preds))
        assert abs(a - ea) < 5e-4 and abs(s - so) < 1e-9 and abs(c - oc) < 1e-9, (
            vocab,
            a,
            s,
            c,
        )
        assert all(got[k] == v for k, v in cats.items()), (vocab, dict(got), cats)
    print("run-2 reproduction: EXACT MATCH on all metrics and category counts")

    # reproduce the '7 of 8 all-wrong inner-clause bases are qpos=1' finding
    allwrong = []
    for vocab in ("fit", "transfer"):
        recs, preds = cells[vocab]
        bybase = collections.defaultdict(list)
        for r, p in zip(recs, preds):
            bybase[r["base_id"]].append((r, p))
        for b, items in sorted(bybase.items()):
            if all(p != r["answer"] for r, p in items):
                cats = {classify(r, p) for r, p in items}
                allwrong.append(
                    dict(
                        vocab=vocab,
                        base_id=b,
                        qpos=items[0][0]["qpos"],
                        cats=sorted(cats),
                    )
                )
    inner_only = [w for w in allwrong if w["cats"] == ["inner_clause"]]
    n_q1 = sum(1 for w in inner_only if w["qpos"] == 1)
    print(
        f"all-wrong bases: {len(allwrong)}; all-six-inner-clause: "
        f"{len(inner_only)}; of those qpos=1: {n_q1}"
    )
    out["allwrong_bases"] = allwrong
    # Run-2 report prose says "8 of the 10 all-wrong bases fail as
    # inner-clause on ALL six permutations, and 7 of those 8 are qpos=1".
    # This recomputation — with the classification logic that reproduces the
    # run-2 episode-level category tables and per-base histograms EXACTLY —
    # yields 7 all-inner bases, ALL of them qpos=1 (the other 3 all-wrong
    # bases are pure opposite-endpoint at qpos=2). Declared discrepancy: the
    # run-2 prose count '8' appears to be a slip; '7 qpos=1 inner-clause
    # bases' is confirmed. This strengthens, not weakens, the stratification
    # rationale.
    assert (
        len(allwrong) == 10 and n_q1 == 7
    ), "failed to reproduce the core run-2 anatomy (10 all-wrong, 7 qpos=1 inner)"
    out["anatomy_discrepancy"] = dict(
        run2_report_claim="8 of 10 all-wrong bases all-six inner-clause; 7 of 8 qpos=1",
        recomputed=f"{len(inner_only)} of {len(allwrong)} all-wrong bases "
        f"all-six inner-clause; {n_q1} of {len(inner_only)} qpos=1; "
        f"remaining {len(allwrong) - len(inner_only)} all-wrong bases "
        f"are pure opposite-endpoint at qpos=2",
        note="episode-level category tables and per-base histograms reproduce "
        "run 2 exactly; the prose count '8' is the only non-reproducing "
        "figure and is recorded as a run-2 reporting slip",
    )
    print(
        f"anatomy recomputed: 10 all-wrong; {len(inner_only)} all-inner, "
        f"all qpos=1 (run-2 prose said 8/7 — declared discrepancy)"
    )

    # ---------- step 2: stratum definitions ----------
    # primary: qpos==1 AND queried sigil is subject of another guards clause
    # (nesting computed structurally from base['chain'], not assumed)
    # sensitivity: qpos==1 regardless of nesting
    out["strata"] = {}
    for vocab in ("fit", "transfer"):
        recs, _ = cells[vocab]
        base_info = {}
        for r in recs:
            b = r["base_id"]
            if b not in base_info:
                base_info[b] = dict(
                    qpos=r["qpos"], nested=queried_is_guards_subject(r["base"])
                )
        prim = {b for b, i in base_info.items() if i["qpos"] == 1 and i["nested"]}
        wide = {b for b, i in base_info.items() if i["qpos"] == 1}
        n_q1_nonnested = len(wide - prim)
        n_q2_nested = sum(
            1 for b, i in base_info.items() if i["qpos"] == 2 and i["nested"]
        )
        out["strata"][vocab] = dict(
            n_bases_total=len(base_info),
            primary_stratum_bases=sorted(prim),
            widened_stratum_bases=sorted(wide),
            primary_n=len(prim),
            widened_n=len(wide),
            qpos1_without_nesting=n_q1_nonnested,
            qpos2_with_nesting=n_q2_nested,
            strata_identical=prim == wide,
        )
        print(
            f"G/{vocab}: primary stratum {len(prim)}/100 bases; widened "
            f"{len(wide)}/100; qpos=1-without-nesting: {n_q1_nonnested}; "
            f"identical: {prim == wide}"
        )

    # ---------- steps 3-6: complement + stratum metrics, gate, sensitivity ----------
    for label, key in (
        ("primary", "primary_stratum_bases"),
        ("widened", "widened_stratum_bases"),
    ):
        out[label] = {}
        gate_pass_all = True
        for vocab in ("fit", "transfer"):
            recs, preds = cells[vocab]
            strat = set(out["strata"][vocab][key])
            all_bases = {r["base_id"] for r in recs}
            comp_recs, comp_preds = subset(recs, preds, all_bases - strat)
            st_recs, st_preds = subset(recs, preds, strat)

            comp_stats = cell_stats(comp_recs, comp_preds)
            comp_gate, comp_pass = gate_eval(comp_stats)
            gate_pass_all &= comp_pass

            st_stats = cell_stats(st_recs, st_preds) if strat else None

            # error-mass split (step 5): run-2 G errors inside vs outside stratum
            errs_in = sum(
                1
                for r, p in zip(recs, preds)
                if p != r["answer"] and r["base_id"] in strat
            )
            errs_out = sum(
                1
                for r, p in zip(recs, preds)
                if p != r["answer"] and r["base_id"] not in strat
            )
            out[label][vocab] = dict(
                complement=comp_stats,
                complement_gate=dict(components=comp_gate, cell_pass=comp_pass),
                stratum=st_stats,
                stratum_share_bases=len(strat) / len(all_bases),
                stratum_share_episodes=len(st_recs) / len(recs),
                errors_inside=errs_in,
                errors_outside=errs_out,
                error_mass_inside_frac=(
                    errs_in / (errs_in + errs_out) if errs_in + errs_out else None
                ),
            )
            print(
                f"[{label}] G/{vocab} complement: "
                f"acc={comp_stats['episode_acc']:.4f} "
                f"CI={comp_stats['episode_acc_ci']} "
                f"strict={comp_stats['strict_orbit']:.4f} "
                f"CI={comp_stats['strict_orbit_ci']} "
                f"cons={comp_stats['orbit_consistency']:.4f} "
                f"N={comp_stats['n_bases']} bases -> "
                f"{'PASS' if comp_pass else 'FAIL'} {comp_gate}"
            )
            if st_stats:
                print(
                    f"[{label}] G/{vocab} stratum:    "
                    f"acc={st_stats['episode_acc']:.4f} "
                    f"strict={st_stats['strict_orbit']:.4f} "
                    f"cons={st_stats['orbit_consistency']:.4f} "
                    f"N={st_stats['n_bases']} bases; "
                    f"errors in/out={errs_in}/{errs_out}"
                )
        out[label]["gate_overall_complement"] = gate_pass_all
        print(
            f"[{label}] complement gate overall (both vocabs): "
            f"{'PASS' if gate_pass_all else 'FAIL'}"
        )

    out["sensitivity_flag"] = dict(
        strata_identical_both_vocabs=all(
            out["strata"][v]["strata_identical"] for v in ("fit", "transfer")
        ),
        gate_outcomes_differ=(
            out["primary"]["gate_overall_complement"]
            != out["widened"]["gate_overall_complement"]
        ),
    )

    with open(ROOT / "results/verdict/gate/stratified_recheck.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nwritten: results/phase8a/stratified_recheck.json")


if __name__ == "__main__":
    main()

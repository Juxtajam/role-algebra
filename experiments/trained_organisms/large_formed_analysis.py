"""Phase 10 Track B — LOCKED outcome analysis (FORMED + McNemar).

Implements exactly the frozen rules in phase10/trackB/committed_config.json
(sha256 asserted below, recorded in committed_config.sha256):

  FORMED           held_composed_acc >= 0.5  sustained over 5 consecutive evals
  FORMED_STRICT    held_composed_acc >= 0.95 sustained over 5 consecutive evals
  aux_general      min over FORWARD-hop aux types >= 0.95 sustained over 5
  length_3         threehop_acc >= 0.5 sustained over 5
  unseen_perm      full_acc >= 0.5 sustained over 5   (closest logged quantity;
                   see interpretation notes in the output JSON)
  Primary test     McNemar exact binomial on discordant pairs, one-sided
                   (H1: R-forms/C-doesn't > C-forms/R-doesn't), n_pairs=12;
                   if discordant <= 5, report the raw 2x2 with NO test.

Sustained-over-5 rule, verbatim from the config: "threshold must be met or
exceeded at 5 consecutive eval points (every 500 steps); first eval to meet
threshold begins the count; if met at step N, N+500, N+1000, N+1500, N+2000
-> FORMED at step N".

Interpretation choices recorded (also embedded in the output JSON):
 1. "composed-query held-out accuracy" = trajectory field `held_composed_acc`
    (held-out fit-vocabulary composed orbits, full permutation set, k=3).
 2. "forward-hop single-hop aux types" = A_PS, A_SN, A_SG (A_NS is the
    backward read, excluded, matching the whole programme's forward/backward
    distinction). The all-types variant (incl. A_NS) is computed alongside as
    a diagnostic, not an outcome.
 3. "unseen-permutation-product accuracy" = trajectory field `full_acc`
    (full-S_k orbit eval; training saw generators only). Caveat recorded:
    the full orbit includes the identity and the two generators, so this
    DILUTES the unseen products with seen permutations; a pure unseen-only
    accuracy was not logged per eval. `gen_acc` (generators-only) is emitted
    alongside so the gap is visible.

Reads:  results/phase10/trackB/finetune/seed{S}/{R,C}/trajectory.json
Writes: results/phase10/trackB/formed_analysis.json
"""

import hashlib
import json
import pathlib
from math import comb

ROOT = pathlib.Path(__file__).resolve().parents[2]
FT = ROOT / "results/trained_organisms/large/finetune"
CFG = ROOT / "phase10/trackB/committed_config.json"
CFG_SHA_FILE = ROOT / "phase10/trackB/committed_config.sha256"
OUT = ROOT / "results/trained_organisms/large/formed_analysis.json"

SEEDS = [0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 13]
ARMS = ["R", "C"]
FORWARD_AUX = ["A_PS", "A_SN", "A_SG"]
SUSTAIN = 5


def sha256(p):
    return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()


def sustained(series, thresh, n=SUSTAIN):
    """First step at which `series` (list of (step, value)) meets `thresh`
    at n consecutive eval points; None if never. Missing/None values break
    the run of consecutive successes (they are not >= thresh)."""
    run, start = 0, None
    for step, v in series:
        if v is not None and v >= thresh:
            if run == 0:
                start = step
            run += 1
            if run >= n:
                return start
        else:
            run, start = 0, None
    return None


def outcomes_for(traj):
    def series(key):
        return [(r["step"], r.get(key)) for r in traj]

    def aux_series(types):
        out = []
        for r in traj:
            accs = r.get("aux_accs") or {}
            vals = [accs.get(t) for t in types]
            out.append(
                (r["step"], min(vals) if all(v is not None for v in vals) else None)
            )
        return out

    res = {}
    res["FORMED_at"] = sustained(series("held_composed_acc"), 0.5)
    res["FORMED_STRICT_at"] = sustained(series("held_composed_acc"), 0.95)
    res["aux_general_at"] = sustained(aux_series(FORWARD_AUX), 0.95)
    res["aux_general_alltypes_at"] = sustained(
        aux_series(FORWARD_AUX + ["A_NS"]), 0.95
    )  # diagnostic only
    res["length_3_at"] = sustained(series("threehop_acc"), 0.5)
    res["unseen_perm_at"] = sustained(series("full_acc"), 0.5)
    for k in list(res):
        res[k.replace("_at", "")] = res[k] is not None
    last = traj[-1]
    res["final_step"] = last["step"]
    res["n_evals"] = len(traj)
    res["final_held_composed_acc"] = last.get("held_composed_acc")
    res["final_full_acc"] = last.get("full_acc")
    res["final_gen_acc"] = last.get("gen_acc")
    res["final_threehop_acc"] = last.get("threehop_acc")
    res["final_aux_accs"] = last.get("aux_accs")
    return res


def add_formed_final(res, seed, arm):
    """D18 substitute primary outcome: pure Q_P held-out episode accuracy
    >= 0.5 at the final checkpoint (query_breakdown.json). Q_G and the OR
    variant reported as sensitivity. The trajectory-based outcomes above are
    MIXED-battery (D18) and retained as superseded diagnostics."""
    qb_path = FT / f"seed{seed}/{arm}/query_breakdown.json"
    if not qb_path.exists():
        res["FORMED_final"] = None
        return
    qb = json.load(open(qb_path))
    res["QP_final_acc"] = qb["Q_P"]["accuracy"]
    res["QG_final_acc"] = qb["Q_G"]["accuracy"]
    res["QP_final_strict"] = qb["Q_P"]["strict_orbit"]
    res["FORMED_final"] = qb["Q_P"]["accuracy"] >= 0.5
    res["FORMED_final_QG"] = qb["Q_G"]["accuracy"] >= 0.5
    res["FORMED_final_or"] = res["FORMED_final"] or res["FORMED_final_QG"]


def mcnemar(per_seed, outcome):
    """Exact one-sided McNemar on discordant pairs for a binary outcome.
    H1: P(R forms, C doesn't) > P(C forms, R doesn't)."""
    b = sum(
        1
        for s in per_seed
        if per_seed[s]["R"][outcome] and not per_seed[s]["C"][outcome]
    )
    c = sum(
        1
        for s in per_seed
        if per_seed[s]["C"][outcome] and not per_seed[s]["R"][outcome]
    )
    both = sum(
        1 for s in per_seed if per_seed[s]["R"][outcome] and per_seed[s]["C"][outcome]
    )
    neither = sum(
        1
        for s in per_seed
        if not per_seed[s]["R"][outcome] and not per_seed[s]["C"][outcome]
    )
    n = b + c
    table = dict(
        R_only=b,
        C_only=c,
        both=both,
        neither=neither,
        n_pairs=len(per_seed),
        discordant=n,
    )
    if n <= 5:
        table["test"] = (
            "NOT RUN per frozen caveat (discordant <= 5): "
            "the 2x2 table IS the result"
        )
        table["p_one_sided"] = None
    else:
        # P(X >= b) under Binomial(n, 0.5)
        p = sum(comb(n, k) for k in range(b, n + 1)) / 2**n
        table["test"] = "exact binomial one-sided on discordant pairs"
        table["p_one_sided"] = p
    return table


def main():
    cfg_sha = sha256(CFG)
    recorded = CFG_SHA_FILE.read_text().split()[0]
    assert cfg_sha == recorded, f"config sha mismatch: {cfg_sha} != {recorded}"
    print(f"committed_config verified sha256={cfg_sha[:16]}...")

    per_seed, missing = {}, []
    for s in SEEDS:
        row = {}
        for arm in ARMS:
            tj = FT / f"seed{s}/{arm}/trajectory.json"
            if not tj.exists():
                missing.append(f"seed{s}/{arm}")
                continue
            traj = json.load(open(tj))
            row[arm] = outcomes_for(traj)
            add_formed_final(row[arm], s, arm)
        if len(row) == 2:
            per_seed[s] = row

    print(f"pairs with both arms: {sorted(per_seed)}  missing: {missing}")

    complete = {
        s: v
        for s, v in per_seed.items()
        if v["R"]["final_step"] >= 50000 and v["C"]["final_step"] >= 50000
    }
    print(f"pairs complete to 50k: {sorted(complete)}")

    analyses = {}
    for outcome in [
        "FORMED_final",
        "FORMED_final_QG",
        "FORMED_final_or",
        "FORMED",
        "FORMED_STRICT",
        "aux_general",
        "length_3",
        "unseen_perm",
    ]:
        pool = {
            s: v
            for s, v in complete.items()
            if all(v[a].get(outcome) is not None for a in ARMS)
        }
        analyses[outcome] = mcnemar(pool, outcome) if pool else None

    out = dict(
        committed_config_sha256=cfg_sha,
        interpretation_notes={
            "D18": (
                "held_composed_acc and transfer/k3/k4/gen/full are "
                "MIXED-battery fields (~40% aux + ~60% composed; "
                "finetune_modal.py lines 228-236) and do NOT implement "
                "the frozen 'composed-query held-out accuracy'. The "
                "primary outcome is FORMED_final: pure Q_P episode acc "
                ">= 0.5 at the final checkpoint (query_breakdown.json), "
                "Trajectory-based FORMED retained as "
                "a superseded MIXED diagnostic."
            ),
            "composed_field_mixed": "held_composed_acc (MIXED, superseded)",
            "forward_aux_types": FORWARD_AUX,
            "unseen_perm_field": (
                "full_acc — MIXED-battery AND includes "
                "identity+generators; diagnostic only"
            ),
            "sustained_rule": "5 consecutive evals at 500-step spacing "
            "(applies to trajectory-based outcomes only)",
        },
        seeds_expected=SEEDS,
        pairs_missing=missing,
        pairs_analyzed=sorted(complete),
        per_seed={str(s): per_seed[s] for s in sorted(per_seed)},
        mcnemar=analyses,
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=1)
    print(f"wrote {OUT}")

    print("\nPer-pair table (FORMED_final = D18 primary; MIXED@ = superseded):")
    hdr = (
        f"{'seed':>4} | {'arm':>3} | {'QP_fin':>6} {'FRM':>4} | "
        f"{'QG_fin':>6} | {'MIXED@':>7} | {'aux@':>7} | {'3hop@':>7} | mix_fin"
    )
    print(hdr)
    for s in sorted(per_seed):
        for arm in ARMS:
            r = per_seed[s][arm]
            f = lambda k: str(r[k]) if r[k] is not None else "-"
            qp = r.get("QP_final_acc")
            qg = r.get("QG_final_acc")
            ff = r.get("FORMED_final")
            print(
                f"{s:>4} | {arm:>3} | "
                f"{qp:6.3f} {'YES' if ff else ('-' if ff is None else 'no'):>4} | "
                f"{qg:6.3f} | {f('FORMED_at'):>7} | {f('aux_general_at'):>7} | "
                f"{f('length_3_at'):>7} | {r['final_held_composed_acc']:.3f}"
                if qp is not None
                else f"{s:>4} | {arm:>3} | {'—':>6} {'-':>4} | {'—':>6} | "
                f"{f('FORMED_at'):>7} | {f('aux_general_at'):>7} | "
                f"{f('length_3_at'):>7} | {r['final_held_composed_acc']:.3f}"
            )
    print("\nMcNemar / 2x2 per outcome:")
    for k, v in analyses.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()

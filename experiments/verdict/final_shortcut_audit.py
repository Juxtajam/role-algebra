"""Phase 8A-final — Step 3.2: shortcut-audit ladder battery, every cell.

Runs on the stored 72B gate predictions (dv3-results:phase8a_final/) against
the FROZEN run-2 episodes; also reruns the identical battery on the stored
run-2 32B predictions so the same code produces both columns of the scale
comparison. Battery (run-2 ladder):
  1. position-match rates: pred == first-/last-mentioned name in the prompt,
     vs the gold base rate (answer == first/last), per cell.
  2. lexical selection per name: predicted-frequency vs gold-frequency per
     name; lexical co-location rate (pred == name whose holds-fact mentions
     the queried symbol; for G that is inner_name) vs gold co-location rate.
  3. direction-confusion split (G cells): correct / inner-clause /
     opposite-endpoint / other, via the run-2 classification fields.
  4. qpos stratification (G: qpos 1 vs 2; P: qi 0/1/2), episode acc each.
Writes results/phase8a/phase8a_final_shortcut_audit.json.
"""

import collections
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]  # repository root
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from gate_modal import orbit_metrics  # noqa: E402  frozen scoring, reused

TASKS = ROOT / "results/verdict/gate/tasks"
PREDS72 = ROOT / "results/verdict/answer_position"  # pulled from volume
PREDS32 = ROOT / "results/verdict/gate/preds"  # run-2 (G cells stored)
CELLS = [(p, v) for p in ("P", "G") for v in ("fit", "transfer")]


def classify(rec, pred):
    if pred == rec["answer"]:
        return "correct"
    if pred == rec.get("inner_name"):
        return "inner_clause"
    if pred == rec.get("third_name"):
        return "opposite_endpoint"
    return "other"


def mention_order(rec):
    """Names in order of first mention in the rendered user prompt."""
    names = rec["base"]["names"]
    pos = sorted(names, key=lambda n: rec["user"].index(n))
    return pos


def audit_cell(recs, preds):
    out = {}
    n = len(recs)
    # 1. position match
    first = [mention_order(r)[0] for r in recs]
    last = [mention_order(r)[-1] for r in recs]
    out["position_match"] = dict(
        pred_first=sum(p == f for p, f in zip(preds, first)) / n,
        gold_first=sum(r["answer"] == f for r, f in zip(recs, first)) / n,
        pred_last=sum(p == l for p, l in zip(preds, last)) / n,
        gold_last=sum(r["answer"] == l for r, l in zip(recs, last)) / n,
    )
    # 2. lexical selection per name + co-location
    pred_freq = collections.Counter(preds)
    gold_freq = collections.Counter(r["answer"] for r in recs)
    out["name_selection"] = {
        name: dict(pred=pred_freq.get(name, 0) / n, gold=g / n)
        for name, g in sorted(gold_freq.items())
    }
    out["max_abs_name_bias"] = max(
        abs(pred_freq.get(nm, 0) - g) / n for nm, g in gold_freq.items()
    )
    coloc = []
    for r in recs:
        if r["path"] == "G":
            coloc.append(r["inner_name"])  # holder of queried symbol
        else:
            qi = r["base"]["qi"]  # P: holder of queried symbol
            coloc.append(r["base"]["names"][r["g"][qi]])
    out["colocation"] = dict(
        pred_coloc=sum(p == c for p, c in zip(preds, coloc)) / n,
        gold_coloc=sum(r["answer"] == c for r, c in zip(recs, coloc)) / n,
    )
    # 3. direction-confusion split (G only; P gets correct/wrong)
    if recs[0]["path"] == "G":
        out["direction_split"] = dict(
            collections.Counter(classify(r, p) for r, p in zip(recs, preds))
        )
    else:
        out["direction_split"] = dict(
            collections.Counter(
                "correct" if p == r["answer"] else "wrong" for r, p in zip(recs, preds)
            )
        )
    # 4. qpos / qi stratification
    strat = collections.defaultdict(lambda: [0, 0])
    for r, p in zip(recs, preds):
        key = f"qpos={r['qpos']}" if r["path"] == "G" else f"qi={r['base']['qi']}"
        strat[key][0] += int(p == r["answer"])
        strat[key][1] += 1
    out["q_stratification"] = {
        k: dict(acc=c / t, n=t) for k, (c, t) in sorted(strat.items())
    }
    # per-base anatomy: all-wrong bases, their qpos, unanimous category
    bybase = collections.defaultdict(list)
    for r, p in zip(recs, preds):
        bybase[r["base_id"]].append((r, p))
    allwrong = []
    for b, items in sorted(bybase.items()):
        if all(p != r["answer"] for r, p in items):
            cats = sorted({classify(r, p) for r, p in items})
            allwrong.append(dict(base_id=b, qpos=items[0][0].get("qpos"), cats=cats))
    out["allwrong_bases"] = allwrong
    acc, strict, cons = orbit_metrics(recs, preds)
    out["metrics"] = dict(episode_acc=acc, strict_orbit=strict, orbit_consistency=cons)
    return out


def main():
    report = {}
    for tag, preds_dir, cells in (
        ("72B", PREDS72, CELLS),
        ("32B_run2", PREDS32, CELLS),
    ):
        report[tag] = {}
        for path, vocab in cells:
            recs = [json.loads(l) for l in open(TASKS / f"gate_{path}_{vocab}.jsonl")]
            fp = preds_dir / f"preds_gate_{path}_{vocab}.json"
            if not fp.exists():
                report[tag][f"{path}/{vocab}"] = "preds not stored"
                continue
            preds = json.load(open(fp))
            assert len(preds) == len(recs) == 600
            report[tag][f"{path}/{vocab}"] = audit_cell(recs, preds)
            a = report[tag][f"{path}/{vocab}"]
            print(
                f"[{tag}] {path}/{vocab}: acc={a['metrics']['episode_acc']:.4f} "
                f"pos(first p/g)={a['position_match']['pred_first']:.3f}/"
                f"{a['position_match']['gold_first']:.3f} "
                f"(last)={a['position_match']['pred_last']:.3f}/"
                f"{a['position_match']['gold_last']:.3f} "
                f"coloc={a['colocation']['pred_coloc']:.3f}/"
                f"{a['colocation']['gold_coloc']:.3f} "
                f"maxnamebias={a['max_abs_name_bias']:.3f} "
                f"split={a['direction_split']} "
                f"qstrat={ {k: round(v['acc'],3) for k,v in a['q_stratification'].items()} } "
                f"allwrong={len(a['allwrong_bases'])}"
            )
    out = ROOT / "results/verdict/gate/phase8a_final_shortcut_audit.json"
    json.dump(report, open(out, "w"), indent=2)
    print("written:", out)


if __name__ == "__main__":
    main()

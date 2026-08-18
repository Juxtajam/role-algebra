"""
D8 Shortcut Audit — Phase 10 Track A.
Reads per-episode CSVs for Nemotron, Llama-3.3, and Qwen2.5-72B gate results,
joins with frozen JSONL episodes for position analysis, and computes:
  1. Position-match rates (first / middle / last / queried)
  2. Lexical selection rates per name
  3. Direction-confusion split
  4. qpos stratification

Summary tables computed from stored bytes.
"""

import collections
import csv
import json
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parents[2]
TASKS = ROOT / "results/verdict/gate/tasks"  # Frozen JSONL episodes
OUT_DIR = ROOT / "results/cross_family/error_diagnostic"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Cell definitions: (path, vocab) -> [(model_label, csv_dir)]
CELLS = {
    ("P", "fit"): [
        ("Nemotron", ROOT / "results/cross_family/nemotron/gate/episodes_gate_P_fit.csv"),
        ("Llama-3.3", ROOT / "results/cross_family/llama/gate/episodes_gate_P_fit.csv"),
        ("Qwen2.5-72B", ROOT / "results/verdict/answer_position/episodes_gate_P_fit.csv"),
    ],
    ("P", "transfer"): [
        ("Nemotron", ROOT / "results/cross_family/nemotron/gate/episodes_gate_P_transfer.csv"),
        ("Llama-3.3", ROOT / "results/cross_family/llama/gate/episodes_gate_P_transfer.csv"),
        ("Qwen2.5-72B", ROOT / "results/verdict/answer_position/episodes_gate_P_transfer.csv"),
    ],
    ("G", "fit"): [
        ("Nemotron", ROOT / "results/cross_family/nemotron/gate/episodes_gate_G_fit.csv"),
        ("Llama-3.3", ROOT / "results/cross_family/llama/gate/episodes_gate_G_fit.csv"),
        ("Qwen2.5-72B", ROOT / "results/verdict/answer_position/episodes_gate_G_fit.csv"),
    ],
    ("G", "transfer"): [
        ("Nemotron", ROOT / "results/cross_family/nemotron/gate/episodes_gate_G_transfer.csv"),
        ("Llama-3.3", ROOT / "results/cross_family/llama/gate/episodes_gate_G_transfer.csv"),
        ("Qwen2.5-72B", ROOT / "results/verdict/answer_position/episodes_gate_G_transfer.csv"),
    ],
}


def load_jsonl(path):
    return [json.loads(line) for line in open(path)]


def mention_order(rec):
    """Names in order of first mention in the rendered user prompt."""
    names = rec["base"]["names"]
    user = rec["user"]
    # Find first occurrence position of each name
    # Some names might be substrings of others, so use word-boundary aware matching
    pos = []
    for n in names:
        # Escape regex chars, use word boundary
        pattern = r"\b" + re.escape(n) + r"\b"
        m = re.search(pattern, user)
        if m:
            pos.append((m.start(), n))
        else:
            pos.append((99999, n))
    pos.sort()
    return [n for _, n in pos]


def queried_name(rec):
    """The co-located name: for G this is inner_name, for P it's names[g[qi]]."""
    if rec["path"] == "G":
        return rec.get("inner_name", "")
    else:
        qi = rec["base"]["qi"]
        return rec["base"]["names"][rec["g"][qi]]


def classify(rec, pred):
    """Direction-confusion classification for G cells."""
    if pred == rec["answer"]:
        return "correct"
    if pred == rec.get("inner_name"):
        return "inner_clause"
    if pred == rec.get("third_name"):
        return "opposite_endpoint"
    return "other"


def audit_cell(recs, preds, path, vocab, model_label):
    """Compute full shortcut audit for one cell."""
    n = len(recs)
    out = {"model": model_label, "path": path, "vocab": vocab, "n": n}

    # 1. Position-match: compute gold and pred rates for first/middle/last/queried
    if path == "G":
        # Use inner_name / third_name from the JSONL for G
        queried = []
        firsts, middles, lasts = [], [], []
        for r in recs:
            mo = mention_order(r)
            firsts.append(mo[0])
            middles.append(mo[1])
            lasts.append(mo[2])
            queried.append(r.get("inner_name", ""))
    else:
        queried = []
        firsts, middles, lasts = [], [], []
        for r in recs:
            mo = mention_order(r)
            firsts.append(mo[0])
            middles.append(mo[1])
            lasts.append(mo[2])
            qi = r["base"]["qi"]
            queried.append(r["base"]["names"][r["g"][qi]])

    # Gold rates
    gold_first = sum(r["answer"] == f for r, f in zip(recs, firsts))
    gold_middle = sum(r["answer"] == m for r, m in zip(recs, middles))
    gold_last = sum(r["answer"] == l for r, l in zip(recs, lasts))
    gold_queried = sum(r["answer"] == q for r, q in zip(recs, queried))

    # Pred rates
    pred_first = sum(p == f for p, f in zip(preds, firsts))
    pred_middle = sum(p == m for p, m in zip(preds, middles))
    pred_last = sum(p == l for p, l in zip(preds, lasts))
    pred_queried = sum(p == q for p, q in zip(preds, queried))

    out["position_match"] = {
        "first_fact": {
            "pred": round(pred_first / n, 4),
            "gold": round(gold_first / n, 4),
        },
        "middle_fact": {
            "pred": round(pred_middle / n, 4),
            "gold": round(gold_middle / n, 4),
        },
        "last_fact": {"pred": round(pred_last / n, 4), "gold": round(gold_last / n, 4)},
        "queried": {
            "pred": round(pred_queried / n, 4),
            "gold": round(gold_queried / n, 4),
        },
    }

    # 2. Lexical selection per name
    pred_freq = collections.Counter(preds)
    gold_freq = collections.Counter(r["answer"] for r in recs)
    out["name_selection"] = {}
    for name, g in sorted(gold_freq.items()):
        out["name_selection"][name] = {
            "pred": round(pred_freq.get(name, 0) / n, 4),
            "gold": round(g / n, 4),
            "pred_count": pred_freq.get(name, 0),
            "gold_count": g,
        }
    out["max_abs_name_bias"] = round(
        max(abs(pred_freq.get(nm, 0) - g) / n for nm, g in gold_freq.items()), 4
    )

    # 3. Direction-confusion split
    if path == "G":
        split_counts = collections.Counter(classify(r, p) for r, p in zip(recs, preds))
        out["direction_split"] = dict(split_counts)
    else:
        correct_count = sum(p == r["answer"] for r, p in zip(recs, preds))
        wrong_count = n - correct_count
        out["direction_split"] = {"correct": correct_count, "wrong": wrong_count}

    # 4. qpos / qi stratification
    strat = collections.defaultdict(lambda: [0, 0])
    for r, p in zip(recs, preds):
        if path == "G":
            key = f"qpos={r.get('qpos', '?')}"
        else:
            key = f"qi={r['base']['qi']}"
        strat[key][0] += int(p == r["answer"])
        strat[key][1] += 1
    out["q_stratification"] = {}
    for k, (c, t) in sorted(strat.items()):
        out["q_stratification"][k] = {"acc": round(c / t, 4), "n": t, "correct": c}

    # Overall accuracy
    correct_count = sum(p == r["answer"] for r, p in zip(recs, preds))
    out["overall_acc"] = round(correct_count / n, 4)
    out["n_correct"] = correct_count

    # Per-base error anatomy: all-wrong bases
    bybase = collections.defaultdict(list)
    for r, p in zip(recs, preds):
        bybase[r["base_id"]].append((r, p))
    allwrong = []
    for b, items in sorted(bybase.items()):
        if all(p != r["answer"] for r, p in items):
            if path == "G":
                cats = sorted({classify(r, p) for r, p in items})
                qpos = items[0][0].get("qpos")
            else:
                cats = ["wrong"]
                qpos = None
            allwrong.append(
                {"base_id": b, "qpos": qpos, "cats": cats, "n_errors": len(items)}
            )
    out["allwrong_bases"] = allwrong
    out["n_allwrong_bases"] = len(allwrong)

    return out


def main():
    report = {}

    for (path, vocab), model_csvs in CELLS.items():
        # Load the frozen JSONL episodes for this cell
        jl_path = TASKS / f"gate_{path}_{vocab}.jsonl"
        if not jl_path.exists():
            print(f"WARNING: {jl_path} not found, skipping {path}/{vocab}")
            continue
        recs = load_jsonl(jl_path)
        print(f"Loaded {len(recs)} episodes from {jl_path}")

        for model_label, csv_path in model_csvs:
            if not csv_path.exists():
                print(
                    f"WARNING: {csv_path} not found, skipping {model_label} {path}/{vocab}"
                )
                continue

            # Read CSV predictions
            with open(csv_path) as f:
                csv_rows = list(csv.DictReader(f))
            preds = [row["pred"] for row in csv_rows]
            assert len(preds) == len(
                recs
            ), f"CSV length {len(preds)} != JSONL length {len(recs)} for {csv_path}"

            key = f"{model_label}/{path}/{vocab}"
            print(f"  Computing {key}...")
            result = audit_cell(recs, preds, path, vocab, model_label)
            report[key] = result

            # Print summary
            r = result
            pos = r["position_match"]
            dirs = r["direction_split"]
            qs = {k: f"acc={v['acc']:.4f}" for k, v in r["q_stratification"].items()}
            print(
                f"    {key}: acc={r['overall_acc']:.4f} "
                f"pos(first p/g)={pos['first_fact']['pred']}/{pos['first_fact']['gold']} "
                f"(mid)={pos['middle_fact']['pred']}/{pos['middle_fact']['gold']} "
                f"(last)={pos['last_fact']['pred']}/{pos['last_fact']['gold']} "
                f"(queried)={pos['queried']['pred']}/{pos['queried']['gold']} "
                f"maxnamebias={r['max_abs_name_bias']:.4f} "
                f"split={dirs} "
                f"qstrat={qs} "
                f"allwrong={r['n_allwrong_bases']}"
            )

    # Write JSON
    json_path = OUT_DIR / "d8_shortcut_audit.json"
    json.dump(report, open(json_path, "w"), indent=2)
    print(f"\nWritten JSON: {json_path}")

    # Write Markdown
    md_lines = []
    md_lines.append("# D8 Shortcut Audit — Phase 10 Track A")
    md_lines.append("")
    md_lines.append("Computed from stored bytes.")
    md_lines.append("")
    md_lines.append(
        "Models: Nemotron (Track A), Llama-3.3 (Track A2), Qwen2.5-72B (Phase 8A-final)"
    )
    md_lines.append("")
    md_lines.append("## Legend")
    md_lines.append("")
    md_lines.append(
        "- **Position match**: gold/predicted rate of answer/prediction matching the first/middle/last mentioned name or the queried (co-located) name"
    )
    md_lines.append(
        "- **Lexical selection**: per-name prediction frequency vs gold frequency"
    )
    md_lines.append(
        "- **Direction split** (G cells): correct / inner_clause / opposite_endpoint / other"
    )
    md_lines.append(
        "- **qpos stratification**: accuracy split by query position (qpos for G, qi for P)"
    )
    md_lines.append("")

    # Group by model
    for model in ["Nemotron", "Llama-3.3", "Qwen2.5-72B"]:
        md_lines.append(f"## {model}")
        md_lines.append("")

        for path in ["P", "G"]:
            for vocab in ["fit", "transfer"]:
                key = f"{model}/{path}/{vocab}"
                if key not in report:
                    continue
                r = report[key]
                md_lines.append(f"### {path}/{vocab}")
                md_lines.append("")
                md_lines.append(
                    f"- **Overall accuracy**: {r['overall_acc']:.4f} ({r['n_correct']}/{r['n']})"
                )
                md_lines.append("")

                pos = r["position_match"]
                md_lines.append("**Position-match rates**:")
                md_lines.append("")
                md_lines.append("| Position  | Pred rate | Gold rate |")
                md_lines.append("|-----------|-----------|-----------|")
                for pos_name in ["first_fact", "middle_fact", "last_fact", "queried"]:
                    d = pos[pos_name]
                    label = pos_name.replace("_", " ")
                    md_lines.append(f"| {label} | {d['pred']:.4f} | {d['gold']:.4f} |")
                md_lines.append("")

                md_lines.append("**Lexical selection per name**:")
                md_lines.append("")
                md_lines.append(
                    "| Name | Pred rate | Gold rate | Pred count | Gold count |"
                )
                md_lines.append(
                    "|------|-----------|-----------|------------|------------|"
                )
                for name, ns in r["name_selection"].items():
                    md_lines.append(
                        f"| {name} | {ns['pred']:.4f} | {ns['gold']:.4f} | {ns['pred_count']} | {ns['gold_count']} |"
                    )
                md_lines.append(
                    f"| *max abs bias* | {r['max_abs_name_bias']:.4f} | | | |"
                )
                md_lines.append("")

                md_lines.append("**Direction-confusion split**:")
                md_lines.append("")
                dirs = r["direction_split"]
                md_lines.append("| Category | Count | Rate |")
                md_lines.append("|----------|-------|------|")
                for cat in sorted(dirs.keys()):
                    md_lines.append(f"| {cat} | {dirs[cat]} | {dirs[cat]/r['n']:.4f} |")
                md_lines.append("")

                md_lines.append("**qpos stratification**:")
                md_lines.append("")
                md_lines.append("| Stratum | Accuracy | n (correct/total) |")
                md_lines.append("|---------|----------|-------------------|")
                for k, v in r["q_stratification"].items():
                    md_lines.append(
                        f"| {k} | {v['acc']:.4f} | {v['correct']}/{v['n']} |"
                    )
                md_lines.append("")

                if r["allwrong_bases"]:
                    md_lines.append(f"**All-wrong bases** ({r['n_allwrong_bases']}):")
                    md_lines.append("")
                    for aw in r["allwrong_bases"]:
                        md_lines.append(
                            f"- base_id={aw['base_id']} qpos={aw['qpos']} cats={aw['cats']} n_errors={aw['n_errors']}"
                        )
                    md_lines.append("")

    md_path = OUT_DIR / "d8_shortcut_audit.md"
    md_path.write_text("\n".join(md_lines))
    print(f"Written MD: {md_path}")


if __name__ == "__main__":
    main()

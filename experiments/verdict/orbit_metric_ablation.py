"""Orbit-metric ablation table for the paper's methods.

Descriptive only, from stored gate results. Shows what each behavioural metric
reveals or hides: episode accuracy (per-question), strict-orbit (all 6 orbit
members correct), orbit consistency (answers permute lawfully with g). The
point: episode accuracy systematically OVERSTATES role competence relative to
the orbit-level metrics; the gap is the role-consistency signal, and a gate on
episode accuracy alone would pass models the orbit metrics fail.

Reads the stored gate_results.json across all families/phases; writes a single
table to results/phase10/E7_orbit_metric_ablation.{json,md}. No compute.
"""

import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[2]
SOURCES = [
    ("Qwen2.5-72B (8A frozen)", "results/verdict/answer_position/gate_results.json"),
    ("Qwen2.5-72B (9 joint-perm)", "results/binding_sites/gate_local/gate_results.json"),
    ("Nemotron-70B (10A)", "results/cross_family/nemotron/gate/gate_results.json"),
    ("Llama-3.3-70B (10A)", "results/cross_family/llama/gate/gate_results.json"),
]
CELLS = ["P/fit", "P/transfer", "G/fit", "G/transfer"]
KEYS = ("episode_acc", "strict_orbit", "orbit_consistency")


def get_cell(d, cell):
    c = d.get("cells", {}).get(cell)
    if not c:
        return None
    return {k: c.get(k) for k in KEYS}


def main():
    rows = []
    for label, rel in SOURCES:
        p = ROOT / rel
        if not p.exists():
            rows.append(dict(model=label, error=f"missing {rel}"))
            continue
        d = json.load(open(p))
        for cell in CELLS:
            c = get_cell(d, cell)
            if not c or c["episode_acc"] is None:
                continue
            ep, st, co = c["episode_acc"], c["strict_orbit"], c["orbit_consistency"]
            rows.append(
                dict(
                    model=label,
                    cell=cell,
                    episode_acc=ep,
                    strict_orbit=st,
                    orbit_consistency=co,
                    ep_minus_strict=round(ep - st, 4),
                    hidden_by_episode_acc=bool(ep >= 0.95 and st < 0.90),
                )
            )

    scored = [r for r in rows if "episode_acc" in r]
    # Annotate the known scorer artifact: Nemotron's raw P-cell gap is the
    # markdown-emphasis (`**Name**`) artifact corrected by D10 (P/fit
    # 0.952->1.000, P/transfer 0.908->1.000), NOT genuine role inconsistency.
    for r in scored:
        r["caveat"] = (
            "Nemotron P-cell gap is the pre-D10 markdown-asterisk "
            "scorer artifact (D10: P/fit->1.000, P/transfer->1.000), "
            "not genuine orbit inconsistency"
            if r["model"].startswith("Nemotron") and r["cell"].startswith("P/")
            else None
        )
    # genuine gaps = exclude the annotated scorer artifact
    genuine = [r for r in scored if r["caveat"] is None]
    gaps = [r["ep_minus_strict"] for r in genuine]
    hidden = [r for r in genuine if r["hidden_by_episode_acc"]]
    summary = dict(
        n_cells=len(scored),
        max_ep_minus_strict=max(gaps),
        mean_ep_minus_strict=round(sum(gaps) / len(gaps), 4),
        n_cells_episode_pass_but_strict_fail=len(hidden),
        cells_hidden=[f"{r['model']} {r['cell']}" for r in hidden],
        max_genuine_gap_cell=max(genuine, key=lambda r: r["ep_minus_strict"]).get(
            "model"
        )
        + " "
        + max(genuine, key=lambda r: r["ep_minus_strict"]).get("cell"),
        reading=(
            "Excluding the Nemotron pre-D10 markdown artifact, episode "
            f"accuracy overstates orbit competence by up to {max(gaps):.3f} "
            "on genuine cells. The sharpest genuine case is Qwen2.5-72B "
            "under joint permutation (G-cells): episode accuracy 0.984/0.988 "
            "reads as near-ceiling, while strict-orbit 0.933/0.940 exposes "
            "the role-consistency failure that carried the Phase 9 "
            "H_retrieval_everywhere finding. A gate on episode accuracy alone "
            "would have passed it. This is why the gate is conjunctive over "
            "all three metrics."
        ),
    )
    out = dict(sources=[s[1] for s in SOURCES], rows=scored, summary=summary)
    json.dump(
        out, open(ROOT / "results/verdict/E7_orbit_metric_ablation.json", "w"), indent=1
    )

    lines = [
        "# E7 — Orbit-metric ablation (descriptive; stored gate results)",
        "",
        "| model | cell | episode | strict-orbit | consistency | ep−strict | hidden? |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in scored:
        lines.append(
            f"| {r['model']} | {r['cell']} | {r['episode_acc']:.3f} | "
            f"{r['strict_orbit']:.3f} | {r['orbit_consistency']:.3f} | "
            f"{r['ep_minus_strict']:+.3f} | "
            f"{'**yes**' if r['hidden_by_episode_acc'] else '—'} |"
        )
    lines += [
        "",
        f"**Max episode−strict gap:** {summary['max_ep_minus_strict']:.3f}. "
        f"**Cells passing episode-acc 0.95 but failing strict-orbit 0.90:** "
        f"{summary['n_cells_episode_pass_but_strict_fail']} "
        f"({', '.join(summary['cells_hidden']) or 'none'}).",
        "",
        summary["reading"],
    ]
    (ROOT / "results/verdict/E7_orbit_metric_ablation.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()

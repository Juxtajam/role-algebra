"""Track B — fidelity check for the extracted organism_data module.

Rebuilds the finetune's own final query-breakdown eval sets with
phase10/trackB/organism_data.py, runs the stored final checkpoint, and
compares episode/strict/consistency to the stored query_breakdown.json.
Exact float equality required (same episodes, deterministic model) — any
mismatch means the extraction is NOT faithful and nothing downstream runs.
"""

import json
import pathlib
import sys

import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
import trained_organism_data as od  # noqa: E402
from trained.model import TinyTransformer, masked_answer_preds  # noqa: E402

QUERY_TYPES = [
    ("A_PS", od.A_PS),
    ("A_SN", od.A_SN),
    ("A_NS", od.A_NS),
    ("A_SG", od.A_SG),
    ("Q_P", od.Q_P),
    ("Q_G", od.Q_G),
    ("Q_3H", od.Q_3H),
]


def check(seed, arm):
    p = ROOT / f"results/trained_organisms/large/finetune/seed{seed}/{arm}"
    stored = json.load(open(p / "query_breakdown.json"))
    ck = torch.load(p / "ckpt.pt", map_location="cpu", weights_only=False)
    model = TinyTransformer(seed=seed, n_layers=8)
    model.load_state_dict(ck["model"])
    model.eval()

    n_mismatch = 0
    for label, qt in QUERY_TYPES:
        ev = od.build_eval_orbits(arm, "fit", 96, seed=9900 + 31 * qt + seed, qtok=qt)
        with torch.no_grad():
            preds = masked_answer_preds(
                model, ev["tokens"], ev["answer_pos"], ev["candidates"]
            )
        m = od.orbit_metrics(preds, ev)
        s = stored[label]
        ok = (
            m["episode_acc"] == s["accuracy"]
            and m["strict_orbit_acc"] == s["strict_orbit"]
            and m["orbit_consistency"] == s["orbit_consistency"]
        )
        if not ok:
            n_mismatch += 1
            print(
                f"  MISMATCH {label}: local acc={m['episode_acc']:.6f} "
                f"stored={s['accuracy']:.6f} | strict {m['strict_orbit_acc']:.6f}"
                f" vs {s['strict_orbit']:.6f}"
            )
    status = "EXACT" if n_mismatch == 0 else f"{n_mismatch} MISMATCHES"
    print(f"seed{seed}/{arm}: {status} across {len(QUERY_TYPES)} query types")
    return n_mismatch == 0


if __name__ == "__main__":
    runs = [(s, a) for s in [7, 8, 9, 10, 11, 13] for a in "RC"]
    if len(sys.argv) > 1:
        runs = [(int(sys.argv[1]), sys.argv[2])]
    all_ok = all([check(s, a) for s, a in runs])
    print("FIDELITY:", "PASS — extraction is faithful" if all_ok else "FAIL")
    sys.exit(0 if all_ok else 1)

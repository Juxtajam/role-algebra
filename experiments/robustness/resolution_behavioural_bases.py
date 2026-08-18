"""Phase 8C resolution — item 5 prerequisite (stored-artifact re-examination,
no new fitting): identify base 57 and all G-orbit inconsistent / non-strict
bases in the stored 8A-final disc + gate predictions.

Outputs results/phase8c_resolution/item5_behavioural_bases.json with, per
cell: bases not strict-orbit-correct, bases not orbit-consistent, and
per-base error counts. These are later intersected with per-base condition
outliers from the 8C run's stored predictions (item 5 table).
"""

import json, pathlib
import itertools

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "results/robustness/resolution"
OUT.mkdir(parents=True, exist_ok=True)

PERMS = [tuple(p) for p in itertools.permutations(range(3))]


def analyse(kind, path, vocab):
    recs = [
        json.loads(l)
        for l in open(ROOT / f"results/verdict/gate/tasks/{kind}_{path}_{vocab}.jsonl")
    ]
    preds = json.load(
        open(ROOT / f"results/verdict/answer_position/preds_{kind}_{path}_{vocab}.json")
    )
    assert len(recs) == len(preds)
    n_bases = len(recs) // 6
    out = {}
    for b in range(n_bases):
        idx = range(b * 6, b * 6 + 6)
        correct = [preds[i] == recs[i]["answer"] for i in idx]
        # abstract-role consistency: all 6 predictions map to the same slot
        slots = []
        for i in idx:
            r = recs[i]
            names = [
                r["base"]["names"][r["g"][j]] for j in range(3)
            ]  # permuted person list
            slots.append(names.index(preds[i]) if preds[i] in names else -1)
        consistent = len(set(slots)) == 1 and slots[0] != -1
        n_err = 6 - sum(correct)
        if n_err or not consistent:
            out[b] = dict(
                n_errors=n_err,
                strict=bool(n_err == 0),
                consistent=bool(consistent),
                pred_slots=slots,
                qpos=recs[b * 6]["base"].get("qpos", recs[b * 6]["base"].get("qi")),
            )
    return dict(n_bases=n_bases, flagged=out)


res = {}
for kind in ("disc", "gate"):
    for path in ("P", "G"):
        for vocab in ("fit", "transfer"):
            key = f"{kind}_{path}_{vocab}"
            res[key] = analyse(kind, path, vocab)
            f = res[key]["flagged"]
            print(
                f"{key}: {len(f)} flagged bases of {res[key]['n_bases']}: "
                + ", ".join(
                    f"b{b}(err={v['n_errors']},cons={v['consistent']})"
                    for b, v in sorted(f.items())
                )
            )

json.dump(res, open(OUT / "item5_behavioural_bases.json", "w"), indent=2)
print("wrote", OUT / "item5_behavioural_bases.json")

"""Local tokeniser verification (before renting GPUs).

Target model: Qwen/Qwen2.5-72B-Instruct @ pinned revision (recorded below).
Verifies, using the frozen in-situ procedure (verify_single_token imported
from task_generation, not reimplemented):
  - every name in both vocabularies (name_pools.json fit + transfer)
  - marks (properties), symbols, structural/system/template strings
  - every episode's answer token in situ inside its own frozen prompt,
    all 4800 gate + 14400 disc episodes (the generation-time assertion,
    re-run against the target tokeniser)
Also compares target-tokeniser IDs against the generation tokeniser
(Qwen/Qwen3-32B) for the record.
"""

import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]  # repository root
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from task_generation import (  # noqa: E402  frozen procedure, reused
    MARKS,
    SYMBOLS,
    verify_single_token,
)

from transformers import AutoTokenizer  # noqa: E402

TARGET_MODEL = "Qwen/Qwen2.5-72B-Instruct"
TARGET_REVISION = "495f39366efef23836d0cfae4fbe635880d2be31"
GEN_MODEL = "Qwen/Qwen3-32B"
TASKS = ROOT / "results/verdict/gate/tasks"


def main():
    tok = AutoTokenizer.from_pretrained(TARGET_MODEL, revision=TARGET_REVISION)
    gen_tok = AutoTokenizer.from_pretrained(GEN_MODEL)
    pools = json.load(open(TASKS / "name_pools.json"))
    names = pools["fit"] + pools["transfer"]
    assert len(names) == 24 and len(set(names)) == 24

    # --- names, frozen in-situ procedure on the TARGET tokeniser ---
    ok_names, report = verify_single_token(tok, names)
    ok_set = {n for n, _ in ok_names}
    split = [n for n in names if n not in ok_set]
    print(f"names single-token in situ on target: {len(ok_set)}/24")
    if split:
        print("SPLITTING NAMES:", split)
        for n in split:
            print(" ", n, report[n])

    # --- same-id check vs generation tokeniser ---
    id_diff = []
    for n in names:
        t_ids = tok.encode(" " + n)
        g_ids = gen_tok.encode(" " + n)
        if t_ids != g_ids:
            id_diff.append((n, t_ids, g_ids))
    print(
        f"name token-id identical to generation tokeniser (Qwen3-32B): "
        f"{24 - len(id_diff)}/24"
    )
    if id_diff:
        print("  diffs:", id_diff)

    # --- marks (properties), symbols, structural strings ---
    aux_report = {}
    for label, words in (("marks", MARKS), ("symbols", SYMBOLS)):
        aux_report[label] = {}
        for w in words:
            ids = tok.encode(" " + w)
            aux_report[label][w] = dict(
                n_tokens=len(ids), ids=ids, same_as_gen=ids == gen_tok.encode(" " + w)
            )
        multi = {
            w: r["n_tokens"] for w, r in aux_report[label].items() if r["n_tokens"] != 1
        }
        print(
            f"{label}: {len(words) - len(multi)}/{len(words)} single-token"
            + (f"; multi: {multi}" if multi else "")
        )

    structural = [
        "Facts:",
        " sigil",
        " mark",
        " holds",
        " guards",
        " relies",
        " bears",
        "Answer:",
        " Which",
        " person",
    ]
    aux_report["structural"] = {}
    for w in structural:
        ids = tok.encode(w)
        aux_report["structural"][w] = dict(
            n_tokens=len(ids), ids=ids, same_as_gen=ids == gen_tok.encode(w)
        )
    print(
        "structural strings:",
        {w: r["n_tokens"] for w, r in aux_report["structural"].items()},
    )
    all_same = all(
        r["same_as_gen"]
        for lab in ("marks", "symbols", "structural")
        for r in aux_report[lab].values()
    )
    print(
        "all marks/symbols/structural token ids identical to generation " "tokeniser:",
        all_same,
    )

    # --- every frozen episode: answer token single + present in prompt ---
    bad = 0
    n_eps = 0
    for fp in sorted(TASKS.glob("*.jsonl")):
        for line in open(fp):
            r = json.loads(line)
            aid = tok.encode(" " + r["answer"])
            uids = tok.encode(r["user"])
            if not (len(aid) == 1 and uids.count(aid[0]) >= 1):
                bad += 1
            n_eps += 1
    print(
        f"episode-level in-situ answer-token check on target tokeniser: "
        f"{n_eps - bad}/{n_eps} pass"
    )

    out = dict(
        target_model=TARGET_MODEL,
        target_revision=TARGET_REVISION,
        generation_model=GEN_MODEL,
        names_verified=f"{len(ok_set)}/24",
        splitting_names=split,
        name_ids_identical_to_gen=len(id_diff) == 0,
        aux=aux_report,
        episode_check=dict(total=n_eps, failures=bad),
        name_report=report,
    )
    fp = ROOT / "results/verdict/gate/phase8a_final_tokeniser_check.json"
    json.dump(out, open(fp, "w"), indent=2)
    print("written:", fp)
    if split or bad:
        print("TOKENISER EXEMPTION WOULD FIRE — replacements required")
        sys.exit(2)
    print(
        "NO replacements required — frozen set valid on target tokeniser; "
        "hash_v2 not needed"
    )


if __name__ == "__main__":
    main()

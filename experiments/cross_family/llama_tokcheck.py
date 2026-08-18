"""Phase 10 Track A — Step 1: LOCAL tokeniser verification (before renting).

Target model (approved fallback; Llama-3.3 token lacks scopes):
  meta-llama/Llama-3.3-70B-Instruct
  revision 6f6073b423013f6a7d4d9f39144961bfbfbc386b, bf16, ungated,
  Llama-3.1 family (non-Qwen — the Track A cross-family requirement).

Procedure = 8A-final tokcheck (phase8a_final_tokcheck.py), verbatim where
possible: the FROZEN in-situ procedure verify_single_token is IMPORTED from
phase8a_generate_tasks.py, not reimplemented. Verifies:
  - every name in both frozen vocabularies (name_pools.json fit+transfer;
    local copy verified byte-identical to dv3-results:phase8a/tasks/ —
    sha256 5797fb7fd675db1177e17840c1fdd59a329b0c45253d8287f623b7ef11e4b263)
  - marks (properties), symbols, structural/system/template strings
  - every frozen episode's answer token in situ inside its own prompt
    (4800 gate + 14400 disc episodes), on the target tokeniser
Also compares target-tokeniser IDs against the generation tokeniser
(Qwen/Qwen3-32B) for the record (expected to differ everywhere: different
BPE family).

DECLARED ADAPTATION (D-numbered in the deviation record): the Llama-3.1
tokeniser prepends BOS (<|begin_of_text|>, id 128000) on encode() by
default; the frozen procedure was written for Qwen (no BOS). The target
tokeniser is wrapped so that encode() == encode(add_special_tokens=False).
This preserves the frozen procedure's semantics exactly (a name is one
token in situ; BOS is a constant prefix independent of the name) and
changes no threshold, no context string, no counting rule.

REPLACEMENT-ONLY-ON-SPLIT EXEMPTION (8A-final rule, unchanged): if any
frozen name splits on the target tokeniser, ONLY the splitting names are
replaced, drawn from the frozen NONCE_CANDIDATES list via the frozen
verification procedure and frequency-matched on THIS tokeniser (BPE
merge-rank = token id, the frozen proxy): each splitting name is replaced
by the unused verified candidate whose target token id is nearest the
median target id of the surviving names in the SAME pool (ties -> lower
id; candidates in frozen list order). Replacements re-verified with the
frozen procedure; every replacement recorded with a diff. Affected
episodes are then regenerated deterministically by trackA_regen_tasks.py
(same frozen generation code + seeds, pool order preserved) and hash_v2
computed. The frozen hash 84f2e54d... governs unchanged episodes.

Outputs: phase10/trackA2/tokeniser_check.json (+ exit 2 if exemption fires).
"""

import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]  # ~/modal_dv3
sys.path.insert(0, str(ROOT))
from task_generation import (  # noqa: E402  frozen procedure, reused
    MARKS,
    NONCE_CANDIDATES,
    SYMBOLS,
    verify_single_token,
)

from transformers import AutoTokenizer  # noqa: E402

TARGET_MODEL = "meta-llama/Llama-3.3-70B-Instruct"
TARGET_REVISION = "6f6073b423013f6a7d4d9f39144961bfbfbc386b"
GEN_MODEL = "Qwen/Qwen3-32B"
TASKS = ROOT / "results/verdict/gate/tasks"
TOKDIR = ROOT / "phase10/trackA2/tokenizer_files"  # rev-pinned direct fetch
OUT = ROOT / "phase10/trackA2/tokeniser_check.json"


class NoSpecialEncode:
    """BOS-stripping shim (declared adaptation, see module docstring)."""

    def __init__(self, tok):
        self._tok = tok

    def encode(self, s):
        return self._tok.encode(s, add_special_tokens=False)


def main():
    raw = AutoTokenizer.from_pretrained(str(TOKDIR))
    assert raw.encode("", add_special_tokens=True) == [
        128000
    ], "expected Llama BOS-prepending behaviour"
    tok = NoSpecialEncode(raw)
    gen_tok = NoSpecialEncode(AutoTokenizer.from_pretrained(GEN_MODEL))
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

    # --- same-id check vs generation tokeniser (for the record) ---
    id_diff = []
    for n in names:
        t_ids = tok.encode(" " + n)
        g_ids = gen_tok.encode(" " + n)
        if t_ids != g_ids:
            id_diff.append((n, t_ids, g_ids))
    print(
        f"name token-id identical to generation tokeniser (Qwen3-32B): "
        f"{24 - len(id_diff)}/24 (cross-family: identity NOT required)"
    )

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

    # --- every frozen episode: answer token single + present in prompt ---
    bad = 0
    n_eps = 0
    bad_by_file = {}
    for fp in sorted(TASKS.glob("*.jsonl")):
        nb = 0
        for line in open(fp):
            r = json.loads(line)
            aid = tok.encode(" " + r["answer"])
            uids = tok.encode(r["user"])
            if not (len(aid) == 1 and uids.count(aid[0]) >= 1):
                nb += 1
            n_eps += 1
        bad += nb
        bad_by_file[fp.name] = nb
    print(
        f"episode-level in-situ answer-token check on target tokeniser: "
        f"{n_eps - bad}/{n_eps} pass"
    )

    # --- replacement pool preview (candidate survival on target) ---
    ok_cand, cand_report = verify_single_token(tok, NONCE_CANDIDATES)
    print(
        f"frozen NONCE_CANDIDATES verified on target: "
        f"{len(ok_cand)}/{len(NONCE_CANDIDATES)}"
    )

    out = dict(
        track="10A",
        target_model=TARGET_MODEL,
        target_revision=TARGET_REVISION,
        target_tokenizer_files_sha256=dict(
            (p.name, __import__("hashlib").sha256(p.read_bytes()).hexdigest())
            for p in sorted(TOKDIR.glob("*.json"))
        ),
        generation_model=GEN_MODEL,
        bos_adaptation=(
            "encode(add_special_tokens=False) shim; Llama "
            "prepends BOS id 128000 by default; declared "
            "deviation, see the deviation record"
        ),
        name_pools_sha256=__import__("hashlib")
        .sha256((TASKS / "name_pools.json").read_bytes())
        .hexdigest(),
        names_verified=f"{len(ok_set)}/24",
        splitting_names=split,
        name_ids_identical_to_gen=len(id_diff) == 0,
        name_id_diffs=id_diff,
        aux=aux_report,
        episode_check=dict(total=n_eps, failures=bad, failures_by_file=bad_by_file),
        candidate_pool_on_target=dict(
            verified=f"{len(ok_cand)}/{len(NONCE_CANDIDATES)}",
            ok=[list(t) for t in ok_cand],
            report=cand_report,
        ),
        name_report=report,
    )
    json.dump(out, open(OUT, "w"), indent=2)
    print("written:", OUT)
    if split or bad:
        print(
            "TOKENISER EXEMPTION FIRES — replacement-only-on-split; "
            "run trackA_regen_tasks.py next; hash_v2 required"
        )
        sys.exit(2)
    print(
        "NO replacements required — frozen set valid on target tokeniser; "
        "hash_v2 not needed"
    )


if __name__ == "__main__":
    main()

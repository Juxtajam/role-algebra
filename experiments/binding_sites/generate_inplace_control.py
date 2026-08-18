"""Phase 9 Item 2 (local prep) — base-draw control: IN-PLACE orbits over the
SAME new G base problems used in the joint-permutation set. instruction (spec, Item 2): "the base-draw control. Evaluate in-place
orbits over the same new G base problems used in the joint set (both
vocabularies), identical thresholds and bootstrap protocol. One cell pair.
Write phase9/gate/inplace_control.json. This determines whether the
joint-set consistency drop is permutation-driven or base-sampling."

Construction
============
Base problems: EXACTLY the joint set's G bases. phase9_generate_joint.py
draws bases with rng = np.random.default_rng((20260808, path == "G",
vocab == "transfer")) and 150 sequential gen.make_base(rng, path, pool)
calls per cell with no other rng consumption; this script repeats that
sequence bit-for-bit AND asserts, per base_id, that the regenerated base
dict equals the base dict embedded in the joint episode records
(results/phase9/tasks_joint/joint_G_{fit,transfer}.jsonl). Same facts,
same names, same fact_order per base — only the orbit construction differs.

Orbit construction: the FROZEN 8A in-place action, gen.episode /
gen.facts_for imported from phase8a_generate_tasks.py (not reimplemented):
names permute (role i held by names[g[i]]), fact display order FIXED by
base['fact_order'] across the whole orbit.

Per-orbit invariants asserted below against the joint set:
  1. at g = identity the in-place episode is byte-identical to the joint
     episode of the same (base_id, g) — shared base, shared identity member;
  2. for every g the fact-sentence MULTISET equals the joint episode's
     (same content, different display arrangement) and the answer is equal.

Cells: 2 (G/fit, G/transfer) x 150 bases x 6 perms = 1800 episodes.
Vocabularies: the frozen pools (results/phase8a/tasks/name_pools.json),
same bytes the joint generation read.

Tokeniser verification: target tokeniser Qwen/Qwen2.5-72B-Instruct
@ 495f39366efef23836d0cfae4fbe635880d2be31, frozen in-situ procedure
verify_single_token (24/24 required) + per-episode assertion (answer
single-token, present in prompt) on the target tokeniser.

Outputs (results/phase9/tasks_inplace_control/):
  inplace_G_fit.jsonl, inplace_G_transfer.jsonl   900 episodes each
  manifest.json                                   counts, seeds, sha16,
                                                  base-identity check
  README.md                                       this docstring
"""

import hashlib
import json
import pathlib
import sys

import numpy as np
from transformers import AutoTokenizer

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import task_generation as gen  # noqa: E402  frozen procedure, reused

OUT = ROOT / "results/binding_sites/tasks_inplace_control"
JOINT = ROOT / "results/binding_sites/tasks_joint"
TARGET_MODEL = "Qwen/Qwen2.5-72B-Instruct"
TARGET_REVISION = "495f39366efef23836d0cfae4fbe635880d2be31"
MASTER_SEED_JOINT = 20260808  # SAME seed as the joint set: same bases
K = gen.K  # 3
PERMS = gen.PERMS  # all 6 permutations of range(3)
IDENTITY = tuple(range(K))
N_BASES = 150
CELLS = [("G", "fit"), ("G", "transfer")]  # one cell pair, per instruction


def _fact_multiset(user_text):
    line = user_text.split("\n")[0]
    assert line.startswith("Facts: ")
    line = line[len("Facts: ") :]
    return sorted(s.rstrip(".") for s in line.split(". "))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(TARGET_MODEL, revision=TARGET_REVISION)

    # --- vocabularies: the FROZEN pools, byte-identical source ---
    pools_fp = ROOT / "results/verdict/gate/tasks/name_pools.json"
    pools = json.load(open(pools_fp))
    fit, transfer = pools["fit"], pools["transfer"]
    assert len(fit) == 12 and len(transfer) == 12
    assert not set(fit) & set(transfer)

    # --- tokeniser verification: frozen procedure, target tokeniser ---
    names24 = fit + transfer
    ok_names, _report = gen.verify_single_token(tok, names24)
    ok_set = {n for n, _ in ok_names}
    n_ok = len(ok_set & set(names24))
    print(f"names single-token in situ on target tokeniser: {n_ok}/24")
    assert n_ok == 24, sorted(set(names24) - ok_set)

    manifest = dict(
        phase="9-item2-prep",
        set="inplace-base-draw-control",
        generation_module="phase8a_generate_tasks.py (frozen; imported)",
        rule=(
            "FROZEN 8A in-place action (gen.episode/gen.facts_for): names "
            "permute in place, fact display order fixed by "
            "base['fact_order'] across the orbit. Bases are the SAME new "
            "G base problems as the joint set (same master seed 20260808, "
            "same rng stream, asserted equal per base_id against the base "
            "dicts embedded in joint_G_{fit,transfer}.jsonl)."
        ),
        base_source=(
            "results/binding_sites/tasks_joint/joint_G_{fit,transfer}"
            ".jsonl (regenerated from the same rng stream and "
            "asserted equal per base_id)"
        ),
        target_tokeniser=dict(model=TARGET_MODEL, revision=TARGET_REVISION),
        master_seed=MASTER_SEED_JOINT,
        k=K,
        perms=[list(p) for p in PERMS],
        n_bases_per_cell=N_BASES,
        vocab_source="results/verdict/gate/tasks/name_pools.json (frozen, reused)",
        system=gen.SYSTEM,
        templates=dict(
            bears=gen.FACT_BEARS,
            holds=gen.FACT_HOLDS,
            guards=gen.FACT_GUARDS,
            shelters=gen.FACT_SHELTERS,
            stage_p=gen.STAGE_P,
            query_p=gen.QUERY_P,
            stage_g=gen.STAGE_G,
            query_g=gen.QUERY_G,
        ),
        base_identity_check={},
        cells={},
    )

    ep_check_total, ep_check_bad = 0, 0
    for path, vocab in CELLS:
        # SAME rng construction + call sequence as phase9_generate_joint.py
        rng = np.random.default_rng(
            (MASTER_SEED_JOINT, path == "G", vocab == "transfer")
        )
        pool = fit if vocab == "fit" else transfer

        # joint records for this cell, keyed (base_id, tuple(g))
        joint_recs = [
            json.loads(l) for l in open(JOINT / f"joint_{path}_{vocab}.jsonl")
        ]
        assert len(joint_recs) == N_BASES * len(PERMS)
        joint_by = {(r["base_id"], tuple(r["g"])): r for r in joint_recs}
        joint_base = {r["base_id"]: r["base"] for r in joint_recs}

        fp = OUT / f"inplace_{path}_{vocab}.jsonl"
        n, n_base_match = 0, 0
        with open(fp, "w") as fh:
            for b in range(N_BASES):
                base = gen.make_base(rng, path, pool)
                # BASE IDENTITY with the joint set: same facts, same names,
                # same fact_order — the full base dict must be equal.
                assert base == joint_base[b], (path, vocab, b)
                n_base_match += 1
                for g in PERMS:
                    ep = gen.episode(base, g)  # frozen in-place
                    jr = joint_by[(b, g)]
                    # invariant 1: identity member byte-identical to joint
                    if g == IDENTITY:
                        assert ep["user"] == jr["user"], (path, vocab, b)
                        assert ep["answer"] == jr["answer"], (path, vocab, b)
                    # invariant 2: same fact multiset + same answer, every g
                    assert _fact_multiset(ep["user"]) == _fact_multiset(jr["user"]), (
                        path,
                        vocab,
                        b,
                        g,
                    )
                    assert ep["answer"] == jr["answer"], (path, vocab, b, g)
                    # generation-time in-situ assertion, TARGET tokeniser
                    aid = tok.encode(" " + ep["answer"])
                    assert len(aid) == 1, (ep["answer"], aid)
                    uids = tok.encode(ep["user"])
                    ep_check_total += 1
                    if uids.count(aid[0]) < 1:
                        ep_check_bad += 1
                    rec = dict(
                        kind="inplace_control",
                        path=path,
                        vocab=vocab,
                        base_id=b,
                        **ep,
                        base=base,
                    )
                    fh.write(json.dumps(rec) + "\n")
                    n += 1
        h16 = hashlib.sha256(fp.read_bytes()).hexdigest()[:16]
        manifest["cells"][f"inplace_control/{path}/{vocab}"] = dict(
            episodes=n, bases=N_BASES, sha256_16=h16
        )
        manifest["base_identity_check"][f"{path}/{vocab}"] = dict(
            bases_compared=N_BASES,
            bases_equal=n_base_match,
            joint_file=f"tasks_joint/joint_{path}_{vocab}.jsonl",
            identity_member_byte_identical=True,
            fact_multiset_equal_all_g=True,
            answer_equal_all_g=True,
        )
        print(
            f"inplace_control/{path}/{vocab}: {n} episodes  sha16={h16}  "
            f"base_identity {n_base_match}/{N_BASES}"
        )

    assert ep_check_bad == 0, ep_check_bad
    manifest["episode_token_check"] = dict(total=ep_check_total, failures=ep_check_bad)
    json.dump(manifest, open(OUT / "manifest.json", "w"), indent=2)
    print(
        f"episode-level in-situ answer-token check: "
        f"{ep_check_total - ep_check_bad}/{ep_check_total} pass"
    )

    readme = pathlib.Path(__file__).read_text().split('"""')[1]
    (OUT / "README.md").write_text(
        "# Phase 9 base-draw control: in-place orbits over the joint-set "
        "G bases\n\n" + readme + "\n"
    )

    # sample: one in-place G prompt vs its joint counterpart, same (base, g)
    recs = [json.loads(l) for l in open(OUT / "inplace_G_fit.jsonl")]
    r = recs[1]  # first non-identity member
    print(
        f"\n--- sample in-place control prompt (g={r['g']}) ---\n"
        f"{r['user']}\nanswer={r['answer']}"
    )


if __name__ == "__main__":
    main()

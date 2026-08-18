# Phase 9 base-draw control: in-place orbits over the joint-set G bases

Phase 9 Item 2 (local prep) — base-draw control: IN-PLACE orbits over the
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


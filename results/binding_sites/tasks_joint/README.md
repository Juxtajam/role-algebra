# Phase 9 joint-permutation episode set

Phase 9 STAGE 1 (local prep) — Task 2: generate the JOINT-PERMUTATION
episode set.

Same generation procedure and vocabularies as the frozen 84f2e54d... set:
templates, SYSTEM string, make_base sampling, K=3, full six-member orbits are
IMPORTED from the frozen phase8a_generate_tasks.py (not reimplemented).
Differences, exactly as authorised:
  - n_bases = 150 per (path x vocabulary) cell (>= 150 required), 4 cells,
    900 episodes per cell (150 bases x 6 permutations), 3600 total.
  - the group element g permutes NAMES AND FACT ORDER TOGETHER.

JOINT-PERMUTATION RULE (the single change vs the frozen in-place action)
========================================================================
Frozen in-place action (phase8a_generate_tasks.facts_for): under g, the
holds clause for abstract role i reads "names[g[i]] holds syms[i]" and every
fact clause stays at the display position fixed by base['fact_order'] for
the whole orbit. Positions never move; only names move between clauses.
Consequence: display position is a fixed proxy for abstract role — a
position-indexed binding is indistinguishable from a role binding.

Joint action (THIS set): the SAME permutation g is applied to the fact-clause
order as to the name assignment. Concretely:
  - name assignment (unchanged): role i is held by names[g[i]];
  - clause order: the display slot that under the identity permutation showed
    role j's holds clause now shows role inv_g[j]'s holds clause, where
    inv_g = g^{-1}. Equivalently, role i's holds clause moves TO the identity
    display slot of role g[i] — i.e. to the home slot of its new name
    names[g[i]]: abstract role i moves position with its name.
  - non-holds clauses (bears / guards / relies-on) carry no names; they are
    not acted on by g and keep their identity display slots. Only the K
    name-carrying (CARRY/holds) clauses are permuted, because g acts on
    names and the holds clauses are where the name<->role binding lives.
Resulting invariant (asserted per orbit below): the NAME occupying each
holds display slot is constant across all six orbit members
("names[g[inv_g[j]]] = names[j]"), while the SYMBOL in that slot moves with
g. A binding indexed by clause position therefore cannot be equivariant on
this set: the position->name map never changes while the correct answer
does. The multiset of fact sentences per episode is identical to the
in-place action's under the same (base, g) — asserted below — and at
g = identity the joint episode is byte-identical to the in-place episode.

Tokeniser verification: target tokeniser Qwen/Qwen2.5-72B-Instruct
@ 495f39366efef23836d0cfae4fbe635880d2be31, using the FROZEN in-situ
procedure verify_single_token imported from phase8a_generate_tasks.py
(24/24 names required), plus the generation-time per-episode assertion
(answer single-token, present in prompt) on the target tokeniser.

Outputs (results/phase9/tasks_joint/):
  joint_{P,G}_{fit,transfer}.jsonl   episode files (900 each)
  manifest.json                      counts, seeds, rule, per-file sha16
  README.md                          the joint rule, human-readable
../joint_tokeniser_check.json      24/24 name report + episode check


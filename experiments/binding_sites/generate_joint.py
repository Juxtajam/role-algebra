"""Phase 9 STAGE 1 (local prep) — Task 2: generate the JOINT-PERMUTATION
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

OUT = ROOT / "results/binding_sites/tasks_joint"
TARGET_MODEL = "Qwen/Qwen2.5-72B-Instruct"
TARGET_REVISION = "495f39366efef23836d0cfae4fbe635880d2be31"
MASTER_SEED_JOINT = 20260808  # new seed: new base problems, same procedure
K = gen.K  # 3
PERMS = gen.PERMS  # all 6 permutations of range(3)
IDENTITY = tuple(range(K))
N_BASES = 150  # >= 150 per cell
CELLS = [(p, v) for p in ("P", "G") for v in ("fit", "transfer")]


def facts_for_joint(base, g):
    """Fact sentences under the JOINT action of g (see module docstring).

    Non-holds facts: identical to the frozen facts_for construction.
    Holds facts: role i reads 'names[g[i]] holds syms[i]' (same clause TEXT
    as the in-place action) but is DISPLAYED at the identity slot of role
    g[i]. Implementation: at the fact_order slot whose canonical index is
    the holds fact of role j, emit the holds clause of role inv_g[j].
    """
    syms, names = base["syms"], base["names"]
    fs = []  # canonical non-holds facts
    if base["path"] == "P":
        fs += [
            gen.FACT_BEARS.format(sym=syms[i], mark=base["marks"][i]) for i in range(K)
        ]
    else:
        ch = base["chain"]
        fs += [
            gen.FACT_GUARDS.format(a=syms[ch[j]], b=syms[ch[j + 1]])
            for j in range(K - 1)
        ]
        fs += [
            gen.FACT_SHELTERS.format(a=syms[ch[j]], b=syms[ch[j + 1]])
            for j in range(K - 1)
        ]
    n_nonhold = len(fs)
    inv_g = [g.index(i) for i in range(K)]  # inv_g[j] = g^{-1}(j)
    display = []
    for c in base["fact_order"]:
        if c < n_nonhold:
            display.append(fs[c])  # non-holds: not acted on
        else:
            j = c - n_nonhold  # identity slot of role j
            i = inv_g[j]  # role shown here under g
            display.append(gen.FACT_HOLDS.format(name=names[g[i]], sym=syms[i]))
    return display


def episode_joint(base, g):
    """Identical to the frozen gen.episode except facts_for -> facts_for_joint
    (stage/query/answer/meta construction copied verbatim in logic)."""
    if base["path"] == "P":
        qi = base["qi"]
        stage = gen.STAGE_P.format(mark=base["marks"][qi])
        query = gen.QUERY_P.format(mark=base["marks"][qi])
        answer_slot = qi
    else:
        ch = base["chain"]
        q_sym_idx = ch[base["qpos"]]
        stage = gen.STAGE_G.format(sym=base["syms"][q_sym_idx])
        query = gen.QUERY_G.format(sym=base["syms"][q_sym_idx])
        answer_slot = ch[base["qpos"] - 1]
    user = "Facts: " + " ".join(facts_for_joint(base, g)) + "\n" + stage + " " + query
    answer = base["names"][g[answer_slot]]
    meta = dict(answer_slot=answer_slot, answer=answer, g=list(g))
    if base["path"] == "G":
        ch = base["chain"]
        meta["inner_name"] = base["names"][g[ch[base["qpos"]]]]
        third = [
            i for i in range(K) if i not in (ch[base["qpos"]], ch[base["qpos"] - 1])
        ][0]
        meta["third_name"] = base["names"][g[third]]
        meta["qpos"] = base["qpos"]
    return dict(system=gen.SYSTEM, user=user, **meta)


def holds_names_in_display_order(user_text, names):
    """Names of holds clauses in the order they appear in the prompt."""
    facts_part = user_text[len("Facts: ") : user_text.index("\n")]
    out = []
    for sent in facts_part.split(". "):
        w = sent.strip().rstrip(".").split()
        if len(w) > 1 and w[1] == "holds":
            assert w[0] in names, (w[0], names)
            out.append(w[0])
    return out


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
    ok_names, report = gen.verify_single_token(tok, names24)
    ok_set = {n for n, _ in ok_names}
    n_ok = len(ok_set & set(names24))
    print(f"names single-token in situ on target tokeniser: {n_ok}/24")
    assert n_ok == 24, sorted(set(names24) - ok_set)

    manifest = dict(
        phase="9-stage1-prep",
        set="joint-permutation",
        generation_module="phase8a_generate_tasks.py (frozen; imported)",
        joint_rule=(
            "g applied to names AND holds-clause display order "
            "together: role i is held by names[g[i]] AND role i's "
            "holds clause is displayed at the identity display slot "
            "of role g[i] (the home slot of its new name). Non-holds "
            "clauses keep their identity slots. See README.md and "
            "phase9_generate_joint.py docstring."
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
        cells={},
    )

    ep_check_total, ep_check_bad = 0, 0
    for path, vocab in CELLS:
        rng = np.random.default_rng(
            (MASTER_SEED_JOINT, path == "G", vocab == "transfer")
        )
        pool = fit if vocab == "fit" else transfer
        fp = OUT / f"joint_{path}_{vocab}.jsonl"
        n = 0
        with open(fp, "w") as fh:
            for b in range(N_BASES):
                base = gen.make_base(rng, path, pool)
                orbit_name_seqs = []
                for g in PERMS:
                    ep = episode_joint(base, g)
                    ep_ip = gen.episode(base, g)  # in-place, same (base, g)
                    # invariant 1: identity perm == in-place episode
                    if g == IDENTITY:
                        assert ep["user"] == ep_ip["user"], (path, vocab, b)

                    # invariant 2: same fact multiset as in-place action
                    # (normalise: strip the trailing '.' each sentence keeps
                    # only when it is last in the ' '-joined facts line)
                    def _fact_multiset(user_text):
                        line = user_text.split("\n")[0]
                        assert line.startswith("Facts: ")
                        line = line[len("Facts: ") :]
                        return sorted(s.rstrip(".") for s in line.split(". "))

                    assert _fact_multiset(ep["user"]) == _fact_multiset(
                        ep_ip["user"]
                    ), (path, vocab, b, g)
                    # invariant 3: answer identical to in-place action
                    assert ep["answer"] == ep_ip["answer"], (path, vocab, b, g)
                    orbit_name_seqs.append(
                        tuple(holds_names_in_display_order(ep["user"], base["names"]))
                    )
                    # generation-time in-situ assertion, TARGET tokeniser
                    aid = tok.encode(" " + ep["answer"])
                    assert len(aid) == 1, (ep["answer"], aid)
                    uids = tok.encode(ep["user"])
                    ep_check_total += 1
                    if uids.count(aid[0]) < 1:
                        ep_check_bad += 1
                    rec = dict(
                        kind="joint", path=path, vocab=vocab, base_id=b, **ep, base=base
                    )
                    fh.write(json.dumps(rec) + "\n")
                    n += 1
                # invariant 4: name-at-position constant across the orbit
                assert len(set(orbit_name_seqs)) == 1, (path, vocab, b)
                assert len(orbit_name_seqs[0]) == K
        h16 = hashlib.sha256(fp.read_bytes()).hexdigest()[:16]
        manifest["cells"][f"joint/{path}/{vocab}"] = dict(
            episodes=n, bases=N_BASES, sha256_16=h16
        )
        print(f"joint/{path}/{vocab}: {n} episodes  sha16={h16}")

    assert ep_check_bad == 0, ep_check_bad
    json.dump(manifest, open(OUT / "manifest.json", "w"), indent=2)

    tokcheck = dict(
        target_model=TARGET_MODEL,
        target_revision=TARGET_REVISION,
        procedure=(
            "verify_single_token imported from frozen "
            "phase8a_generate_tasks.py (8A verification code)"
        ),
        names_verified=f"{n_ok}/24",
        episode_check=dict(total=ep_check_total, failures=ep_check_bad),
        name_report=report,
    )
    json.dump(
        tokcheck,
        open(ROOT / "results/binding_sites/joint_tokeniser_check.json", "w"),
        indent=2,
    )
    print(
        f"episode-level in-situ answer-token check: "
        f"{ep_check_total - ep_check_bad}/{ep_check_total} pass"
    )

    readme = pathlib.Path(__file__).read_text().split('"""')[1]
    (OUT / "README.md").write_text(
        "# Phase 9 joint-permutation episode set\n\n" + readme + "\n"
    )

    # sample prompts for the record: one full orbit, path G
    recs = [json.loads(l) for l in open(OUT / "joint_G_fit.jsonl")][:6]
    for r in recs[:2]:
        print(
            f"\n--- sample joint G prompt (g={r['g']}) ---\n{r['user']}\n"
            f"answer={r['answer']}"
        )


if __name__ == "__main__":
    main()

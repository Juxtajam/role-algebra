"""Phase 8A — task generation + tokeniser verification (LOCAL, no GPU).

Pre-registered constraints implemented, all mandatory:
  Path P: property -> symbol -> person (reconstruction declared: the original
    pretrained-arm template is absent from the local dir and the dv3-results
    volume; rebuilt in the same family with the ladder fixes applied).
  Path G (rebuilt):
    - acyclic chain only (A guards B, B guards C; never a cycle)  [ladder 1]
    - active predicates, two distinct verbs, no passive: relation stated in
      both lexical orientations ("X guards Y" / "Y shelters X")  [CHAIN_OUT]
    - no nested relative clauses; staged referent in its own sentence,
      query with a LITERAL REPEAT, never anaphora  [ladder 2, 5, U0-U3]
    - no stale intermediate slots in any scaffold  [ladder 4]
    - query is the final content of the prompt
    - single-token nonce names verified on THIS model's tokeniser
      (Qwen/Qwen3-32B), in situ in every template context, before generation
  Group action: permute person names in place; k=3; full six-member orbits;
  fact order fixed within orbit (randomised per base only).
  Fit/transfer name vocabularies disjoint, frequency-matched (BPE merge-rank
  pairing), >=12 names each.

Outputs (results/phase8a/tasks/):
  name_pools.json           verified pools + frequency-match evidence
  gate_{path}_{vocab}.jsonl gate episodes  (100 bases x 6 perms per cell)
  disc_{path}_{vocab}.jsonl discriminator episodes (300 bases x 6 perms per cell)
  manifest.json             counts, seeds, template hashes
"""

import hashlib
import itertools
import json
import pathlib

import numpy as np
from transformers import AutoTokenizer

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "results/verdict/gate/tasks"
MODEL = "Qwen/Qwen3-32B"
MASTER_SEED = 20260807
K = 3
PERMS = [tuple(p) for p in itertools.permutations(range(K))]
N_GATE_BASES = 100  # >= 90 required per cell
N_DISC_BASES = 300  # >= 300 required per cell

# Hand-screened nonce candidates (no English words, common names, brands,
# places; single-token status verified programmatically below).
NONCE_CANDIDATES = [
    "Bek",
    "Fon",
    "Tus",
    "Zum",
    "Miz",
    "Jal",
    "Gat",
    "Dek",
    "Suk",
    "Zam",
    "Kad",
    "Bik",
    "Kag",
    "Bav",
    "Sik",
    "Nir",
    "Lup",
    "Poz",
    "Jub",
    "Bij",
    "Ler",
    "Gim",
    "Mej",
    "Zot",
    "Peb",
    "Tud",
    "Roz",
    "Nes",
    "Kov",
    "Fus",
    "Zub",
    "Kut",
    "Tup",
    "Vib",
    "Dop",
    "Bez",
    "Laz",
    "Bip",
    "Ruf",
    "Lig",
    "Geg",
    "Jad",
    "Fot",
    "Raz",
    "Vij",
    "Kes",
    "Sug",
    "Lal",
    "Sok",
    "Kul",
    "Kum",
    "Kis",
    "Paz",
    "Lod",
    "Zuk",
    "Nel",
    "Rif",
    "Mim",
    "Tig",
]

MARKS = [
    "crimson",
    "amber",
    "violet",
    "copper",
    "ivory",
    "scarlet",
    "golden",
    "silver",
    "maroon",
    "indigo",
    "teal",
    "coral",
]
SYMBOLS = [
    "anchor",
    "arrow",
    "bell",
    "comet",
    "crown",
    "drum",
    "feather",
    "hammer",
    "ladder",
    "lantern",
    "mirror",
    "ribbon",
    "spiral",
    "wheel",
    "acorn",
    "barrel",
]

SYSTEM = (
    "You answer questions about short fact lists. "
    "Respond with exactly one name and nothing else."
)

FACT_BEARS = "The {sym} sigil bears the {mark} mark."
FACT_HOLDS = "{name} holds the {sym} sigil."
FACT_GUARDS = "The {a} sigil guards the {b} sigil."
# Run 2 wording (single pre-authorised redesign, phase8a_gate_run1_report.md):
# run 1 used "The {b} sigil shelters the {a} sigil.", whose natural reading
# assigns the protector role to its subject (b), contradicting the guards
# clause. "relies on" keeps active voice + distinct verb, with the protected
# party in subject position: orientation consistent with FACT_GUARDS.
FACT_SHELTERS = "The {b} sigil relies on the {a} sigil."
STAGE_P = "One sigil bears the {mark} mark."
QUERY_P = "Which person holds the sigil that bears the {mark} mark?"
STAGE_G = "One sigil guards the {sym} sigil."
QUERY_G = "Which person holds the sigil that guards the {sym} sigil?"


# ------------------------------------------------------------- verification
def verify_single_token(tok, names):
    """Every name must map to exactly ONE token in situ in every context in
    which it can appear in the generated prompts: (a) after a space
    mid-sentence, (b) sentence-initial after '. ' (facts are space-joined on
    one line, so no name is ever newline-initial)."""
    report = {}
    ok_names = []
    for name in names:
        ids_sp = tok.encode(" " + name)
        ctx_a = tok.encode("Facts: " + name + " holds the anchor sigil.")
        ctx_b = tok.encode(
            "The drum sigil shelters the anchor sigil. "
            + name
            + " holds the anchor sigil."
        )
        passes = (
            len(ids_sp) == 1
            and ids_sp[0] in ctx_a
            and ids_sp[0] in ctx_b
            and ctx_a.count(ids_sp[0]) == 1
            and ctx_b.count(ids_sp[0]) == 1
        )
        report[name] = dict(
            token_id=ids_sp[0] if len(ids_sp) == 1 else None,
            n_tokens_spaced=len(ids_sp),
            in_situ_ok=bool(passes),
        )
        if passes:
            ok_names.append((name, ids_sp[0]))
    return ok_names, report


def frequency_matched_split(ok_names, n_per_side=12):
    """BPE merge rank (token id) is the frequency proxy: lower id = earlier
    merge = more frequent piece. Sort by id, take the first 2*n, pair
    consecutive ranks, split each pair across fit/transfer alternately."""
    by_id = sorted(ok_names, key=lambda t: t[1])[: 2 * n_per_side]
    fit, transfer = [], []
    for i, (name, tid) in enumerate(by_id):
        (fit if (i // 1) % 2 == 0 else transfer).append((name, tid))
    fit_ids = np.array([t for _, t in fit], dtype=float)
    tr_ids = np.array([t for _, t in transfer], dtype=float)
    evidence = dict(
        method="token-id (BPE merge rank) consecutive-rank pairing",
        fit_mean_id=float(fit_ids.mean()),
        transfer_mean_id=float(tr_ids.mean()),
        fit_median_id=float(np.median(fit_ids)),
        transfer_median_id=float(np.median(tr_ids)),
        max_pair_gap=int(np.abs(fit_ids - tr_ids).max()),
    )
    return [n for n, _ in fit], [n for n, _ in transfer], evidence


# --------------------------------------------------------------- generation
def make_base(rng, path, name_pool):
    names = list(rng.choice(name_pool, K, replace=False))
    syms = list(rng.choice(SYMBOLS, K, replace=False))
    base = dict(path=path, names=names, syms=syms)
    if path == "P":
        base["marks"] = list(rng.choice(MARKS, K, replace=False))
        base["qi"] = int(rng.integers(0, K))
        n_facts = 2 * K
    else:
        # acyclic chain over a random ordering of the 3 symbols:
        # chain[0] guards chain[1], chain[1] guards chain[2] (indices into syms)
        base["chain"] = [int(x) for x in rng.permutation(K)]
        # query the middle or tail symbol (both have a unique guard);
        # 50/50 so 'answer = chain head' is not an episode-invariant.
        base["qpos"] = int(rng.integers(1, 3))  # 1=middle, 2=tail
        n_facts = 2 * (K - 1) + K  # guards+shelters+holds
    base["fact_order"] = [int(x) for x in rng.permutation(n_facts)]
    return base


def facts_for(base, g):
    """Fact sentences under permutation g (person names permuted IN PLACE:
    symbol i is held by names[g[i]]). Order fixed by base['fact_order']."""
    syms, names = base["syms"], base["names"]
    fs = []
    if base["path"] == "P":
        fs += [FACT_BEARS.format(sym=syms[i], mark=base["marks"][i]) for i in range(K)]
    else:
        ch = base["chain"]
        fs += [
            FACT_GUARDS.format(a=syms[ch[j]], b=syms[ch[j + 1]]) for j in range(K - 1)
        ]
        fs += [
            FACT_SHELTERS.format(a=syms[ch[j]], b=syms[ch[j + 1]]) for j in range(K - 1)
        ]
    fs += [FACT_HOLDS.format(name=names[g[i]], sym=syms[i]) for i in range(K)]
    return [fs[j] for j in base["fact_order"]]


def episode(base, g):
    if base["path"] == "P":
        qi = base["qi"]
        stage = STAGE_P.format(mark=base["marks"][qi])
        query = QUERY_P.format(mark=base["marks"][qi])
        answer_slot = qi  # symbol index
    else:
        ch = base["chain"]
        q_sym_idx = ch[base["qpos"]]  # queried symbol
        stage = STAGE_G.format(sym=base["syms"][q_sym_idx])
        query = QUERY_G.format(sym=base["syms"][q_sym_idx])
        answer_slot = ch[base["qpos"] - 1]  # its guard's index
    user = "Facts: " + " ".join(facts_for(base, g)) + "\n" + stage + " " + query
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
    return dict(system=SYSTEM, user=user, **meta)


def write_cell(tok, rng, path, vocab, pool, n_bases, kind, fh):
    n = 0
    for b in range(n_bases):
        base = make_base(rng, path, pool)
        for g in PERMS:
            ep = episode(base, g)
            # in-situ verification of the answer name inside THIS prompt
            aid = tok.encode(" " + ep["answer"])
            assert len(aid) == 1, (ep["answer"], aid)
            uids = tok.encode(ep["user"])
            assert uids.count(aid[0]) >= 1, ("answer token not in prompt", ep)
            rec = dict(kind=kind, path=path, vocab=vocab, base_id=b, **ep, base=base)
            fh.write(json.dumps(rec) + "\n")
            n += 1
    return n


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(MODEL)
    ok_names, ver_report = verify_single_token(tok, NONCE_CANDIDATES)
    print(f"single-token in-situ verified: {len(ok_names)}/{len(NONCE_CANDIDATES)}")
    fit, transfer, evidence = frequency_matched_split(ok_names, 12)
    print("fit:", fit)
    print("transfer:", transfer)
    print("frequency-match:", evidence)
    assert len(set(fit) & set(transfer)) == 0 and len(fit) >= 12 and len(transfer) >= 12

    json.dump(
        dict(
            model=MODEL,
            verified=ver_report,
            fit=fit,
            transfer=transfer,
            frequency_match=evidence,
        ),
        open(OUT / "name_pools.json", "w"),
        indent=2,
    )

    manifest = dict(
        model=MODEL,
        master_seed=MASTER_SEED,
        k=K,
        perms=[list(p) for p in PERMS],
        n_gate_bases=N_GATE_BASES,
        n_disc_bases=N_DISC_BASES,
        system=SYSTEM,
        templates=dict(
            bears=FACT_BEARS,
            holds=FACT_HOLDS,
            guards=FACT_GUARDS,
            shelters=FACT_SHELTERS,
            stage_p=STAGE_P,
            query_p=QUERY_P,
            stage_g=STAGE_G,
            query_g=QUERY_G,
        ),
        cells={},
    )
    pools = dict(fit=fit, transfer=transfer)
    for kind, n_bases in (("gate", N_GATE_BASES), ("disc", N_DISC_BASES)):
        for path in ("P", "G"):
            for vocab in ("fit", "transfer"):
                rng = np.random.default_rng(
                    (MASTER_SEED, kind == "disc", path == "G", vocab == "transfer")
                )
                fp = OUT / f"{kind}_{path}_{vocab}.jsonl"
                with open(fp, "w") as fh:
                    n = write_cell(
                        tok, rng, path, vocab, pools[vocab], n_bases, kind, fh
                    )
                h = hashlib.sha256(fp.read_bytes()).hexdigest()[:16]
                manifest["cells"][f"{kind}/{path}/{vocab}"] = dict(
                    episodes=n, bases=n_bases, sha256_16=h
                )
                print(f"{kind}/{path}/{vocab}: {n} episodes  sha={h}")
    json.dump(manifest, open(OUT / "manifest.json", "w"), indent=2)

    # sample prompts for the record
    for path in ("P", "G"):
        rec = json.loads(open(OUT / f"gate_{path}_fit.jsonl").readline())
        print(
            f"\n--- sample {path} prompt ---\n{rec['user']}\n" f"answer={rec['answer']}"
        )


if __name__ == "__main__":
    main()

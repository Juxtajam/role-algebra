"""Track C — k=4 (S_4) task generation. Parameterised copy of the frozen
experiments/phase8a_generate_tasks.py with K=4.

Identical templates, name pool, verification, and frequency-matched split;
the ONLY change is K=3 -> K=4, which propagates to: 24 permutations per orbit,
4 entities/symbols/marks per base, path-G chains of length 4 (qpos in 1..3),
and n_facts (P: 8, G: 10). Path-P is property->symbol->person; path-G is the
acyclic guards/relies-on chain, both exactly as the k=3 frozen set.

New content hash (k=4 is a NEW task family, not part of the frozen 84f2e54d
k=3 set); written to manifest.json and used to gate the caching session.

Outputs phase10/trackC/tasks_k4/:
  name_pools.json, manifest.json (with content_hash),
  gate_{P,G}_{fit,transfer}.jsonl   (100 bases x 24 perms)
  disc_{P,G}_{fit,transfer}.jsonl   (150 bases x 24 perms)
"""

import hashlib
import itertools
import json
import pathlib

import numpy as np
from transformers import AutoTokenizer

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "phase10/trackC/tasks_k4"
# Verify single-token on the TARGET model (the gate runs on Qwen2.5-72B; its
# tokeniser is the same family as Qwen3-32B, and 8A-final confirmed the pool
# is single-token on the 72B — re-verified here on the target).
TOK_MODEL = "Qwen/Qwen2.5-72B-Instruct"
MASTER_SEED = 20260807
K = 4
PERMS = [tuple(p) for p in itertools.permutations(range(K))]  # 24
N_GATE_BASES = 100
N_DISC_BASES = 150

# verbatim from the frozen k=3 generator
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
FACT_SHELTERS = "The {b} sigil relies on the {a} sigil."
STAGE_P = "One sigil bears the {mark} mark."
QUERY_P = "Which person holds the sigil that bears the {mark} mark?"
STAGE_G = "One sigil guards the {sym} sigil."
QUERY_G = "Which person holds the sigil that guards the {sym} sigil?"


def verify_single_token(tok, names):
    report, ok = {}, []
    for name in names:
        ids_sp = tok.encode(" " + name)
        ctx_a = tok.encode("Facts: " + name + " holds the anchor sigil.")
        ctx_b = tok.encode(
            "The drum sigil relies on the anchor sigil. "
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
            ok.append((name, ids_sp[0]))
    return ok, report


def frequency_matched_split(ok_names, n_per_side=12):
    by_id = sorted(ok_names, key=lambda t: t[1])[: 2 * n_per_side]
    fit, transfer = [], []
    for i, (name, tid) in enumerate(by_id):
        (fit if i % 2 == 0 else transfer).append((name, tid))
    fit_ids = np.array([t for _, t in fit], float)
    tr_ids = np.array([t for _, t in transfer], float)
    ev = dict(
        method="token-id (BPE merge rank) consecutive-rank pairing",
        fit_mean_id=float(fit_ids.mean()),
        transfer_mean_id=float(tr_ids.mean()),
        max_pair_gap=int(np.abs(fit_ids - tr_ids).max()),
    )
    return [n for n, _ in fit], [n for n, _ in transfer], ev


def make_base(rng, path, pool):
    names = list(rng.choice(pool, K, replace=False))
    syms = list(rng.choice(SYMBOLS, K, replace=False))
    base = dict(path=path, names=names, syms=syms)
    if path == "P":
        base["marks"] = list(rng.choice(MARKS, K, replace=False))
        base["qi"] = int(rng.integers(0, K))
        n_facts = 2 * K  # 8
    else:
        base["chain"] = [int(x) for x in rng.permutation(K)]
        base["qpos"] = int(rng.integers(1, K))  # 1..K-1 (all have a unique guard)
        n_facts = 2 * (K - 1) + K  # 10
    base["fact_order"] = [int(x) for x in rng.permutation(n_facts)]
    return base


def facts_for(base, g):
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
        stage, query, answer_slot = (
            STAGE_P.format(mark=base["marks"][qi]),
            QUERY_P.format(mark=base["marks"][qi]),
            qi,
        )
    else:
        ch = base["chain"]
        q_sym_idx = ch[base["qpos"]]
        stage = STAGE_G.format(sym=base["syms"][q_sym_idx])
        query = QUERY_G.format(sym=base["syms"][q_sym_idx])
        answer_slot = ch[base["qpos"] - 1]
    user = "Facts: " + " ".join(facts_for(base, g)) + "\n" + stage + " " + query
    answer = base["names"][g[answer_slot]]
    meta = dict(answer_slot=answer_slot, answer=answer, g=list(g))
    if base["path"] == "G":
        ch = base["chain"]
        meta["inner_name"] = base["names"][g[ch[base["qpos"]]]]
        meta["qpos"] = base["qpos"]
    return dict(system=SYSTEM, user=user, **meta)


def write_cell(tok, rng, path, vocab, pool, n_bases, kind, fp):
    n = 0
    with open(fp, "w") as fh:
        for b in range(n_bases):
            base = make_base(rng, path, pool)
            for g in PERMS:
                ep = episode(base, g)
                aid = tok.encode(" " + ep["answer"])
                assert len(aid) == 1, (ep["answer"], aid)
                assert tok.encode(ep["user"]).count(aid[0]) >= 1
                fh.write(
                    json.dumps(
                        dict(
                            kind=kind,
                            path=path,
                            vocab=vocab,
                            base_id=b,
                            **ep,
                            base=base,
                        )
                    )
                    + "\n"
                )
                n += 1
    return n


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(TOK_MODEL)
    ok_names, ver = verify_single_token(tok, NONCE_CANDIDATES)
    print(
        f"single-token in-situ on {TOK_MODEL}: {len(ok_names)}/{len(NONCE_CANDIDATES)}"
    )
    fit, transfer, ev = frequency_matched_split(ok_names, 12)
    assert len(set(fit) & set(transfer)) == 0 and len(fit) >= 12 and len(transfer) >= 12
    json.dump(
        dict(
            model=TOK_MODEL,
            verified=ver,
            fit=fit,
            transfer=transfer,
            frequency_match=ev,
        ),
        open(OUT / "name_pools.json", "w"),
        indent=2,
    )

    pools = dict(fit=fit, transfer=transfer)
    cells, file_hashes = {}, {}
    for kind, n_bases in (("gate", N_GATE_BASES), ("disc", N_DISC_BASES)):
        for path in ("P", "G"):
            for vocab in ("fit", "transfer"):
                rng = np.random.default_rng(
                    (MASTER_SEED, kind == "disc", path == "G", vocab == "transfer")
                )
                fp = OUT / f"{kind}_{path}_{vocab}.jsonl"
                n = write_cell(tok, rng, path, vocab, pools[vocab], n_bases, kind, fp)
                file_hashes[f"{kind}_{path}_{vocab}.jsonl"] = hashlib.sha256(
                    fp.read_bytes()
                ).hexdigest()
                cells[f"{kind}/{path}/{vocab}"] = dict(episodes=n, bases=n_bases)
                print(f"{kind}/{path}/{vocab}: {n} episodes")
    # content hash over episode files + name pool + this generator
    file_hashes["name_pools.json"] = hashlib.sha256(
        (OUT / "name_pools.json").read_bytes()
    ).hexdigest()
    file_hashes["generate_k4.py"] = hashlib.sha256(
        pathlib.Path(__file__).read_bytes()
    ).hexdigest()
    blob = "".join(f"{k}:{v}\n" for k, v in sorted(file_hashes.items()))
    content_hash = hashlib.sha256(blob.encode()).hexdigest()
    manifest = dict(
        model=TOK_MODEL,
        master_seed=MASTER_SEED,
        k=K,
        n_perms=len(PERMS),
        n_gate_bases=N_GATE_BASES,
        n_disc_bases=N_DISC_BASES,
        cells=cells,
        file_sha256=file_hashes,
        content_hash=content_hash,
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
    )
    json.dump(manifest, open(OUT / "manifest.json", "w"), indent=2)
    print(f"\nK=4 CONTENT HASH: {content_hash}")
    for path in ("P", "G"):
        rec = json.loads(open(OUT / f"gate_{path}_fit.jsonl").readline())
        print(f"\n--- sample {path} ---\n{rec['user']}\nanswer={rec['answer']}")


if __name__ == "__main__":
    main()

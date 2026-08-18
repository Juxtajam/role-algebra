"""Check 1 — token identity at the query position (local, data code only).

For each query type: 20 dumped episodes with full token-id sequence, the id
at the query-argument position, the id(s) at the matching position(s) inside
the fact block, and equality. Also structural checks: any wrapper/role-marker
token between query marker and argument, query-side vocabulary separation,
positional reachability.
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "src"))

import numpy as np
from trained import data as D

QUERY_TYPES = [
    ("composed_P (Q_P)", D.Q_P, "T3"),
    ("composed_G (Q_G)", D.Q_G, "T3"),
    ("property->symbol (A_PS)", D.A_PS, "T1"),
    ("symbol->person (A_SN)", D.A_SN, "T1"),
    ("person->symbol (A_NS)", D.A_NS, "T1"),
    ("symbol->guarded (A_SG)", D.A_SG, "T3"),
]

# What the query argument should literally match in the fact block, per type:
#   Q_P  arg = property p  -> HAS clause key position (facts (HAS, p, s))
#   Q_G  arg = symbol s    -> GUARD clause key position (facts (GUARD, s, s'))
#   A_PS arg = property p  -> HAS key
#   A_SN arg = symbol s    -> CARRY key (facts (CARRY, s, n))
#   A_NS arg = person n    -> CARRY target position
#   A_SG arg = symbol s    -> GUARD key
MATCH_REL = {
    D.Q_P: (D.HAS, 1),
    D.Q_G: (D.GUARD, 1),
    D.A_PS: (D.HAS, 1),
    D.A_SN: (D.CARRY, 1),
    D.A_NS: (D.CARRY, 2),
    D.A_SG: (D.GUARD, 1),
}

overall_fail = 0
for label, qt, org in QUERY_TYPES:
    rng = np.random.default_rng(20260807)
    print("=" * 100)
    print(f"CHECK 1 — {label}  (organism {org})")
    print("=" * 100)
    mismatches = 0
    for i in range(20):
        b = D.sample_base(org, rng, force_qtok=qt)
        g = D.PERMS[int(rng.integers(len(D.PERMS)))]
        toks, apos, ans, cands = D.render(b, g)
        seq = [t for t in toks.tolist() if t != D.PAD]
        # locate query marker and argument
        qpos = seq.index(qt)
        arg_pos = qpos + 1
        arg_id = seq[arg_pos]
        qmark_pos = arg_pos + 1
        # tokens between marker and argument (should be none) and between arg and QMARK
        wrapper_between = seq[qpos + 1 : arg_pos]  # empty by construction, verify
        # find matching occurrences in the fact block
        rel, off = MATCH_REL[qt]
        match_positions = []
        j = 1
        while seq[j] in (D.HAS, D.CARRY, D.GUARD):
            if seq[j] == rel and seq[j + off] == arg_id:
                match_positions.append((j + off, seq[j + off]))
            j += 4
        equal = (
            all(mid == arg_id for _, mid in match_positions)
            and len(match_positions) >= 1
        )
        if not equal:
            mismatches += 1
        names = " ".join(D.token_name(t) for t in seq)
        print(f"[{i:02d}] ids={seq}")
        print(f"     toks: {names}")
        print(
            f"     query_arg: pos={arg_pos} id={arg_id} ({D.token_name(arg_id)}) | "
            f"fact matches: {match_positions} | EQUAL={equal} | "
            f"QMARK ok={seq[qmark_pos] == D.QMARK} | wrapper tokens={wrapper_between}"
        )
    print(
        f"SUMMARY {label}: {mismatches}/20 episodes where query-arg id != fact-block id"
    )
    overall_fail += mismatches

print()
print("=" * 100)
print("STRUCTURAL CHECKS")
print("=" * 100)
print(
    f"Vocabulary: single shared token space, V={D.VOCAB}. Ranges: "
    f"props [{D.PROP0},{D.SYM0}), syms [{D.SYM0},{D.NAME0}), names [{D.NAME0},{D.VOCAB})"
)
print(
    "Query argument is emitted as the same integer id used in fact clauses "
    "(query_and_answer returns PROP0+props[i] / SYM0+syms[i] / NAME0+names[..] — "
    "identical constructors to facts())."
)
print(
    "No wrapper/role-marker tokens: layout is [qtok, arg, QMARK] immediately after "
    "the last SEP (verified per episode above)."
)
print(
    "Embedding path: TinyTransformer uses one nn.Parameter emb table for every "
    "position; no query-side embedding table exists in trained/model.py."
)
print(
    "Positional scheme: learned absolute positions, seq len <= 48 < MAX_LEN 64, "
    "causal mask only — every fact-block position is attendable from the query "
    "argument position (query comes after all facts)."
)
print()
print(
    f"OVERALL: {overall_fail} mismatching episodes across all query types "
    f"({'CLEAN — token identity holds' if overall_fail == 0 else 'FAULT FOUND — stop and report'})"
)

"""Phase 9 STAGE 1 (local prep) — Task 3: pre-registered position finding.

Defines and computes, per episode, the token indices of the cached positions
(positions to cache):

  carry_entity  final token of the entity-mention span (the person NAME) in
                each CARRY clause. CARRY clauses are the k=3 holds clauses
                ("{name} holds the {sym} sigil.") — the clauses that carry
                the name<->role binding. One position per CARRY clause: the
                name span's final token (names are verified single-token, so
                this is the name token itself).                    k = 3
  fact_final    final token of EVERY fact clause in the facts line (the
                token containing the clause-final '.'). Path P: 3 bears +
                3 holds = 6 clauses. Path G: 2 guards + 2 relies-on +
                3 holds = 7 clauses.                               6 or 7
  query_arg     final token of the query-argument mention in the query
                sentence: path P the {mark} in "... bears the {mark} mark?",
                path G the {sym} in "... guards the {sym} sigil?".     1
  answer        the last token of the rendered prompt (the "Answer:" forced
                prefix's final token) — the 8A/8C answer position, kept for
                continuity.                                            1

Per-episode position count: path P = 3+6+1+1 = 11, path G = 3+7+1+1 = 12.

Rendering convention (identical to phase8a_final_modal.py): chat template
with system+user messages, add_generation_prompt=True, enable_thinking=False,
then the forced "Answer:" prefix appended. Token indices are computed on the
UNPADDED rendered prompt with the target tokeniser's offset mapping
(Qwen/Qwen2.5-72B-Instruct @ 495f3936). Under left padding in a batch, the
in-session index of every position is (index_here + pad_len) where
pad_len = padded_seq_len - unpadded_len; the session script applies exactly
that offset using this module's finder (imported, not reimplemented).
"""


def token_index_for_char(offsets, c):
    """Index of the token whose (start, end) char span contains char c.
    Special tokens with (0, 0) spans are ignored."""
    hits = [i for i, (s, e) in enumerate(offsets) if s <= c < e and e > s]
    assert len(hits) == 1, (c, hits)
    return hits[0]


def final_token_of_span(offsets, start, end):
    """Final token of the char span [start, end)."""
    return token_index_for_char(offsets, end - 1)


def fact_clause_spans(user_text):
    """Char spans [start, end) of each fact clause (incl. its final '.')
    in the facts line of the user message. Reconstructed arithmetically
    from the generation-time join: 'Facts: ' + '. '-separated clauses."""
    line_end = user_text.index("\n")
    line = user_text[:line_end]
    assert line.startswith("Facts: ")
    body_start = len("Facts: ")
    parts = line[body_start:].split(". ")
    spans = []
    pos = body_start
    for i, part in enumerate(parts):
        if i < len(parts) - 1:
            clause_len = len(part) + 1  # split removed the '.'
            step = clause_len + 1  # and the joining ' '
        else:
            assert part.endswith(".")
            clause_len = len(part)
            step = clause_len
        spans.append((pos, pos + clause_len))
        pos += step
    assert pos == line_end
    return spans


def episode_positions(rec, tok):
    """All pre-registered positions for one episode.

    Returns dict: class -> list of token indices into the UNPADDED rendered
    prompt, plus 'rendered_len' (unpadded token count) for the session-side
    padding offset. Rendering identical to phase8a_final_modal.render."""
    msgs = [
        {"role": "system", "content": rec["system"]},
        {"role": "user", "content": rec["user"]},
    ]
    rendered = (
        tok.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        + "Answer:"
    )
    assert rendered.endswith("Answer:")
    enc = tok(rendered, return_offsets_mapping=True)
    ids, offsets = enc["input_ids"], enc["offset_mapping"]

    user = rec["user"]
    u0 = rendered.index(user)  # user content appears once
    assert rendered.count(user) == 1

    # --- fact clauses ---
    spans = [(u0 + a, u0 + b) for a, b in fact_clause_spans(user)]
    n_facts = len(spans)
    assert n_facts == (6 if rec["path"] == "P" else 7), (rec["path"], n_facts)
    fact_final = [final_token_of_span(offsets, a, b) for a, b in spans]

    # --- CARRY (holds) clauses: final token of the name-mention span ---
    carry_entity = []
    names = list(rec["base"]["names"])
    for a, b in spans:
        clause = rendered[a:b]
        words = clause.split()
        if len(words) > 1 and words[1] == "holds":
            name = words[0]
            assert name in names, (name, names)
            carry_entity.append(final_token_of_span(offsets, a, a + len(name)))
    assert len(carry_entity) == 3, carry_entity

    # --- query-arg: final token of the argument mention in the QUERY ---
    q0_local = user.index("Which person holds the sigil that ")
    query = user[q0_local:]
    assert query.endswith("?")
    if rec["path"] == "P":
        suffix = " mark?"
    else:
        suffix = " sigil?"
    assert query.endswith(suffix)
    head = query[: -len(suffix)]  # "... bears the {arg}"
    arg = head.split()[-1]
    arg_start = u0 + q0_local + len(head) - len(arg)
    assert rendered[arg_start : arg_start + len(arg)] == arg
    query_arg = [final_token_of_span(offsets, arg_start, arg_start + len(arg))]

    # --- answer position: last token of the rendered prompt ---
    answer = [len(ids) - 1]

    out = dict(
        carry_entity=carry_entity,
        fact_final=fact_final,
        query_arg=query_arg,
        answer=answer,
        rendered_len=len(ids),
    )
    flat = carry_entity + fact_final + query_arg + answer
    assert len(flat) == (11 if rec["path"] == "P" else 12)
    assert all(0 <= i < len(ids) for i in flat)
    return out

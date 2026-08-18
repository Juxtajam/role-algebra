"""Stage 2 task family (spec v3, "Task and model").

Two-hop composed queries: path P (property -> symbol -> person) and path G
(guard relation -> symbol -> person); the group action permutes person names
in place. Corrections from prior failed runs, all encoded here:

  * fact clause order is key-before-target (CARRY s n, HAS p s, GUARD a b) so
    the hops the composed query traverses are forward reads
  * name pool 800 fit / 200 transfer
  * ~40% single-hop auxiliary queries in the mixture
  * episode-local candidate masking at evaluation (chance = 1/k)
  * k = 3 only

Training distributions, ordered by role-reuse pressure:
  T0  both paths, 800 names, 100% positional shortcut leak   (lowest)
  T1  P only, 24 names                                       (low)
  T2  P only, 800 names                                      (medium)
  T3  both paths interleaved, 800 names                      (highest)
"""

import itertools

import numpy as np

# ---- vocabulary -----------------------------------------------------------
PAD, BOS, SEP, QMARK = 0, 1, 2, 3
CARRY, HAS, GUARD = 4, 5, 6
Q_P, Q_G = 7, 8
A_PS, A_SN, A_NS, A_SG = 9, 10, 11, 12
PROP0, N_PROPS = 13, 16
SYM0, N_SYMS = 29, 16
NAME0, N_NAMES, N_FIT, N_TRANSFER = 45, 1000, 800, 200
VOCAB = NAME0 + N_NAMES
SEQ_LEN = 48
K = 3

PERMS = [tuple(p) for p in itertools.permutations(range(K))]
COMPOSED = (Q_P, Q_G)
AUX = (A_PS, A_SN, A_NS, A_SG)

ORG_SPECS = {
    "T0": dict(paths=("P", "G"), pool=N_FIT, leak=True),
    "T1": dict(paths=("P",), pool=24, leak=False),
    "T2": dict(paths=("P",), pool=N_FIT, leak=False),
    "T3": dict(paths=("P", "G"), pool=N_FIT, leak=False),
}
PRESSURE_ORDER = ("T0", "T1", "T2", "T3")


class Base:
    """A base problem: everything except the name permutation g."""

    __slots__ = (
        "props",
        "syms",
        "names",
        "sigma",
        "order",
        "qtok",
        "qi",
        "leak",
        "guard_facts",
    )

    def __init__(self, rng, name_pool, qtok, guard_facts, leak):
        self.props = rng.choice(N_PROPS, K, replace=False)
        self.syms = rng.choice(N_SYMS, K, replace=False)
        self.names = rng.choice(name_pool, K, replace=False)
        shift = int(rng.integers(1, K))
        self.sigma = [(i + shift) % K for i in range(K)]  # guard_of(s_i) = s_sigma[i]
        self.qtok = qtok
        self.qi = int(rng.integers(0, K))
        self.guard_facts = guard_facts
        self.leak = leak
        n_facts = 2 * K + (K if guard_facts else 0)
        self.order = list(rng.permutation(n_facts))

    def answer_slot(self, g):
        """Slot of the answer name (for name-answer queries)."""
        if self.qtok in (Q_P, A_SN, A_NS):
            return g[self.qi]
        if self.qtok == Q_G:
            return g[self.sigma[self.qi]]
        return None

    def facts(self, g):
        """Key-before-target clauses; CARRY name arguments carry the g-permuted
        assignment (the group action permutes person names in place)."""
        fs = [(HAS, PROP0 + self.props[i], SYM0 + self.syms[i]) for i in range(K)]
        fs += [(CARRY, SYM0 + self.syms[i], NAME0 + self.names[g[i]]) for i in range(K)]
        if self.guard_facts:
            fs += [
                (GUARD, SYM0 + self.syms[i], SYM0 + self.syms[self.sigma[i]])
                for i in range(K)
            ]
        return fs

    def query_and_answer(self, g):
        """(query arg token, answer token, candidate tokens)."""
        i = self.qi
        names = [NAME0 + self.names[j] for j in range(K)]
        syms = [SYM0 + self.syms[j] for j in range(K)]
        if self.qtok == Q_P:
            return PROP0 + self.props[i], names[g[i]], names
        if self.qtok == Q_G:
            return syms[i], names[g[self.sigma[i]]], names
        if self.qtok == A_PS:
            return PROP0 + self.props[i], syms[i], syms
        if self.qtok == A_SN:
            return syms[i], names[g[i]], names
        if self.qtok == A_NS:
            return names[g[i]], syms[i], syms
        if self.qtok == A_SG:
            return syms[i], syms[self.sigma[i]], syms
        raise ValueError(self.qtok)


def render(base, g):
    """-> (tokens[SEQ_LEN], answer_pos, answer_token, candidates).
    The answer is predicted at answer_pos (the QMARK position)."""
    facts = base.facts(g)
    order = list(base.order)
    arg, answer, cands = base.query_and_answer(g)
    if base.leak and answer >= NAME0:
        # T0 positional shortcut: the fact containing the answer name always
        # comes first, so position alone solves the episode
        ans_fact = next(i for i, f in enumerate(facts) if f[2] == answer)
        order = [ans_fact] + [i for i in order if i != ans_fact]
    toks = [BOS]
    for i in order:
        toks.extend(facts[i])
        toks.append(SEP)
    toks.extend([base.qtok, arg, QMARK])
    answer_pos = len(toks) - 1
    toks.append(answer)
    toks.extend([PAD] * (SEQ_LEN - len(toks)))
    return np.array(toks, dtype=np.int64), answer_pos, answer, cands


def sample_base(
    org, rng, vocab="fit", composed_only=False, path=None, leak=None, force_qtok=None
):
    spec = ORG_SPECS[org]
    if vocab == "fit":
        pool = np.arange(spec["pool"])
    else:
        pool = np.arange(N_FIT, N_FIT + N_TRANSFER)
    if path is None:
        path = spec["paths"][int(rng.integers(len(spec["paths"])))]
    if force_qtok is not None:
        if force_qtok in (Q_G, A_SG):
            assert (
                "G" in spec["paths"]
            ), f"{org} has no guard path for qtok {force_qtok}"
        qtok = force_qtok
    elif composed_only:
        qtok = Q_P if path == "P" else Q_G
    elif rng.random() < 0.4:  # ~40% single-hop auxiliaries
        aux = AUX if "G" in spec["paths"] else AUX[:3]
        qtok = aux[int(rng.integers(len(aux)))]
    else:
        qtok = Q_P if path == "P" else Q_G
    guard_facts = "G" in spec["paths"]
    return Base(rng, pool, qtok, guard_facts, spec["leak"] if leak is None else leak)


def render_batch(bases, gs):
    toks, apos, ans, cands = zip(*[render(b, g) for b, g in zip(bases, gs)])
    return (
        np.stack(toks),
        np.array(apos),
        np.array(ans),
        np.array([c + [0] * (K - len(c)) for c in cands]),
    )


def build_eval_orbits(org, vocab, n_bases, seed, leak=None, qtok=None):
    """Fixed held-out orbit sets: n_bases fresh bases, each rendered at all k!
    permutations. Composed queries by default; qtok forces a single query
    type (the per-query-type breakdown). leak=False evaluates a leaked
    organism on unleaked episodes (the T0 shortcut check)."""
    rng = np.random.default_rng(seed)
    bases = [
        sample_base(
            org,
            rng,
            vocab=vocab,
            composed_only=qtok is None,
            leak=leak,
            force_qtok=qtok,
        )
        for _ in range(n_bases)
    ]
    toks, apos, ans, cands, slot = [], [], [], [], []
    for b in bases:
        for g in PERMS:
            t, a, ansi, c = render(b, g)
            toks.append(t)
            apos.append(a)
            ans.append(ansi)
            cands.append(c)
            slot.append(b.answer_slot(g))
    # name-answer queries permute with g; symbol-answer queries are invariant
    mode = "permute" if qtok in (None, Q_P, Q_G, A_SN) else "invariant"
    return dict(
        tokens=np.stack(toks),
        answer_pos=np.array(apos),
        answers=np.array(ans),
        candidates=np.array(cands),
        slots=np.array(slot),
        n_bases=n_bases,
        mode=mode,
    )


_SPECIAL = {
    PAD: "<pad>",
    BOS: "<bos>",
    SEP: ".",
    QMARK: "?",
    CARRY: "CARRY",
    HAS: "HAS",
    GUARD: "GUARD",
    Q_P: "Q_P",
    Q_G: "Q_G",
    A_PS: "A_PS",
    A_SN: "A_SN",
    A_NS: "A_NS",
    A_SG: "A_SG",
}


def token_name(t):
    if t in _SPECIAL:
        return _SPECIAL[t]
    if PROP0 <= t < SYM0:
        return f"p{t - PROP0}"
    if SYM0 <= t < NAME0:
        return f"s{t - SYM0}"
    return f"n{t - NAME0}"


def dump_episodes(org, n=20, seed=7):
    """Dump raw episodes human-readably and verify the clause-order correction
    actually took effect: every emitted clause must be key-before-target
    (HAS p s / CARRY s n / GUARD a b), and each query type must query in the
    assumed direction (A_PS by property, A_SN/A_SG by symbol-key, A_NS by
    name — the one deliberate backward read)."""
    is_prop = lambda t: PROP0 <= t < SYM0
    is_sym = lambda t: SYM0 <= t < NAME0
    is_name = lambda t: t >= NAME0
    clause_shape = {
        HAS: (is_prop, is_sym),
        CARRY: (is_sym, is_name),
        GUARD: (is_sym, is_sym),
    }
    q_arg_shape = {
        Q_P: is_prop,
        Q_G: is_sym,
        A_PS: is_prop,
        A_SN: is_sym,
        A_NS: is_name,
        A_SG: is_sym,
    }
    rng = np.random.default_rng(seed)
    violations = 0
    for i in range(n):
        b = sample_base(org, rng)
        g = PERMS[int(rng.integers(len(PERMS)))]
        toks, apos, ans, _ = render(b, g)
        seq = [t for t in toks.tolist() if t != PAD]
        probs = []
        j = 1
        while seq[j] in clause_shape:
            rel, key, target = seq[j], seq[j + 1], seq[j + 2]
            k_ok, t_ok = clause_shape[rel]
            if not (k_ok(key) and t_ok(target) and seq[j + 3] == SEP):
                probs.append(
                    f"clause at {j} not key-before-target: "
                    f"{token_name(rel)} {token_name(key)} {token_name(target)}"
                )
            j += 4
        qtok, arg = seq[j], seq[j + 1]
        if not q_arg_shape[qtok](arg):
            probs.append(
                f"query {token_name(qtok)} arg has wrong type: {token_name(arg)}"
            )
        status = "OK " if not probs else "VIOLATION "
        print(f"[{i:02d}] {status}" + " ".join(token_name(t) for t in seq))
        for p in probs:
            print(f"     !! {p}")
        violations += len(probs)
    print(
        f"\n{org}: {violations} clause-order/direction violations in {n} episodes"
        + (
            ""
            if violations
            else " — key-before-target confirmed; "
            "aux directions as assumed (A_NS is the one deliberate backward read)"
        )
    )
    return violations


def audit_episodes(org, n=2000, seed=123):
    """Independent end-to-end audit of the training data: an independent
    symbolic solver reads ONLY the rendered token sequence, reconstructs the
    relations, solves the query, and checks it against the supervision label;
    plus loss-masking / candidate-masking / label-alignment checks. Used for
    the stopping condition: if an easy distribution fails at 50k steps, the
    fault is data or supervision, not task difficulty."""
    rng = np.random.default_rng(seed)
    failures = []
    for i in range(n):
        b = sample_base(org, rng)
        g = PERMS[int(rng.integers(len(PERMS)))]
        toks, apos, ans, cands = render(b, g)
        seq = toks.tolist()
        # -- independent symbolic solver over the raw tokens ------------------
        has, carry, guard = {}, {}, {}
        j = 1  # skip BOS
        while seq[j] not in (Q_P, Q_G) + AUX:
            rel, a, t = seq[j], seq[j + 1], seq[j + 2]
            assert seq[j + 3] == SEP, f"clause not SEP-terminated at {j}"
            {HAS: has, CARRY: carry, GUARD: guard}[rel][a] = t
            j += 4
        qtok, arg, qm = seq[j], seq[j + 1], seq[j + 2]
        if qtok == Q_P:
            solved = carry[has[arg]]
        elif qtok == Q_G:
            solved = carry[guard[arg]]
        elif qtok == A_PS:
            solved = has[arg]
        elif qtok == A_SN:
            solved = carry[arg]
        elif qtok == A_NS:
            solved = {v: k for k, v in carry.items()}[arg]
        else:  # A_SG
            solved = guard[arg]
        # -- supervision / masking / alignment checks -------------------------
        checks = [
            (solved == ans, f"solver got {solved}, label {ans}"),
            (qm == QMARK and apos == j + 2, "answer_pos misaligned with QMARK"),
            (
                seq[apos + 1] == ans,
                "label not at answer_pos + 1 (LM target misaligned)",
            ),
            (ans in list(cands), "answer missing from candidate mask"),
            (ans != PAD, "answer is PAD (would be dropped from the loss)"),
            (len(set(cands)) == K, "candidate mask malformed"),
        ]
        for ok, msg in checks:
            if not ok:
                failures.append(dict(i=i, org=org, qtok=int(qtok), msg=msg))
    return failures


def orbit_metrics(pred_tokens, ev):
    """Episode accuracy, strict-orbit accuracy, and abstract-role orbit
    consistency (pred_slot(g.x) == g[pred_slot(x)]) under candidate masking."""
    n, P = ev["n_bases"], len(PERMS)
    correct = (pred_tokens == ev["answers"]).reshape(n, P)
    # slot of the predicted name inside the episode's candidate list
    pred_slot = (
        ev["candidates"].reshape(n, P, K) == pred_tokens.reshape(n, P, 1)
    ).argmax(axis=2)
    cons = np.zeros((n, P), dtype=bool)
    if ev.get("mode", "permute") == "invariant":
        # symbol answers do not move under g: consistency = same prediction
        # across the whole orbit
        cons = pred_slot == pred_slot[:, :1]
    else:
        for pi, g in enumerate(PERMS):
            # identity perm is PERMS[0]; consistency vs the identity episode
            g_arr = np.array(g)
            cons[:, pi] = pred_slot[:, pi] == g_arr[pred_slot[:, 0]]
    return dict(
        episode_acc=float(correct.mean()),
        strict_orbit_acc=float(correct.all(axis=1).mean()),
        orbit_consistency=float(cons.all(axis=1).mean()),
    )

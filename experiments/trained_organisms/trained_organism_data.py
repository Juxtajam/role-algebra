"""Track B organism data generation — LOCAL extraction for the discriminator.

Lifted from `phase10/trackB/finetune_modal.py` (the code of record for all 24
fine-tunes; byte-identical protocol across batches per the diff recorded in
the Track B report). The token layout, BaseEpisode, render, sample_base,
build_eval_orbits and orbit_metrics below are VERBATIM ports of the inner
definitions of `finetune()` (lines 66–320 of finetune_modal.py), with three
additions, each marked ADDED:

  1. `render_joint` / joint orbits — the Phase 9 joint name+order permutation
     (g permutes slot content across fact positions), used by the descriptive
     C7 condition. In-place rendering keeps every slot's clauses at fixed
     textual positions across an orbit; joint rendering moves slot i's
     clauses to position g(i) within each fact block, decoupling role from
     prompt position.
  2. Carry-position tracking: `build_eval_orbits` also returns, per episode,
     the textual position of the final token (the name) of each CARRY clause,
     indexed BY SLOT, for the entity_mention_carry position class.
  3. An explicit `arm` parameter replacing the enclosing function's variable.

Nothing else is changed. Chance level for name-answer queries is 1/k under
episode-local candidate masking.
"""

import numpy as np

# ---- Token layout (verbatim; matches trained.data VOCAB=1045, NAME0=45) ----
PAD, BOS, SEP, QMARK = 0, 1, 2, 3
CARRY, HAS, GUARD = 4, 5, 6
Q_P, Q_G = 7, 8
Q_3H = 9
A_PS, A_SN, A_NS, A_SG = 10, 11, 12, 13
PROP0 = 14
N_PROPS = 15
SYM0 = 29
N_SYMS = 16
NAME0 = 45
N_NAMES = 1000
VOCAB = 1045
SEQ_LEN = 64

N_TRANSFER = 200

K3, K4 = 3, 4
PERMS3 = [
    (0, 1, 2),
    (1, 0, 2),
    (0, 2, 1),
    (2, 1, 0),
    (1, 2, 0),
    (2, 0, 1),
]
GEN3 = [(1, 0, 2), (0, 2, 1)]
MOVED3 = {(1, 0, 2): (0, 1), (0, 2, 1): (1, 2)}


def n_fit_names(arm):
    if arm == "R":
        return 800
    if arm == "C":
        return 24
    raise ValueError(f"Unknown arm: {arm}")


def name_pool(arm, vocab):
    nf = n_fit_names(arm)
    if vocab == "fit":
        return np.arange(nf)
    return np.arange(nf, nf + N_TRANSFER)


def perms_for(k):
    assert k == K3, "discriminator battery is k=3 only (D19: k4 UNAVAILABLE)"
    return PERMS3


def generators_for(k):
    assert k == K3
    return GEN3


class BaseEpisode:
    __slots__ = (
        "k",
        "props",
        "syms",
        "names",
        "sigma",
        "qtok",
        "qi",
        "guard_facts",
        "order",
    )

    def __init__(self, rng, k, n_pool, s_pool, p_pool, qtok, guard_facts=True):
        self.k = k
        self.props = rng.choice(p_pool, k, replace=False)
        self.syms = rng.choice(s_pool, k, replace=False)
        self.names = rng.choice(n_pool, k, replace=False)
        shift = int(rng.integers(1, k))
        self.sigma = [(i + shift) % k for i in range(k)]
        self.qtok = qtok
        self.qi = int(rng.integers(0, k))
        self.guard_facts = guard_facts
        n_facts = 2 * k + (k if guard_facts else 0)
        self.order = list(rng.permutation(n_facts))

    def answer_slot(self, g):
        if self.qtok in (Q_P, Q_G, Q_3H, A_SN, A_NS):
            return g[self.qi]
        if self.qtok in (A_PS, A_SG):
            return self.qi
        return None

    def facts(self, g, joint=False):
        """Fact triples. joint=False: in-place (positions fixed, names move).
        joint=True (ADDED): slot content moves with g inside each block, so
        role and prompt position decouple across the orbit."""
        idx = list(g) if joint else list(range(self.k))
        fs = []
        for j in range(self.k):
            i = idx[j]
            fs.append((HAS, PROP0 + int(self.props[i]), SYM0 + int(self.syms[i])))
        for j in range(self.k):
            i = idx[j]
            fs.append((CARRY, SYM0 + int(self.syms[i]), NAME0 + int(self.names[g[i]])))
        if self.guard_facts:
            for j in range(self.k):
                i = idx[j]
                fs.append(
                    (
                        GUARD,
                        SYM0 + int(self.syms[i]),
                        SYM0 + int(self.syms[self.sigma[i]]),
                    )
                )
        return fs

    def query_and_answer(self, g):
        i = self.qi
        ns = [NAME0 + int(self.names[j]) for j in range(self.k)]
        ss = [SYM0 + int(self.syms[j]) for j in range(self.k)]
        ps = [PROP0 + int(self.props[j]) for j in range(self.k)]
        if self.qtok == Q_P:
            return ps[i], ns[g[i]], ns
        if self.qtok == Q_G:
            return ss[i], ns[g[self.sigma[i]]], ns
        if self.qtok == Q_3H:
            return ps[i], ns[g[self.sigma[i]]], ns
        if self.qtok == A_PS:
            return ps[i], ss[i], ss
        if self.qtok == A_SN:
            return ss[i], ns[g[i]], ns
        if self.qtok == A_NS:
            return ns[g[i]], ss[i], ss
        if self.qtok == A_SG:
            return ss[i], ss[self.sigma[i]], ss
        raise ValueError(f"Unknown query type: {self.qtok}")


def render(base, g, joint=False):
    """Returns (tokens, answer_pos, answer, candidates, carry_pos_by_slot).
    carry_pos_by_slot[i] = textual position of the name token of slot i's
    CARRY clause (ADDED — needed for entity_mention_carry)."""
    facts = base.facts(g, joint=joint)
    order = list(base.order)
    arg, answer, cands = base.query_and_answer(g)

    # Which fact index is slot i's CARRY clause?  In-place: index k+i.
    # Joint: the block was re-indexed by g, so slot i sits at block position
    # g^{-1}(i); its fact index is k + g.index(i).
    if joint:
        carry_fact_idx = {i: base.k + list(g).index(i) for i in range(base.k)}
    else:
        carry_fact_idx = {i: base.k + i for i in range(base.k)}

    toks = [BOS]
    fact_start = {}
    for idx in order:
        fact_start[idx] = len(toks)
        toks.extend(facts[idx])
        toks.append(SEP)
    toks.extend([base.qtok, arg, QMARK])
    answer_pos = len(toks) - 1
    toks.append(answer)
    pad_needed = SEQ_LEN - len(toks)
    if pad_needed < 0:
        raise ValueError(f"Episode too long: {len(toks)} tokens (k={base.k})")
    toks.extend([PAD] * pad_needed)

    carry_pos = np.array(
        [fact_start[carry_fact_idx[i]] + 2 for i in range(base.k)], dtype=np.int64
    )
    return (np.array(toks, dtype=np.int64), answer_pos, answer, cands, carry_pos)


def sample_base(rng, arm, vocab="fit", force_qtok=None, force_k=None):
    if force_k is not None:
        k = force_k
    else:
        k = K3 if rng.random() < 0.5 else K4
    n_pool = name_pool(arm, vocab)
    s_pool = np.arange(N_SYMS)
    p_pool = np.arange(N_PROPS)
    if force_qtok is not None:
        qtok = force_qtok
    elif rng.random() < 0.4:
        aux_list = [A_PS, A_SN, A_NS, A_SG]
        qtok = aux_list[int(rng.integers(len(aux_list)))]
    else:
        qtok = Q_P if rng.random() < 0.5 else Q_G
    return BaseEpisode(rng, k, n_pool, s_pool, p_pool, qtok, True)


def build_eval_orbits(
    arm, vocab, n_bases, seed, qtok=None, force_k=K3, perm_mode="full", joint=False
):
    """Fixed held-out orbit evaluation set (verbatim structure + ADDED
    carry positions and joint mode). k=3 only for the discriminator."""
    rng = np.random.default_rng(seed)
    perm_fn = perms_for if perm_mode == "full" else generators_for

    bases = [
        sample_base(rng, arm, vocab=vocab, force_qtok=qtok, force_k=force_k)
        for _ in range(n_bases)
    ]

    toks_list, apos_list, ans_list, cands_list, slot_list = [], [], [], [], []
    carry_list, k_vals = [], []
    for b in bases:
        for g in perm_fn(b.k):
            t, a, ansi, c, cp = render(b, g, joint=joint)
            toks_list.append(t)
            apos_list.append(a)
            ans_list.append(ansi)
            cands_list.append(c)
            slot_list.append(b.answer_slot(g))
            carry_list.append(cp)
            k_vals.append(b.k)

    mode = "permute" if qtok in (None, Q_P, Q_G, Q_3H, A_SN) else "invariant"
    max_k = max(k_vals)
    padded = np.zeros((len(cands_list), max_k), dtype=np.int64)
    for i, c in enumerate(cands_list):
        padded[i, : len(c)] = c

    return dict(
        tokens=np.stack(toks_list),
        answer_pos=np.array(apos_list),
        answers=np.array(ans_list),
        candidates=padded,
        slots=np.array(slot_list),
        carry_pos=np.stack(carry_list),
        n_bases=n_bases,
        mode=mode,
        k_values=np.array(k_vals),
        perm_mode=perm_mode,
        joint=bool(joint),
        seed=seed,
        qtok=qtok,
        vocab=vocab,
        arm=arm,
    )


def orbit_metrics(pred_tokens, ev):
    """Verbatim port."""
    n = ev["n_bases"]
    n_perm = len(ev["tokens"]) // n
    correct = (pred_tokens == ev["answers"]).reshape(n, n_perm)
    max_k = ev["candidates"].shape[1]
    pred_slot = (
        ev["candidates"].reshape(n, n_perm, max_k) == pred_tokens.reshape(n, n_perm, 1)
    ).argmax(axis=2)
    cons = np.zeros((n, n_perm), dtype=bool)
    if ev.get("mode", "permute") == "invariant":
        cons = pred_slot == pred_slot[:, :1]
    else:
        perm_fn = generators_for if ev.get("perm_mode") == "generators" else perms_for
        k = int(ev["k_values"][0])
        for pi, g in enumerate(perm_fn(k)):
            g_arr = np.array(g)
            cons[:, pi] = pred_slot[:, pi] == g_arr[pred_slot[:, 0]]
    return dict(
        episode_acc=float(correct.mean()),
        strict_orbit_acc=float(correct.all(axis=1).mean()),
        orbit_consistency=float(cons.all(axis=1).mean()),
        n_bases=n,
        n_perm=n_perm,
    )

"""Forced-reuse task — "permutation transfer" (operator-forcing).

The programme's finding: transformers solve role-binding by retrieval, because
the answer is always directly stated in context. This task removes that escape:
the model must INFER a permutation g from a labelled KEY set and APPLY the same
g to a QUERY set whose permuted assignment is NEVER stated — so the answer is
not retrievable, only derivable by transferring g.

Episode (k entities):
  KEY   : canonical order  slot i -> key_i           (i = 0..k-1)
          shuffled result  slot i -> key_{g(i)}       (states g, relative to canonical)
  QUERY : canonical order  slot i -> qry_i
  ASK   : slot j  ->  answer = qry_{g(j)}             (NOT stated anywhere)

key and query entity sets are DISJOINT and drawn fresh per episode from large
pools (no per-binding memorisation). g varies over S_k. The discriminator fits
R_a between the answer-position state at g and at a∘g: if the model represents
g as a linear operator on the query-role state, R_a exists (H_role); if it
routes by attention, it does not (H_retrieval / position-mixture).

k=4 (S_4, 24 perms). Token layout below; chance = 1/k under query-candidate
masking.
"""

import itertools

import numpy as np

# ---- token layout ----
PAD, BOS, SEP, KEY, SHUF, QRY, ASK = 0, 1, 2, 3, 4, 5, 6
SLOT0 = 7  # slot index tokens SLOT0..SLOT0+k-1
K = 4
NAME0 = SLOT0 + K  # name tokens NAME0..NAME0+N_NAMES-1
N_NAMES = 2000
N_FIT = 1600  # fit-vocab names [0,N_FIT); transfer [N_FIT,N_NAMES)
VOCAB = NAME0 + N_NAMES
SEQ_LEN = 64

PERMS = [tuple(p) for p in itertools.permutations(range(K))]  # 24
PERM_IDX = {p: i for i, p in enumerate(PERMS)}
# adjacent transposition generators of S_4
GEN = [(1, 0, 2, 3), (0, 2, 1, 3), (0, 1, 3, 2)]


def compose(a, b):
    return tuple(a[b[j]] for j in range(K))


def name_pool(vocab):
    return np.arange(N_FIT) if vocab == "fit" else np.arange(N_FIT, N_NAMES)


class Base:
    """A base problem = the two entity sets + the query slot. The permutation
    g is applied on top (the orbit)."""

    __slots__ = ("key", "qry", "qj")

    def __init__(self, rng, vocab):
        pool = name_pool(vocab)
        pick = rng.choice(pool, 2 * K, replace=False)
        self.key = pick[:K]
        self.qry = pick[K:]
        self.qj = int(rng.integers(0, K))  # queried slot

    def answer(self, g):
        return NAME0 + int(self.qry[g[self.qj]])


def render(base, g):
    """Token sequence; returns (tokens, answer_pos, answer, candidates).
    candidates = the k query names (episode-local masking)."""
    t = [BOS, KEY]
    for i in range(K):  # canonical key
        t += [SLOT0 + i, NAME0 + int(base.key[i])]
    t += [SEP, SHUF]
    for i in range(K):  # shuffled key: slot i -> key_{g(i)}
        t += [SLOT0 + i, NAME0 + int(base.key[g[i]])]
    t += [SEP, QRY]
    for i in range(K):  # canonical query
        t += [SLOT0 + i, NAME0 + int(base.qry[i])]
    t += [SEP, ASK, SLOT0 + base.qj]
    answer_pos = len(t) - 1  # predict at the ASK-slot token
    t += [base.answer(g)]
    pad = SEQ_LEN - len(t)
    assert pad >= 0, len(t)
    t += [PAD] * pad
    cands = [NAME0 + int(base.qry[i]) for i in range(K)]
    return np.array(t, dtype=np.int64), answer_pos, base.answer(g), cands


def sample_base(rng, vocab):
    return Base(rng, vocab)


def build_eval(vocab, n_bases, seed, perms=None):
    """Fixed held-out orbit set. perms defaults to all 24 (full group)."""
    rng = np.random.default_rng(seed)
    perms = perms or PERMS
    bases = [sample_base(rng, vocab) for _ in range(n_bases)]
    toks, apos, ans, cands, slots = [], [], [], [], []
    for b in bases:
        for g in perms:
            tk, ap, an, cd = render(b, g)
            toks.append(tk)
            apos.append(ap)
            ans.append(an)
            cands.append(cd)
            slots.append(g[b.qj])  # abstract answer slot
    return dict(
        tokens=np.stack(toks),
        answer_pos=np.array(apos),
        answers=np.array(ans),
        candidates=np.array(cands),
        slots=np.array(slots),
        n_bases=n_bases,
        perms=perms,
    )


def orbit_metrics(preds, ev):
    n, np_ = ev["n_bases"], len(ev["perms"])
    correct = (preds == ev["answers"]).reshape(n, np_)
    return dict(
        episode_acc=float(correct.mean()), strict_orbit=float(correct.all(1).mean())
    )

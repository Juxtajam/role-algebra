"""
=== Track B Arm R (ROLE-REWARDING) — Design Specification ===
Phase 10 organism programme.

THIS IS A DESIGN DOCUMENT. No training, no GPU — the deliverable is the
complete specification for a subsequent implementation pass.

CORE HYPOTHESIS: When the training distribution is engineered so that
per-episode vocabularies are large enough that no token pair recurs across
episodes, the model CANNOT memorise name→answer associations directly.
Instead, it MUST learn the abstract role structure (HAS, CARRY, GUARD
relations plus the permutation group action) to generalise. The role
structure is the CHEAP solution.

PREDICTION (pre-registered): converged Arm R seeds will show role-operator
structure at a higher rate than Arm C seeds. If Arm R never forms structure
despite the training distribution making role abstraction pay, that is
itself an informative negative result (reported with formation evidence).

================================================================================
1. VOCABULARY — large pools, per-episode random draws
================================================================================

Per-episode random vocabulary drawn from large pools so that no token pair
recurs across episodes often enough to memorise:

  Pool          Size (fit)   Transfer   Total    Rationale
  ────────────  ───────────  ─────────  ───────  ───────────────────────────
  Properties     256           64        320     >10× the old 16; no recurrence
  Symbols        256           64        320     >10× the old 16
  Names         2000          500       2500     >10× the old 800-fit pool

The old T2/T3 pools (16 props, 16 syms, 800 names) were large enough for
the 72B model but small enough that a 4L/128d model might still learn
lexical shortcuts. These larger pools FORCE role generalisation at this
scale.

Token layout (the old IDs are shifted to make room):

  ID range        Content
  ────────────    ──────────────────────────────────────
  0                PAD (LM loss ignore_index)
  1                BOS
  2                SEP  (clause separator)
  3                QMARK (answer-position prediction token)
  4                CARRY  (relation: symbol -> person)
  5                HAS    (relation: property -> symbol)
  6                GUARD  (relation: symbol -> symbol)
  7                Q_P    (2-hop composed: property -> person, TRAIN)
  8                Q_G    (2-hop composed: symbol -> person, TRAIN)
  9                Q_3H   (3-hop composed: property -> person, TEST ONLY)
  10               A_PS   (aux single-hop: property -> symbol)
  11               A_SN   (aux single-hop: symbol -> person)
  12               A_NS   (aux single-hop: person -> symbol)
  13               A_SG   (aux single-hop: symbol -> guard_symbol)
  14..269          Property tokens  (PROP0=14, N_PROPS=256)
  270..525         Symbol tokens    (SYM0=270, N_SYMS=256)
  526..3025        Name tokens      (NAME0=526, N_FIT=2000, N_TRANSFER=500)

VOCAB = 3026.
"""

# ============================================================================
# Token constants
# ============================================================================
PAD, BOS, SEP, QMARK = 0, 1, 2, 3
CARRY, HAS, GUARD = 4, 5, 6

# --- 2-hop composed queries (training) ---
Q_P, Q_G = 7, 8

# --- 3-hop composed query (test only) ---
Q_3H = 9

# --- Single-hop auxiliaries ---
A_PS, A_SN, A_NS, A_SG = 10, 11, 12, 13

# --- Property pool ---
PROP0 = 14
N_PROPS = 256  # fit pool
N_PROPS_TRANSFER = 64
PROP0_TRANSFER = PROP0 + N_PROPS

# --- Symbol pool ---
SYM0 = PROP0 + N_PROPS + N_PROPS_TRANSFER  # = 334
N_SYMS = 256
N_SYMS_TRANSFER = 64
SYM0_TRANSFER = SYM0 + N_SYMS

# --- Name pool ---
NAME0 = SYM0 + N_SYMS + N_SYMS_TRANSFER  # = 654
N_FIT_NAMES = 2000
N_TRANSFER_NAMES = 500
N_NAMES = N_FIT_NAMES + N_TRANSFER_NAMES  # = 2500

VOCAB = NAME0 + N_NAMES  # = 3154

# Sequence length must accommodate k=4 facts:
# k=4: 2*k + k = 12 facts (if guard_facts), each fact = 3 tokens + SEP = 4
# 12 * 4 = 48 tokens + BOS + query(3) + answer(1) = 53 tokens
# Round up to next multiple for positional embedding safety.
SEQ_LEN = 64

# ============================================================================
# 2. COMPOSITION STRUCTURE — k mixed {3, 4}, train ≤2 hops, test 3 hops
# ============================================================================
"""
k mixed {3, 4} within training:
  - Each episode randomly draws k ∈ {3, 4} with equal probability.
  - k=3: 6 permutations (S3), generators (1,0,2) and (0,2,1)
  - k=4: 24 permutations (S4), generators (1,0,2,3) and (0,2,3,1)

TRAIN (≤2 hops):
  - Q_P: property -> HAS -> symbol -> CARRY -> person (2 hops)
  - Q_G: symbol -> GUARD -> symbol' -> CARRY -> person (2 hops)
  - ~40% single-hop aux mix: A_PS, A_SN, A_NS, A_SG

TEST (3 hops, UNSET during training):
  - Q_3H: property -> HAS -> symbol -> GUARD -> symbol' -> CARRY -> person
    (3 hops; the model has never seen this query type)

TRAIN GENERATORS ONLY, TEST FULL GROUP:
  - Training episodes use ONLY generator permutations:
      k=3: g12=(1,0,2), g23=(0,2,1)
      k=4: g12=(1,0,2,3), g23=(0,2,3,1)
  - Test orbits use ALL permutations:
      k=3: full S3 (6 permutations)
      k=4: full S4 (24 permutations)
  - This tests TRUE algebraic generalisation — the model sees only the
    generators during training and must infer the full group structure.

PER-EPISODE RANDOM VOCAB:
  Every episode draws fresh property, symbol, and name tokens from the
  large fit pools. No token pair recurs across episodes often enough for
  memorisation (2000 names × 256 symbols × 256 props ≈ 130M combinations;
  at 50k steps × 256 batch = 12.8M episodes, ~90% of episodes have
  completely novel triplets). Episodes are rendered with key-before-target
  clause order (HAS p s / CARRY s n / GUARD a b), then shuffled.
"""

# k=3 constants
K3 = 3
PERMS3 = [
    (0, 1, 2),  # identity
    (1, 0, 2),  # g12 — generator
    (0, 2, 1),  # g23 — generator
    (2, 1, 0),
    (1, 2, 0),
    (2, 0, 1),
]
GEN3 = [(1, 0, 2), (0, 2, 1)]  # training generators for S3

# k=4 constants
K4 = 4
PERMS4 = [
    (0, 1, 2, 3),
    (1, 0, 2, 3),
    (0, 2, 1, 3),
    (0, 1, 3, 2),
    (2, 1, 0, 3),
    (1, 2, 0, 3),
    (2, 0, 1, 3),
    (0, 3, 2, 1),
    (1, 0, 3, 2),
    (0, 2, 3, 1),
    (0, 3, 1, 2),
    (3, 1, 2, 0),
    (1, 3, 2, 0),
    (2, 0, 3, 1),
    (3, 0, 2, 1),
    (2, 3, 0, 1),
    (1, 2, 3, 0),
    (3, 2, 0, 1),
    (2, 1, 3, 0),
    (3, 1, 0, 2),
    (1, 3, 0, 2),
    (2, 3, 1, 0),
    (3, 0, 1, 2),
    (3, 2, 1, 0),
]
GEN4 = [(1, 0, 2, 3), (0, 2, 3, 1)]  # training generators for S4


# Dynamic selection functions
def perms_for(k):
    """All permutations for group S_k."""
    return {3: PERMS3, 4: PERMS4}[k]


def generators_for(k):
    """Training-only generators for S_k."""
    return {3: GEN3, 4: GEN4}[k]


# Train/test flag: perms to use during orbit evaluation
TRAIN_PERMS = "generators"  # use generators_for(k)
TEST_PERMS = "full"  # use perms_for(k) — all permutations

# Composition query IDs
COMPOSED_TRAIN = (Q_P, Q_G)  # 2-hop, seen during training
COMPOSED_TEST = (Q_3H,)  # 3-hop, test-only
AUX = (A_PS, A_SN, A_NS, A_SG)  # single-hop auxiliaries
ALL_QUERY_TYPES = (Q_P, Q_G, Q_3H) + AUX

# ============================================================================
# 3. DATA GENERATION — per-episode random vocab, key-before-target
# ============================================================================
"""
Episode structure (identical to existing data.py Base class, extended):

  Episode E = (k, {p_i}_{i=0..k-1}, {s_i}_{i=0..k-1}, {n_i}_{i=0..k-1},
               sigma, qi, qtok)
  where:
    k ∈ {3, 4}                             (random per episode)
    p_i drawn without replacement from PROP0..PROP0+N_PROPS-1
    s_i drawn without replacement from SYM0..SYM0+N_SYMS-1
    n_i drawn without replacement from NAME0..NAME0+N_FIT_NAMES-1
    sigma: cyclic shift permutation (shift ∈ {1..k-1})
    qi: query index ∈ {0..k-1}
    qtok: query type token

Facts (key-before-target clause order, identical to existing):
  HAS    p_i  s_i     for i in 0..k-1
  CARRY  s_i  n_{g[i]} for i in 0..k-1
  GUARD  s_i  s_{sigma[i]} for i in 0..k-1   (if guard_facts=True)

Query, mapped to tokens:
  Q_P:  query token Q_P,  argument p_{qi}
        → answer: n_{g[qi]} via HAS[sym probe] then CARRY[name lookup]
  Q_G:  query token Q_G,  argument s_{qi}
        → answer: n_{g[sigma[qi]]} via GUARD[sym probe] then CARRY[name lookup]
  Q_3H: query token Q_3H, argument p_{qi}
        → answer: n_{g[sigma[qi]]} via HAS→GUARD→CARRY (3-hop, TEST ONLY)
  A_PS: query token A_PS, argument p_{qi}
        → answer: s_{qi}
  A_SN: query token A_SN, argument s_{qi}
        → answer: n_{g[qi]}
  A_NS: query token A_NS, argument n_{g[qi]}
        → answer: s_{qi}
  A_SG: query token A_SG, argument s_{qi}
        → answer: s_{sigma[qi]}

Rendering (key-before-target, as in existing render()):
  [BOS] fact_clauses_separated_by_SEP [qtok arg QMARK] answer [PAD...]

  Fact clauses are ordered randomly (permuted) within each episode. T0-style
  positional leak is NEVER applied (leak=False always). Clause order is
  key-before-target: HAS p s, CARRY s n, GUARD a b.

Candidate masking:
  For name-answer queries (Q_P, Q_G, Q_3H, A_SN): candidates = {n_0..n_{k-1}}
  For symbol-answer queries (A_PS, A_NS, A_SG): candidates = {s_0..s_{k-1}}
  Chance = 1/k (episode-local masking).

Auxiliary mixture:
  ~40% single-hop aux (A_PS, A_SN, A_NS, A_SG), ~60% composed (Q_P, Q_G).
  Q_3H is NEVER sampled during training — it is test-only.
"""


class BaseEpisode:
    """A base episode: everything except the name permutation g.

    Mirrors the existing `data.Base` class but supports:
      - k ∈ {3, 4}
      - Large per-episode random vocabularies
      - 3-hop query type (test-only)
      - train-generators-only vs full-group mode
    """

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

    def __init__(
        self, rng, k, name_pool, symbol_pool, prop_pool, qtok, guard_facts=True
    ):
        self.k = k
        # Draw k distinct tokens from each pool
        self.props = rng.choice(prop_pool, k, replace=False)
        self.syms = rng.choice(symbol_pool, k, replace=False)
        self.names = rng.choice(name_pool, k, replace=False)
        # Cyclic shift permutation sigma: guard_of(s_i) = s_{sigma[i]}
        shift = int(rng.integers(1, k))
        self.sigma = [(i + shift) % k for i in range(k)]
        self.qtok = qtok
        self.qi = int(rng.integers(0, k))
        self.guard_facts = guard_facts
        # Random fact-clause order
        n_facts = 2 * k + (k if guard_facts else 0)
        self.order = list(rng.permutation(n_facts))

    def answer_slot(self, g):
        """Slot of the answer within the episode's candidate list (0-based).

        For name-answer queries, permutes with g.
        For symbol-answer queries, invariant under g.
        """
        if self.qtok in (Q_P, Q_G, Q_3H, A_SN):
            return g[self.qi]
        if self.qtok == A_NS:
            return g[self.qi]
        if self.qtok in (A_PS, A_SG):
            return self.qi
        return None

    def facts(self, g):
        """Key-before-target clauses.

        HAS    p_i  s_i           for i in 0..k-1
        CARRY  s_i  n_{g[i]}     (names permuted by g)
        GUARD  s_i  s_{sigma[i]}  (cyclic shift, invariant under g)
        """
        fs = []
        for i in range(self.k):
            fs.append((HAS, PROP0 + int(self.props[i]), SYM0 + int(self.syms[i])))
        for i in range(self.k):
            fs.append((CARRY, SYM0 + int(self.syms[i]), NAME0 + int(self.names[g[i]])))
        if self.guard_facts:
            for i in range(self.k):
                fs.append(
                    (
                        GUARD,
                        SYM0 + int(self.syms[i]),
                        SYM0 + int(self.syms[self.sigma[i]]),
                    )
                )
        return fs

    def query_and_answer(self, g):
        """Returns (query_arg_token, answer_token, candidate_token_list)."""
        i = self.qi
        names = [NAME0 + int(self.names[j]) for j in range(self.k)]
        syms = [SYM0 + int(self.syms[j]) for j in range(self.k)]
        props = [PROP0 + int(self.props[j]) for j in range(self.k)]

        if self.qtok == Q_P:
            # 2-hop: property -> symbol -> person
            return props[i], names[g[i]], names
        if self.qtok == Q_G:
            # 2-hop: symbol -> guard_symbol -> person
            return syms[i], names[g[self.sigma[i]]], names
        if self.qtok == Q_3H:
            # 3-hop: property -> symbol -> guard_symbol -> person (TEST ONLY)
            return props[i], names[g[self.sigma[i]]], names
        if self.qtok == A_PS:
            return props[i], syms[i], syms
        if self.qtok == A_SN:
            return syms[i], names[g[i]], names
        if self.qtok == A_NS:
            return names[g[i]], syms[i], syms
        if self.qtok == A_SG:
            return syms[i], syms[self.sigma[i]], syms
        raise ValueError(f"Unknown query type: {self.qtok}")


def render(base, g):
    """Render an episode to token sequence.

    Returns: (tokens[SEQ_LEN], answer_pos, answer_token, candidates)

    tokens layout:
      [BOS] fact_clauses... [qtok arg QMARK] answer [PAD...]
      where each fact_clause is [rel key target SEP]
    """
    facts = base.facts(g)
    order = list(base.order)
    arg, answer, cands = base.query_and_answer(g)

    toks = [BOS]
    for idx in order:
        toks.extend(facts[idx])
        toks.append(SEP)
    toks.extend([base.qtok, arg, QMARK])
    answer_pos = len(toks) - 1
    toks.append(answer)
    # Pad
    k = base.k
    pad_needed = SEQ_LEN - len(toks)
    if pad_needed < 0:
        raise ValueError(
            f"Episode too long for SEQ_LEN={SEQ_LEN}: {len(toks)} tokens " f"(k={k})"
        )
    toks.extend([PAD] * pad_needed)
    return (np.array(toks, dtype=np.int64), answer_pos, answer, cands)


def render_batch(bases, gs):
    """Batch render: list of (base, g) -> stacked arrays."""
    toks, apos, ans, cands = zip(*[render(b, g) for b, g in zip(bases, gs)])
    max_k = max(b.k for b in bases)
    padded_cands = []
    for c in cands:
        padded = list(c) + [0] * (max_k - len(c))
        padded_cands.append(padded)
    return (
        np.stack(toks),
        np.array(apos),
        np.array(ans),
        np.array(padded_cands, dtype=np.int64),
    )


def sample_base(rng, vocab="fit", force_qtok=None, force_k=None):
    """Sample a random BaseEpisode.

    Args:
        rng: numpy random generator
        vocab: "fit" (training pool) or "transfer" (held-out pool)
        force_qtok: if set, force this query type (for per-type breakdowns)
        force_k: if set, force k=3 or k=4 (for per-k breakdowns)

    Returns: BaseEpisode

    Training distribution (vocab="fit"):
      - k randomly chosen from {3, 4} (equal probability)
      - ~40% single-hop aux, ~60% composed (Q_P, Q_G)
      - Q_3H NEVER sampled (test-only)
      - Guard facts always present

    Transfer distribution (vocab="transfer"):
      - Same structure but using transfer-pool tokens
    """
    if force_k is not None:
        k = force_k
    else:
        k = K3 if rng.random() < 0.5 else K4

    if vocab == "fit":
        name_pool = np.arange(N_FIT_NAMES)
        sym_pool = np.arange(N_SYMS)
        prop_pool = np.arange(N_PROPS)
    else:
        name_pool = np.arange(N_FIT_NAMES, N_FIT_NAMES + N_TRANSFER_NAMES)
        sym_pool = np.arange(N_SYMS, N_SYMS + N_SYMS_TRANSFER)
        prop_pool = np.arange(N_PROPS, N_PROPS + N_PROPS_TRANSFER)

    if force_qtok is not None:
        qtok = force_qtok
    elif rng.random() < 0.4:
        # ~40% single-hop auxiliaries
        aux_list = list(AUX)
        qtok = aux_list[int(rng.integers(len(aux_list)))]
    else:
        # ~60% composed (Q_P or Q_G — 2-hop, NOT Q_3H)
        qtok = Q_P if rng.random() < 0.5 else Q_G

    guard_facts = True  # always present for role structure

    return BaseEpisode(rng, k, name_pool, sym_pool, prop_pool, qtok, guard_facts)


# ============================================================================
# 4. EVALUATION BATTERY
# ============================================================================
"""
Every evaluation point (every 500 steps, and final) runs the FULL battery:

  (a) Held-out fit-vocab composed accuracy (2-hop Q_P + Q_G)
      - n_bases=192, full group orbits
      - strict-orbit accuracy (all permutations correct for a base)
      - abstract-role orbit consistency

  (b) Held-out transfer-vocab composed accuracy
      - n_bases=96, full group orbits
      - Tests whether the abstract role structure generalises to unseen names
      - Pass threshold: >= 0.95

  (c) Held-out fit-vocab single-hop aux accuracy (per query type)
      - n_bases=64 per type, full orbits
      - A_PS, A_SN, A_NS, A_SG

  (d) Train-distribution accuracy (composed queries only)
      - 96 bases sampled from current training pool, separate eval rng stream

  (e) 3-HOP GENERALISATION (test-only, the KEY metric)
      - Q_3H query on held-out fit vocab, full group orbits
      - n_bases=192
      - Model has NEVER seen this query type during training
      - If the model learned the abstract role structure, it should solve
        this by composing HAS→GUARD→CARRY without explicit 3-hop training
      - Reported as: 3hop_acc, 3hop_strict, 3hop_consistency

  (f) Per-k breakdown
      - Separate eval sets for k=3 and k=4
      - Reports accuracy per query type per k
      - Tests whether k-mixing impairs either group size

  (g) UNSEEN PERMUTATION PRODUCTS (test-only)
      - Train on generators only (g12, g23)
      - Test on full group (all 6 or 24 permutations)
      - Report: gen_acc (accuracy on generator permutations) vs
        full_acc (accuracy on all permutations incl. products like g12∘g23)
      - The product permutations are UNSEEN during training
      - If the model learned the group structure, full_acc ≈ gen_acc

  (h) Induction head survival (Step 4, per eval)
      - Copy accuracy + per-head mass on held-out induction sequences
      - Tracks whether the induction head survives task finetuning
"""


def build_eval_orbits(vocab, n_bases, seed, qtok=None, force_k=None, perm_mode="full"):
    """Build fixed held-out orbit evaluation set.

    Args:
        vocab: "fit" or "transfer"
        n_bases: number of fresh bases
        seed: rng seed
        qtok: force single query type (for per-type breakdowns)
        force_k: force k=3 or k=4 (for per-k breakdowns)
        perm_mode: "full" (all permutations) or "generators" (training-only)

    Returns: dict with tokens, answer_pos, answers, candidates, slots,
             n_bases, mode, k_values, perm_mode
    """
    rng = np.random.default_rng(seed)

    # Determine permutations to use
    if perm_mode == "full":
        perm_fn = perms_for
    else:
        perm_fn = generators_for

    bases = []
    for _ in range(n_bases):
        actual_k = force_k if force_k is not None else K3
        if qtok is not None:
            b = sample_base(rng, vocab=vocab, force_qtok=qtok, force_k=actual_k)
        else:
            b = sample_base(rng, vocab=vocab, force_k=actual_k)
        bases.append(b)

    toks_list, apos_list, ans_list, cands_list, slot_list = [], [], [], [], []
    k_vals = []

    for b in bases:
        perms = perm_fn(b.k)
        for g in perms:
            t, a, ansi, c = render(b, g)
            toks_list.append(t)
            apos_list.append(a)
            ans_list.append(ansi)
            cands_list.append(c)
            slot_list.append(b.answer_slot(g))
            k_vals.append(b.k)

    # Determine mode for orbit_metrics
    if qtok in (None, Q_P, Q_G, Q_3H, A_SN):
        mode = "permute"
    else:
        mode = "invariant"

    # Pad candidates to max k
    max_k = max(k_vals)
    padded_cands = np.zeros((len(cands_list), max_k), dtype=np.int64)
    for i, c in enumerate(cands_list):
        padded_cands[i, : len(c)] = c

    return dict(
        tokens=np.stack(toks_list),
        answer_pos=np.array(apos_list),
        answers=np.array(ans_list),
        candidates=padded_cands,
        slots=np.array(slot_list),
        n_bases=n_bases,
        mode=mode,
        k_values=np.array(k_vals),
        perm_mode=perm_mode,
    )


# ============================================================================
# 5. ARCHITECTURE — identical to Phase 4b/5, scaled to 8 layers
# ============================================================================
"""
TinyTransformer at 8 layers, d_model=128, 4 heads, d_ff=512, pre-LN, fp32,
learned absolute positional embeddings, max seq len=64.

Fit-pool embeddings TRAINABLE, transfer-pool embeddings FROZEN and TIED to
the unembedding (logits = emb @ emb.T). Phase 4b weight-decay exclusion:
ALL name rows (emb_names) are in an explicit parameter group with
weight_decay=0.0, so frozen rows are bit-exact constant across training.
The gradient mask (names_grad_mask) opens fit-pool rows and leaves
transfer-pool rows zeroed.

Architecture constants:
"""
N_LAYERS = 8
D_MODEL = 128
N_HEADS = 4
D_FF = 512
MAX_LEN = 64

# ============================================================================
# 6. INDUCTION PRETRAINING (identical to Phase 5, with larger vocab)
# ============================================================================
"""
Phase 5 Step 2 induction pretraining on the v2 corpus (unique prefix of
L ∈ [16, 32] over the FULL task vocabulary — PAD excluded, all other tokens
eligible — tiled cyclically to the 64-token window, variable repeat offset).

Hyperparameters (verbatim from Phase 5):
  - Batch 256, AdamW lr 1e-3, wd 0.01 (name rows wd=0.0 group)
  - Cosine schedule over hard 50k steps with 500 warmup
  - Full-sequence LM loss, eval every 500, fp32
  - NO task episodes — pure induction corpus
  - Fresh batch each step (no pool/refresh)

Verification thresholds (pre-declared, identical to Phase 5):
  - Behavioural: copy accuracy >= 0.9 on held-out sequences
  - Mechanistic: some head >= 0.25 mass AND >= 5× uniform baseline
    on the prev-token-successor edge

IMPORTANT: With VOCAB ≈ 3154 (vs old ~1045), the induction corpus has
~3× more tokens. The copy-accuracy threshold of 0.9 should still be
achievable with the same architecture at 8 layers (more capacity than
the Phase 5 4L models). If seeds fail to meet thresholds after 50k,
replace them — report the failure rate.

Seed induction pretraining key: stage2/trackB/armR/induction/pretrain/seed{s}
"""

# Induction pretraining constants (from induction.py)
IND_LEN = 64
L_MIN, L_MAX = 16, 32
BEHAV_THRESH = 0.9
EDGE_ABS, EDGE_REL = 0.25, 5.0

IND_STEPS = 50_000  # hard, no early stop
IND_BATCH = 256
IND_LR = 1e-3
IND_WD = 0.01
IND_WARMUP = 500
IND_EVAL_EVERY = 500

# ============================================================================
# 7. TASK FINETUNING
# ============================================================================
"""
Load the induction-pretrained checkpoint, then finetune on the Track B
task distribution (k mixed {3, 4}, generators only in training, Q_3H
excluded from training). Optimizer starts FRESH (pretrained initialisation
is the single variable, per Phase 5 convention).

Hyperparameters:
  - Batch 256, AdamW lr 1e-3, wd 0.01 (name rows wd=0.0 group)
  - Cosine schedule over hard 50k steps with 500 warmup
  - Full-sequence LM loss, eval every 500, fp32
  - Pool size 8192, pool refresh every 2500 steps (on)
  - Train-accuracy logging on (separate eval rng stream)
  - Episode-local candidate masking (chance = 1/k)

Fit-pool embeddings are TRAINABLE during finetuning (the intervention):
  names_grad_mask[:N_FIT_NAMES] = 1.0  (fit-pool rows unfrozen)
  names_grad_mask[N_FIT_NAMES:] = 0.0  (transfer-pool rows frozen)

Pre-training verifications (mandated, same as Item 3):
  1. Tied-unembedding round-trip for TRANSFER pool
  2. Gradient flow: fit-pool rows get nonzero gradient, transfer rows get zero
  3. Freeze audit every eval: transfer-pool rows bit-exact vs init

Per-Step-4 induction head survival tracking (mandated):
  - Copy accuracy + per-head mass on held-out induction sequences
  - Logged every eval alongside task metrics

Finetuning key: stage2/trackB/armR/finetune/seed{s}
"""

TASK_STEPS = 50_000  # hard, no early stop
TASK_BATCH = 256
TASK_LR = 1e-3
TASK_WD = 0.01
TASK_WARMUP = 500
TASK_EVAL_EVERY = 500
POOL_SIZE = 8192
POOL_REFRESH = 2500

# ============================================================================
# 8. SEEDS AND FORMATION STATISTICS
# ============================================================================
"""
Minimum 10 seeds per arm (formation stochasticity ~25% demands it).

Seeds: 0, 1, 2, ..., 9 (minimum; more if budget allows)

Formation rates reported with binomial CIs per:
  - Single-hop aux accuracy (pooled across query types)
  - Composed 2-hop accuracy (Q_P + Q_G)
  - 3-hop generalisation accuracy (Q_3H, test-only)
  - Full-group vs generators accuracy gap
  - Transfer vocabulary generalisation
  - Orbit consistency

Convergence gate (per seed, all must pass to enter discriminator):
  - Composed held-out accuracy >= 0.95
  - Transfer-vocab accuracy >= 0.95
  - Abstract-role orbit consistency >= 0.95

Per-query-type breakdown is part of EVERY training run (blocking requirement).
"""

MIN_SEEDS = 10
SEEDS = list(range(MIN_SEEDS))

# ============================================================================
# 9. IMPLEMENTATION PLAN (sequential phases)
# ============================================================================
"""
Phase A: Write phase10/trackB/data_trackB.py
  - Implement the extended data module: BaseEpisode, render, render_batch,
    sample_base, build_eval_orbits, orbit_metrics for k ∈ {3, 4}, large
    vocab, 3-hop query, train-generators vs full-group support.
  - Include dump_episodes() and audit_episodes() for verification.
  - All constants (vocab sizes, token IDs, PERMS3/4, generators) defined
    identically to this file.

Phase B: Write phase10/trackB/train_trackB.py
  - Implement training: induction pretraining + task finetuning, shared
    between both arms.
  - Induction: load TinyTransformer(n_layers=8), pretrain on induction
    corpus v2 (same logic as phase5_pretrain.py but with 8L and arm-specific
    vocab), verify both thresholds, write verification to volume.
  - Task finetune: load pretrained checkpoint, set names_grad_mask for
    fit-pool rows, verify gradient flow + tied round-trip, train on
    arm-specific data distribution, log full eval battery every 500 steps,
    track induction head survival.
  - Expose arm-specific parameters (vocab sizes, name pool, compositions)
    via a config dict so the same training code serves both arms.

Phase C: Run Arm R (this file's design)
  - 10+ seeds, induction pretrain → task finetune → discriminator

Phase D: Run Arm C (armC_design.py)
  - 10+ seeds, induction pretrain → task finetune → discriminator

Phase E: Discriminator (adapted from Phase 8C/9, d=128)
  - Run full condition battery on every converged seed of both arms
  - Pre-registered predictions recorded in armR_design.py / armC_design.py
  - Verdict: instrument-validated / fails-middle-rung / indistinguishable

BUDGET: $250 (cap). At $2.50/h for A10G, 20+ seeds × ~3h/seed
(induction + finetune, 100k steps total × ~36 steps/s) ≈ 60h ≈ $150.
Discriminator (A100, ~$5/h, ~2h per seed × ~20) ≈ $200. Total ≈ $350
estimated — may require seed count reduction or the $250 cap is a soft
guideline with approval at $300.
"""

# ============================================================================
# 10. PRE-REGISTERED PREDICTIONS
# ============================================================================
"""
Arm R predictions (role-rewarding training distribution):

  PR1 (formation): Arm R converged seeds should achieve >= 0.95 composed
      held-out accuracy. If Arm R fails to converge at all despite the
      training distribution making role abstraction necessary, role structure
      may not be learnable at this scale — report with formation evidence.

  PR2 (3-hop generalisation): Arm R converged seeds should show 3-hop
      accuracy ABOVE CHANCE (>> 1/k). Learning the abstract role structure
      means the model composes HAS→GUARD→CARRY from its component relations
      without 3-hop training. If 3-hop stays at chance despite 2-hop being
      solved, the model learned 2-hop composition without abstracting the
      role structure — a finding in itself.

  PR3 (full-group generalisation): Converged seeds should show
      full-group accuracy ≈ generator accuracy (the gap ≤ 0.05). The
      generators span S_k (g12 and g23 generate all of S3 and S4), and a
      model that has learned the group action should generalise to products.

  PR4 (transfer generalisation): Converged seeds should show transfer-vocab
      accuracy >= 0.95. The role structure is independent of specific name
      tokens; a model relying on per-name memorisation would fail here.

  PR5 (discriminator): Arm R should show role-operator structure (C1 + C7 +
      group laws) at a higher rate than Arm C.

  PR6 (induction head): Induction head survival rate should be correlated
      with task convergence — seeds where the induction head is destroyed
      during finetuning should perform worse on composed queries.

All predictions are committed to this file before any training runs.
SHA256 of this file should be recorded in the pre-registration manifest.
"""

import numpy as np  # noqa: E402 (for type hints in docstrings)

# ============================================================================
# Token name mapping for dump/debug
# ============================================================================
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
    Q_3H: "Q_3H",
    A_PS: "A_PS",
    A_SN: "A_SN",
    A_NS: "A_NS",
    A_SG: "A_SG",
}


def token_name(t):
    """Human-readable token name."""
    if t in _SPECIAL:
        return _SPECIAL[t]
    if PROP0 <= t < SYM0:
        return f"p{t - PROP0}"
    if SYM0 <= t < NAME0:
        return f"s{t - SYM0}"
    return f"n{t - NAME0}"


# ============================================================================
# Verification: dump episodes to confirm clause order + query direction
# ============================================================================
def dump_episodes(n=20, seed=7):
    """Dump raw episodes human-readably and verify clause-order correction.

    Every emitted clause must be key-before-target:
      HAS p s / CARRY s n / GUARD a b

    Query direction: Q_P queries by property, Q_G by symbol, Q_3H by property.
    """
    rng = np.random.default_rng(seed)
    violations = 0

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
        Q_3H: is_prop,
        A_PS: is_prop,
        A_SN: is_sym,
        A_NS: is_name,
        A_SG: is_sym,
    }

    for i in range(n):
        b = sample_base(rng)
        g = perms_for(b.k)[int(rng.integers(len(perms_for(b.k))))]
        toks, apos, ans, _ = render(b, g)
        seq = [t for t in toks.tolist() if t != PAD]
        probs = []
        j = 1
        while j < len(seq) and seq[j] in clause_shape:
            rel, key, target = seq[j], seq[j + 1], seq[j + 2]
            k_ok, t_ok = clause_shape[rel]
            if not (k_ok(key) and t_ok(target) and seq[j + 3] == SEP):
                probs.append(
                    f"clause at {j} not key-before-target: "
                    f"{token_name(rel)} {token_name(key)} {token_name(target)}"
                )
            j += 4
        if j < len(seq):
            qtok, arg = seq[j], seq[j + 1]
            if qtok in q_arg_shape and not q_arg_shape[qtok](arg):
                probs.append(
                    f"query {token_name(qtok)} arg wrong type: {token_name(arg)}"
                )
        status = "OK " if not probs else "VIOLATION "
        print(f"[{i:02d}] {status}" + " ".join(token_name(t) for t in seq))
        for p in probs:
            print(f"     !! {p}")
        violations += len(probs)

    print(
        f"\nArm R: {violations} clause-order/direction violations in {n} episodes"
        + (" — key-before-target confirmed" if not violations else "")
    )
    return violations


# ============================================================================
# Self-verification when run directly
# ============================================================================
if __name__ == "__main__":
    print("=== Track B Arm R Design Verification ===\n")
    print(f"VOCAB: {VOCAB}")
    print(
        f"  Properties: {N_PROPS} fit + {N_PROPS_TRANSFER} transfer "
        f"(start: {PROP0}, transfer start: {PROP0_TRANSFER})"
    )
    print(
        f"  Symbols:    {N_SYMS} fit + {N_SYMS_TRANSFER} transfer "
        f"(start: {SYM0}, transfer start: {SYM0_TRANSFER})"
    )
    print(
        f"  Names:      {N_FIT_NAMES} fit + {N_TRANSFER_NAMES} transfer "
        f"(start: {NAME0})"
    )
    print(f"  SEQ_LEN:    {SEQ_LEN}")
    print(
        f"  Architecture: {N_LAYERS}L d_model={D_MODEL} "
        f"n_heads={N_HEADS} d_ff={D_FF}"
    )
    print("  k-values:   training {3, 4} mixed equally")
    print(f"  Training perms: generators only (k=3: {GEN3}, k=4: {GEN4})")
    print(f"  Test perms:     full group (k=3: {len(PERMS3)}, k=4: {len(PERMS4)})")
    print(
        f"  Train queries:  {[token_name(q) for q in COMPOSED_TRAIN]} "
        f"+ {[token_name(q) for q in AUX]}"
    )
    print(f"  Test-only:     {[token_name(q) for q in COMPOSED_TEST]}")
    print(f"  Induction:     hard {IND_STEPS} steps")
    print(f"  Finetune:      hard {TASK_STEPS} steps")
    print(f"  Seeds (min):   {MIN_SEEDS}")
    print("  Budget cap:    $250\n")

    print("--- Episode dump (sanity check) ---")
    dump_episodes(n=10, seed=42)

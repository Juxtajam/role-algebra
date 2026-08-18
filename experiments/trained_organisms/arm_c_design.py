"""
=== Track B Arm C (RETRIEVAL-SUFFICIENT CONTROL) — Design Specification ===
Phase 10 organism programme.

THIS IS A DESIGN DOCUMENT. No training, no GPU — the deliverable is the
complete specification for a subsequent implementation pass.

DESIGN PRINCIPLE: Arm C is the matched control arm. EVERYTHING is identical
to Arm R (architecture, steps, data volume, eval battery, induction
pretraining) EXCEPT the training vocabulary — Arm C uses a small fixed
vocabulary of 24 names with per-episode rebinding. With only 24 names, the
model sees every name many times across episodes. Retrieval/memorisation
SUFFICES — the model can memorise name→role pairings directly, so role
abstraction never pays. This isolates the training distribution's effect:
any difference between Arm R and Arm C is caused by whether role structure
is necessary, not by any confounded variable.

PREDICTION (pre-registered): Arm C converged seeds should show the
identity/near-identity signature the 72B showed — role-operator structure
(C1 + C7 + group laws) at a lower rate than Arm R seeds. If the
discriminator cannot distinguish arms whose training distributions differ
exactly in whether role structure pays, that is the instrument's failed
middle rung.

================================================================================
1. VOCABULARY — small fixed name pool, per-episode rebinding
================================================================================

Arm C shares the same property and symbol pools as Arm R (256 each) to keep
the comparison clean — the only difference is the name vocabulary.

  Pool          Size (fit)   Transfer   Total    Rationale
  ────────────  ───────────  ─────────  ───────  ───────────────────────────
  Properties     256           64        320     Same as Arm R
  Symbols        256           64        320     Same as Arm R
  Names           24          200        224     Small: can memorise all 24

With only 24 fit-pool names, the model sees each name approximately
(k × batch_size × 50k) / 24 ≈ (3.5 avg × 256 × 50000) / 24 ≈ 1.87M times
during training — far more than enough to memorise each name's embedding.

PER-EPISODE REBINDING: each episode randomly draws k names from the 24-name
pool and assigns them to the k roles. The same name n_3 could be role 0 in
one episode and role 2 in another. This FORCES the model to rely on
memorisation of the per-episode assignment rather than abstract role
structure.

Token layout (identical to Arm R, but with 24 fit names):

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
  526..749         Name tokens      (NAME0=526, N_FIT=24, N_TRANSFER=200)

VOCAB = 750.
"""

# ============================================================================
# Token constants (identical to Arm R except name pool sizes)
# ============================================================================
PAD, BOS, SEP, QMARK = 0, 1, 2, 3
CARRY, HAS, GUARD = 4, 5, 6

# --- 2-hop composed queries (training) ---
Q_P, Q_G = 7, 8

# --- 3-hop composed query (test only) ---
Q_3H = 9

# --- Single-hop auxiliaries ---
A_PS, A_SN, A_NS, A_SG = 10, 11, 12, 13

# --- Property pool (same as Arm R) ---
PROP0 = 14
N_PROPS = 256
N_PROPS_TRANSFER = 64
PROP0_TRANSFER = PROP0 + N_PROPS

# --- Symbol pool (same as Arm R) ---
SYM0 = PROP0 + N_PROPS + N_PROPS_TRANSFER  # = 334
N_SYMS = 256
N_SYMS_TRANSFER = 64
SYM0_TRANSFER = SYM0 + N_SYMS

# --- Name pool (THE DIFFERENCE: 24 fit names instead of 2000) ---
NAME0 = SYM0 + N_SYMS + N_SYMS_TRANSFER  # = 654
N_FIT_NAMES = 24  # SMALL: can fully memorise
N_TRANSFER_NAMES = 200
N_NAMES = N_FIT_NAMES + N_TRANSFER_NAMES  # = 224

VOCAB = NAME0 + N_NAMES  # = 878

# Sequence length (same as Arm R)
SEQ_LEN = 64

# ============================================================================
# 2. COMPOSITION STRUCTURE — identical to Arm R
# ============================================================================
"""
IDENTICAL to Arm R:
  - k mixed {3, 4} within training (equal probability)
  - Train: 2-hop composed (Q_P, Q_G), ~40% single-hop aux mix
  - Test: 3-hop composed (Q_3H), test-only, unseen during training
  - Train generators only, test full group
  - Key-before-target clause order
  - Episode-local candidate masking
  - Guard facts always present

THE ONLY DIFFERENCE: with 24 names, the model can memorise each name's
embedding and track per-episode assignments directly. Memorisation is
computationally cheaper than learning abstract role structure. Arm C tests
whether the discriminator correctly identifies that models trained on
this distribution DON'T learn role operators.
"""

# k=3 constants (same as Arm R)
K3 = 3
PERMS3 = [
    (0, 1, 2),  # identity
    (1, 0, 2),  # g12 — generator
    (0, 2, 1),  # g23 — generator
    (2, 1, 0),
    (1, 2, 0),
    (2, 0, 1),
]
GEN3 = [(1, 0, 2), (0, 2, 1)]

# k=4 constants (same as Arm R)
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
GEN4 = [(1, 0, 2, 3), (0, 2, 3, 1)]


def perms_for(k):
    return {3: PERMS3, 4: PERMS4}[k]


def generators_for(k):
    return {3: GEN3, 4: GEN4}[k]


# Query type groupings (same as Arm R)
COMPOSED_TRAIN = (Q_P, Q_G)
COMPOSED_TEST = (Q_3H,)
AUX = (A_PS, A_SN, A_NS, A_SG)


# ============================================================================
# 3. DATA GENERATION — small fixed pool, per-episode rebinding
# ============================================================================
"""
Episode structure: identical to Arm R's BaseEpisode class. The only
difference is in how name tokens are drawn:

  Arm R:  names drawn from 2000-name fit pool (NAME0..NAME0+1999)
  Arm C:  names drawn from   24-name fit pool (NAME0..NAME0+23)

With only 24 names, per-episode rebinding means: each episode draws k names
randomly from the 24-name pool without replacement, then assigns them to the
k roles via the permutation g. Every name is seen many times across episodes,
each time in potentially different roles. This is the retrieval-sufficient
condition — the model doesn't need to abstract the role structure because it
can simply memorise the per-token identity across episodes.

The key control: Arm C has the SAME number of episodes, SAME data volume,
SAME architecture, SAME steps as Arm R. The only variable is whether the
training vocabulary forces role abstraction (Arm R: 2000 names, never the
same name pair across episodes) or allows memorisation (Arm C: 24 names,
every name repeated thousands of times).

Implementation note: The BaseEpisode class from Arm R's data module is
reusable here — only the name_pool passed to sample_base() differs.
"""


class BaseEpisode:
    """Identical to Arm R's BaseEpisode.

    Included here for self-contained specification. The implementation
    should share code with Arm R (both use the same data_trackB.py module,
    differing only in vocab parameters).
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
        self.props = rng.choice(prop_pool, k, replace=False)
        self.syms = rng.choice(symbol_pool, k, replace=False)
        self.names = rng.choice(name_pool, k, replace=False)
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

    def facts(self, g):
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
        i = self.qi
        names = [NAME0 + int(self.names[j]) for j in range(self.k)]
        syms = [SYM0 + int(self.syms[j]) for j in range(self.k)]
        props = [PROP0 + int(self.props[j]) for j in range(self.k)]

        if self.qtok == Q_P:
            return props[i], names[g[i]], names
        if self.qtok == Q_G:
            return syms[i], names[g[self.sigma[i]]], names
        if self.qtok == Q_3H:
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
    """Render episode to token sequence.

    Identical to Arm R render() — the difference is only in vocabulary
    constants (N_FIT_NAMES = 24 here vs 2000 in Arm R), which affects
    which token IDs appear in NAME0 + names[j].
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
    pad_needed = SEQ_LEN - len(toks)
    if pad_needed < 0:
        raise ValueError(
            f"Episode too long for SEQ_LEN={SEQ_LEN}: {len(toks)} tokens "
            f"(k={base.k})"
        )
    toks.extend([PAD] * pad_needed)
    return (np.array(toks, dtype=np.int64), answer_pos, answer, cands)


def render_batch(bases, gs):
    """Batch render — identical to Arm R."""
    toks, apos, ans, cands = zip(*[render(b, g) for b, g in zip(bases, gs)])
    max_k = max(b.k for b in bases)
    padded_cands = np.zeros((len(cands), max_k), dtype=np.int64)
    for i, c in enumerate(cands):
        padded_cands[i, : len(c)] = c
    return (np.stack(toks), np.array(apos), np.array(ans), padded_cands)


def sample_base(rng, vocab="fit", force_qtok=None, force_k=None):
    """Sample a random BaseEpisode.

    The Arm C difference: name_pool is only 24 tokens for fit vocab.
    Everything else (k mixing, aux ratio, query types, guard facts) is
    identical to Arm R.

    With 24 fit names, each name appears in approximately:
      (k_avg × total_episodes × episodes_per_name_fraction) / 24
      ≈ (3.5 × 50000 × 256) / 24 ≈ 1.87M distinct episode participations
    Far more than enough to memorise per-name statistics.
    """
    if force_k is not None:
        k = force_k
    else:
        k = K3 if rng.random() < 0.5 else K4

    if vocab == "fit":
        name_pool = np.arange(N_FIT_NAMES)  # 0..23 — 24 names
        sym_pool = np.arange(N_SYMS)
        prop_pool = np.arange(N_PROPS)
    else:
        name_pool = np.arange(N_FIT_NAMES, N_FIT_NAMES + N_TRANSFER_NAMES)
        sym_pool = np.arange(N_SYMS, N_SYMS + N_SYMS_TRANSFER)
        prop_pool = np.arange(N_PROPS, N_PROPS + N_PROPS_TRANSFER)

    if force_qtok is not None:
        qtok = force_qtok
    elif rng.random() < 0.4:
        aux_list = list(AUX)
        qtok = aux_list[int(rng.integers(len(aux_list)))]
    else:
        qtok = Q_P if rng.random() < 0.5 else Q_G

    guard_facts = True

    return BaseEpisode(rng, k, name_pool, sym_pool, prop_pool, qtok, guard_facts)


def build_eval_orbits(vocab, n_bases, seed, qtok=None, force_k=None, perm_mode="full"):
    """Build fixed held-out orbit evaluation set.

    Identical to Arm R — the only difference is vocab constants affecting
    which token IDs appear.
    """
    rng = np.random.default_rng(seed)
    perm_fn = perms_for if perm_mode == "full" else generators_for

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

    if qtok in (None, Q_P, Q_G, Q_3H, A_SN):
        mode = "permute"
    else:
        mode = "invariant"

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
# 4. EVALUATION BATTERY — identical to Arm R
# ============================================================================
"""
Every evaluation point runs the FULL battery (same as Arm R):

  (a) Held-out fit-vocab composed accuracy (2-hop Q_P + Q_G)
      - n_bases=192, full group orbits, strict-orbit + consistency

  (b) Held-out transfer-vocab composed accuracy
      - n_bases=96, full group orbits, threshold >= 0.95

  (c) Held-out fit-vocab single-hop aux accuracy (per query type)
      - A_PS, A_SN, A_NS, A_SG, n_bases=64 per type

  (d) Train-distribution accuracy (composed queries only)
      - 96 bases from current training pool, separate eval rng

  (e) 3-hop generalisation (Q_3H, test-only)
      - n_bases=192, full group orbits
      - For Arm C: if the model memorised per-name assignments, 3-hop
        should FAIL (the memorised 2-hop pathway doesn't compose into 3-hop
        without the abstract role structure)

  (f) Per-k breakdown (k=3, k=4 separately)

  (g) Unseen permutation products
      - Generator accuracy vs full-group accuracy
      - For Arm C: if the model uses per-name memorisation, it may still
        generalise to product permutations (the memorised name→slot mapping
        is permutation-agnostic) — this is a DISCRIMINATIVE test

  (h) Induction head survival (Step 4, per eval)
      - Copy accuracy + per-head mass on held-out induction sequences

ARM C SPECIFIC PREDICTION (pre-registered):
  - Composed 2-hop held-out accuracy should reach >= 0.95 (retrieval works)
  - Transfer-vocab accuracy should reach >= 0.95 (name identity is learned)
  - 3-hop generalisation should be AT CHANCE (~1/k) — the model has no need
    to learn abstract role composition, and memorised name→slot mappings
    don't compose across three relations
  - Full-group accuracy should be close to generator accuracy (per-name
    memorisation is permutation-agnostic for the name→answer mapping, but
    the symbol-level relations still require the group structure for the
    intermediate hops — this is a key measurement)
  - Per-seed confidence intervals reported alongside Arm R

Convergence gate (per seed, all must pass to enter discriminator):
  - Composed held-out accuracy >= 0.95
  - Single-hop aux accuracy >= 0.95
  - Transfer-vocab accuracy >= 0.95
  - Orbit consistency >= 0.95
"""


# ============================================================================
# 5. ARCHITECTURE — identical to Arm R (and Phase 4b/5)
# ============================================================================
"""
TinyTransformer at 8 layers, d_model=128, 4 heads, d_ff=512, pre-LN, fp32,
learned absolute positional embeddings, max seq len=64.

Fit-pool embeddings TRAINABLE, transfer-pool embeddings FROZEN and TIED to
the unembedding. Phase 4b weight-decay exclusion: ALL name rows (emb_names)
in explicit parameter group with weight_decay=0.0.

Same as Arm R — the architecture and training setup are matched variables.
"""
N_LAYERS = 8
D_MODEL = 128
N_HEADS = 4
D_FF = 512
MAX_LEN = 64

# ============================================================================
# 6. INDUCTION PRETRAINING — identical to Arm R
# ============================================================================
"""
Phase 5 Step 2 induction pretraining, identical protocol:

  - Corpus v2: unique prefix L ∈ [16, 32] over full VOCAB (PAD excluded),
    tiled cyclically to 64, variable repeat offset
  - Batch 256, AdamW lr 1e-3, wd 0.01 (name rows wd=0.0 group)
  - Cosine schedule, hard 50k steps, 500 warmup
  - Full-sequence LM loss, eval every 500, fp32
  - Verification: copy acc >= 0.9, head mass >= 0.25 and >= 5× baseline

With VOCAB ≈ 878 (vs Arm R's 3154), the induction corpus is smaller, which
makes the copy task EASIER (fewer distractors). This is an advantage for
Arm C — if anything, Arm C induction should succeed at a higher rate than
Arm R. Document this asymmetry.

Seed induction pretraining key: stage2/trackB/armC/induction/pretrain/seed{s}
"""
IND_STEPS = 50_000
IND_BATCH = 256
IND_LR = 1e-3
IND_WD = 0.01
IND_WARMUP = 500
IND_EVAL_EVERY = 500

BEHAV_THRESH = 0.9
EDGE_ABS, EDGE_REL = 0.25, 5.0

# ============================================================================
# 7. TASK FINETUNING — identical to Arm R
# ============================================================================
"""
Load induction-pretrained checkpoint, finetune on Arm C task distribution.
Optimizer FRESH. Fit-pool name rows (0..23 of emb_names) TRAINABLE.

The finetuning protocol is identical to Arm R in every respect:
  - Same batch size, learning rate, weight decay, schedule, warmup
  - Same pool size, refresh interval, eval frequency
  - Same pre-training verifications (tied round-trip, gradient flow, freeze)
  - Same per-eval induction head survival tracking
  - Same candidate masking, same clause order, same aux mixture ratio

The ONLY difference is the name pool: 24 names instead of 2000. The model
sees each name thousands of times and can memorise per-name associations,
so the training loss should drop faster than Arm R and potentially converge
sooner (though the hard-50k schedule means it runs the full duration either
way for matched comparison).

Finetuning key: stage2/trackB/armC/finetune/seed{s}
"""
TASK_STEPS = 50_000
TASK_BATCH = 256
TASK_LR = 1e-3
TASK_WD = 0.01
TASK_WARMUP = 500
TASK_EVAL_EVERY = 500
POOL_SIZE = 8192
POOL_REFRESH = 2500

# ============================================================================
# 8. SEEDS AND FORMATION STATISTICS — identical to Arm R
# ============================================================================
"""
Minimum 10 seeds per arm. Same seed range as Arm R (0..9) for direct
comparison — same random initialisations, different training distributions.

Formation rates with binomial CIs per:
  - Single-hop aux accuracy
  - Composed 2-hop accuracy
  - 3-hop generalisation accuracy (expected: AT CHANCE for Arm C)
  - Full-group vs generators accuracy gap
  - Transfer vocabulary generalisation
  - Orbit consistency
"""
MIN_SEEDS = 10
SEEDS = list(range(MIN_SEEDS))

# ============================================================================
# 9. IMPLEMENTATION — shared code with Arm R
# ============================================================================
"""
Arm C shares the implementation with Arm R:

  - Same data_trackB.py module (selects vocab sizes via config dict)
  - Same train_trackB.py module (selects arm via config dict)
  - Same induction pretraining logic (different vocab size = different
    embedding table, but same code path)
  - Same eval battery code

  The config dict for Arm C:
    {
      "arm": "C",
      "n_fit_names": 24,
      "n_transfer_names": 200,
      "n_props": 256,
      "n_syms": 256,
      "n_props_transfer": 64,
      "n_syms_transfer": 64,
      "n_layers": 8,
      "d_model": 128,
      "n_heads": 4,
      "d_ff": 512,
      "max_steps_ind": 50000,
      "max_steps_task": 50000,
      "pool_size": 8192,
      "pool_refresh": 2500,
      "batch": 256,
      "lr": 0.001,
      "wd": 0.01,
      "warmup": 500,
      "k_values": [3, 4],
      "seed_base": 0,
      "n_seeds": 10,
      "results_key": "stage2/trackB/armC",
    }
"""

ARM_C_CONFIG = {
    "arm": "C",
    "description": "retrieval-sufficient control",
    "n_fit_names": 24,
    "n_transfer_names": 200,
    "n_props": 256,
    "n_syms": 256,
    "n_props_transfer": 64,
    "n_syms_transfer": 64,
    "n_layers": 8,
    "d_model": 128,
    "n_heads": 4,
    "d_ff": 512,
    "max_steps_ind": 50_000,
    "max_steps_task": 50_000,
    "pool_size": 8192,
    "pool_refresh": 2500,
    "batch": 256,
    "lr": 0.001,
    "wd": 0.01,
    "warmup": 500,
    "k_values": [3, 4],
    "seed_base": 0,
    "n_seeds": 10,
    "results_key": "stage2/trackB/armC",
}

# ============================================================================
# 10. PRE-REGISTERED PREDICTIONS
# ============================================================================
"""
Arm C predictions (retrieval-sufficient control):

  PR1 (2-hop convergence): Arm C should converge FASTER than Arm R on 2-hop
      composed queries. With only 24 names, memorisation is easy — the model
      can learn per-episode name→slot mappings without abstracting the role
      structure.

  PR2 (transfer): Transfer-vocab accuracy should be >= 0.95. The model has
      learned to read per-episode name assignments (the memorised embedding
      for each name token carries the episode-specific role via the fact
      clauses) — this generalises to unseen names because the mechanism is
      query-driven (look up the name in the relevant fact), not name-specific
      memorisation. Expected: high transfer despite small fit pool.

  PR3 (3-hop failure): 3-hop (Q_3H) accuracy should be AT CHANCE (~1/k).
      The model has learned to solve 2-hop by direct lookup (memorising
      per-episode HAS and CARRY mappings), not by composing abstract
      relations. Without explicit 3-hop training, the model cannot compose
      HAS→GUARD→CARRY because it never learned the relation operators as
      composable functions — it learned them as per-episode lookup tables.

  PR4 (full-group): Full-group accuracy should be close to generator
      accuracy (gap ≤ 0.05). Per-name memorisation is permutation-agnostic
      for the name→answer mapping step, though the intermediate symbol-level
      lookups still need the group structure. The key measurement is whether
      the intermediate lookups use role operators (Arm R pattern) or per-ep
      lookup tables (Arm C pattern) — the full-group generalisation pattern
      may not distinguish them cleanly.

  PR5 (discriminator): Arm C should show the identity/near-identity signature
      the 72B showed — role-operator structure (C1 + C7 + group laws) at a
      LOWER rate than Arm R. The discriminator should distinguish arms whose
      training distributions differ exactly in whether role structure pays.

  PR6 (induction head): Induction head should survive finetuning in Arm C
      (the task is easy enough that task-specific circuits don't need to
      compete with the induction head). If the induction head is destroyed
      in Arm C but survives in Arm R, that's evidence that Arm C's
      memorisation strategy conflicts with the induction head while Arm R's
      relational strategy is compatible.

NOT FAILURE: If Arm C forms role-operator structure at rates comparable to
Arm R, the training distribution manipulation FAILED to create the intended
difference. This is the "arms indistinguishable" outcome — reported as such,
and it bounds the 72B negative's interpretation.

All predictions committed BEFORE any training. SHA256 of this file recorded
in the pre-registration manifest.
"""

import numpy as np  # noqa: E402

# ============================================================================
# Token name mapping
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
    if t in _SPECIAL:
        return _SPECIAL[t]
    if PROP0 <= t < SYM0:
        return f"p{t - PROP0}"
    if SYM0 <= t < NAME0:
        return f"s{t - SYM0}"
    return f"n{t - NAME0}"


def dump_episodes(n=20, seed=7):
    """Verify clause-order and query direction on Arm C episodes."""
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

    # Track name reuse across episodes — should be HIGH with 24 names
    seen_names = set()

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

        # Count name reuse
        episode_names = set()
        for t in seq:
            if t >= NAME0:
                episode_names.add(t)
        seen_names.update(episode_names)

    print(
        f"\nArm C: {violations} clause-order/direction violations in {n} episodes"
        + (" — key-before-target confirmed" if not violations else "")
    )
    print(
        f"  Name reuse: {len(seen_names)} unique name tokens seen across "
        f"{n} episodes (max possible: {N_FIT_NAMES}) — "
        f"{'memorisation-sufficient' if len(seen_names) <= N_FIT_NAMES else 'UNEXPECTED'}"
    )

    return violations


# ============================================================================
# 11. COMPARISON TABLE — Arm R vs Arm C
# ============================================================================
"""
                 Arm R (role-rewarding)        Arm C (retrieval control)
  ────────────   ────────────────────────────  ────────────────────────────
  Fit names      2000                          24
  Transfer       500                           200
  Properties     256 + 64 transfer             256 + 64 transfer
  Symbols        256 + 64 transfer             256 + 64 transfer
  VOCAB          3154                          878
  k              3, 4 mixed                    same
  Train queries  2-hop + aux                   same
  Test queries   2-hop + 3-hop + aux           same
  Train perms    generators only               same
  Test perms     full group                    same
  Architecture   8L d=128 4h d_ff=512          same
  Fit-pool       trainable                     same
  Transfer-pool  frozen, wd=0.0                same
  Induction      50k hard                      same
  Finetune       50k hard                      same
  Seeds          10+                           same

  Name pairs     ~12.8M episodes / 2000 names  ~12.8M / 24 names
  per episode    ≈ 6400×/name (rare reuse)     ≈ 533K×/name (very frequent)
  Reuse pattern  Novel each episode            ~every name every few episodes
  Memorisation   NOT possible                  THE LOW-COST PATH
  Role pressure  HIGH — only way to generalise  LOW — memorise and retrieve
  Expected C1+C7 HIGH rate                     LOW rate (near identity)
"""


# ============================================================================
# Self-verification when run directly
# ============================================================================
if __name__ == "__main__":
    print("=== Track B Arm C Design Verification ===\n")
    print(f"VOCAB: {VOCAB}")
    print(f"  Properties: {N_PROPS} fit + {N_PROPS_TRANSFER} transfer")
    print(f"  Symbols:    {N_SYMS} fit + {N_SYMS_TRANSFER} transfer")
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
    print("  Budget cap:    $250")
    print(
        f"\n  FIT POOL SIZE: {N_FIT_NAMES} names "
        f"({'MEMORISATION SUFFICIENT' if N_FIT_NAMES <= 24 else 'ROLE PRESSURE'})"
    )
    print(
        f"  Name reuse:    ~{50000 * 256 * 3.5 / N_FIT_NAMES:.0f}x per name "
        f"over training"
    )
    print()

    print("--- Episode dump (sanity check) ---")
    dump_episodes(n=10, seed=42)

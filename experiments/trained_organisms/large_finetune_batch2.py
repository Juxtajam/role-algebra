"""Track B paired-arm fine-tune — Phase 10.
Loads the induction-pretrained checkpoint, initializes the arm-specific
data generator, runs 50k fine-tuning steps with full Phase 5 logging,
writes output per spec.

Usage:
    modal run --detach phase10/trackB/finetune_modal.py
"""

import modal
import pathlib

app = modal.App("dv3-trackB-finetune-batch2")
vol = modal.Volume.from_name("dv3-results")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.4.1", "numpy", "scipy", "pandas", "matplotlib")
    .add_local_dir(
        str(pathlib.Path(__file__).resolve().parents[2] / "src"),
        remote_path="/root/dv3",
    )
    .add_local_dir(
        str(pathlib.Path(__file__).resolve().parents[2] / "phase10"),
        remote_path="/root/phase10",
    )
)

MAX_STEPS = 50_000
MIN_STEPS_FLOOR = 25_000
BATCH = 256
EVAL_EVERY = 500
POOL_SIZE = 8192
POOL_REFRESH = 2500
WARMUP = 500


@app.function(image=image, gpu="A10G", timeout=60 * 60 * 12, volumes={"/results": vol})
def finetune(seed: int, arm: str) -> dict:
    import math, os, sys, threading, time, traceback

    os.environ["DV3_RESULTS"] = "/results"
    sys.path.insert(0, "/root/dv3")
    sys.path.insert(0, "/root/phase10")
    os.chdir("/root/dv3")

    stop_evt = threading.Event()

    def committer():
        while not stop_evt.is_set():
            time.sleep(60)
            try:
                vol.commit()
            except Exception:
                pass

    threading.Thread(target=committer, daemon=True).start()

    import numpy as np
    import torch
    from shared import progress
    from shared.progress import log
    from trained import induction as I
    from trained.model import (
        TinyTransformer,
        lm_loss,
        make_optimizer,
        masked_answer_preds,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- Task output key ----------------------------------------------------
    key = f"phase10/trackB/finetune/seed{seed}/{arm}"
    ckpt_path = progress.results_dir() / f"{key}/ckpt.pt"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    # ---- Token layout (adapted from arm designs to fit model VOCAB=1045) ----
    # Identical to old data.py except Q_3H steals A_PS's old slot (9) and
    # all aux shift right by 1; one fewer property (15 instead of 16) keeps
    # NAME0=45 unchanged so the state dict shapes are identical.
    PAD, BOS, SEP, QMARK = 0, 1, 2, 3
    CARRY, HAS, GUARD = 4, 5, 6
    Q_P, Q_G = 7, 8
    Q_3H = 9  # 3-hop composed (test-only; was A_PS in old data.py)
    A_PS, A_SN, A_NS, A_SG = 10, 11, 12, 13
    PROP0 = 14  # shifted from 13 to make room for Q_3H at 9
    N_PROPS = 15  # reduced from 16
    SYM0 = 29  # = PROP0 + N_PROPS = 14 + 15 = 29 (unchanged)
    N_SYMS = 16
    NAME0 = 45  # = SYM0 + N_SYMS = 29 + 16 = 45 (unchanged)
    N_NAMES = 1000
    VOCAB = 1045
    SEQ_LEN = 64  # enough for k=4 (53 tokens max)

    # ---- Arm-specific pool sizes --------------------------------------------
    # Arm R: 800 fit names (role-rewarding — large pool forces abstraction)
    # Arm C:  24 fit names (retrieval-sufficient — small pool allows memorisation)
    N_TRANSFER = 200
    if arm == "R":
        N_FIT_NAMES = 800
    elif arm == "C":
        N_FIT_NAMES = 24
    else:
        raise ValueError(f"Unknown arm: {arm}")

    # ---- k = {3, 4} support (from arm design specs) -------------------------
    K3, K4 = 3, 4
    PERMS3 = [
        (0, 1, 2),
        (1, 0, 2),
        (0, 2, 1),
        (2, 1, 0),
        (1, 2, 0),
        (2, 0, 1),
    ]
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
    GEN3 = [(1, 0, 2), (0, 2, 1)]
    GEN4 = [(1, 0, 2, 3), (0, 2, 3, 1)]

    def perms_for(k):
        return {K3: PERMS3, K4: PERMS4}[k]

    def generators_for(k):
        return {K3: GEN3, K4: GEN4}[k]

    # ---- Arm-specific name pool wrappers ------------------------------------
    # Arm R uses NAME0..NAME0+N_FIT_NAMES-1; Arm C uses the same range but
    # with N_FIT_NAMES=24. Transfer uses the top N_TRANSFER name rows.
    def name_pool_fit():
        return np.arange(N_FIT_NAMES)

    def name_pool_transfer():
        return np.arange(N_FIT_NAMES, N_FIT_NAMES + N_TRANSFER)

    # ---- Data generation (ported from arm designs, model-compatible) --------
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

        def facts(self, g):
            fs = []
            for i in range(self.k):
                fs.append((HAS, PROP0 + int(self.props[i]), SYM0 + int(self.syms[i])))
            for i in range(self.k):
                fs.append(
                    (CARRY, SYM0 + int(self.syms[i]), NAME0 + int(self.names[g[i]]))
                )
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

    def render(base, g):
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
        toks, apos, ans, cands = zip(*[render(b, g) for b, g in zip(bases, gs)])
        max_k = max(b.k for b in bases)
        padded_cands = np.zeros((len(cands), max_k), dtype=np.int64)
        for i, c in enumerate(cands):
            padded_cands[i, : len(c)] = c
        return (np.stack(toks), np.array(apos), np.array(ans), padded_cands)

    def sample_base(rng, vocab="fit", force_qtok=None, force_k=None, generators=False):
        """Sample a random BaseEpisode (training: generators only)."""
        if force_k is not None:
            k = force_k
        else:
            k = K3 if rng.random() < 0.5 else K4

        n_pool = name_pool_fit() if vocab == "fit" else name_pool_transfer()
        s_pool = np.arange(N_SYMS)
        p_pool = np.arange(N_PROPS)

        if force_qtok is not None:
            qtok = force_qtok
        elif rng.random() < 0.4:
            # ~40% single-hop auxiliaries
            aux_list = [A_PS, A_SN, A_NS, A_SG]
            qtok = aux_list[int(rng.integers(len(aux_list)))]
        else:
            # ~60% composed (Q_P or Q_G — 2-hop; Q_3H is NEVER in training)
            qtok = Q_P if rng.random() < 0.5 else Q_G

        guard_facts = True
        return BaseEpisode(rng, k, n_pool, s_pool, p_pool, qtok, guard_facts)

    def build_eval_orbits(
        vocab, n_bases, seed, qtok=None, force_k=None, perm_mode="full"
    ):
        """Build fixed held-out orbit evaluation set."""
        rng = np.random.default_rng(seed)
        perm_fn = perms_for if perm_mode == "full" else generators_for

        bases = []
        actual_k = force_k if force_k is not None else K3  # uniform k for eval
        for _ in range(n_bases):
            b = sample_base(rng, vocab=vocab, force_qtok=qtok, force_k=actual_k)
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

    def orbit_metrics(pred_tokens, ev):
        """Episode accuracy, strict-orbit accuracy, orbit consistency."""
        n = ev["n_bases"]
        n_perm = len(ev["tokens"]) // n
        correct = (pred_tokens == ev["answers"]).reshape(n, n_perm)
        max_k = ev["candidates"].shape[1]
        pred_slot = (
            ev["candidates"].reshape(n, n_perm, max_k)
            == pred_tokens.reshape(n, n_perm, 1)
        ).argmax(axis=2)
        cons = np.zeros((n, n_perm), dtype=bool)
        if ev.get("mode", "permute") == "invariant":
            # Symbol answers don't move under g: consistency = same pred across orbit
            cons = pred_slot == pred_slot[:, :1]
        else:
            # Name-answer queries permute with g
            # Determine which permutation list was used for this eval set
            if ev.get("perm_mode", "full") == "generators":
                perm_fn = generators_for
            else:
                perm_fn = perms_for
            # k_values may differ per base; use first k (eval sets use uniform k)
            k = int(ev["k_values"][0])
            p_list = perm_fn(k)
            for pi, g in enumerate(p_list):
                g_arr = np.array(g)
                cons[:, pi] = pred_slot[:, pi] == g_arr[pred_slot[:, 0]]
        return dict(
            episode_acc=float(correct.mean()),
            strict_orbit_acc=float(correct.all(axis=1).mean()),
            orbit_consistency=float(cons.all(axis=1).mean()),
            n_bases=n,
            n_perm=n_perm,
        )

    # ---- LR schedule (verbatim from Phase 5) --------------------------------
    def lr_at(step):
        if step < WARMUP:
            return 1e-3 * (step + 1) / WARMUP
        t = (step - WARMUP) / max(1, MAX_STEPS - WARMUP)
        return 1e-3 * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * min(t, 1.0))))

    # ---- Load pretrained checkpoint -----------------------------------------
    pre_ck_rel = f"phase10/trackB/induction/pretrain/seed{seed}/ckpt.pt"
    assert progress.exists(
        pre_ck_rel
    ), f"no pretrain checkpoint for seed{seed} at {pre_ck_rel}"
    log(f"TrackB finetune {key}: loading pretrained checkpoint from {pre_ck_rel}")

    # Pretrain verification gate (from the induction pretrain step)
    ver_rel = f"phase10/trackB/induction/pretrain/seed{seed}/pretrain_verify.json"
    if progress.exists(ver_rel):
        pre_ver = progress.load_json(ver_rel)
        assert pre_ver["pass"], (
            f"seed{seed} FAILED Step 2 verification "
            f"(behavioural={pre_ver['behavioural_pass']}, "
            f"mechanistic={pre_ver['mechanistic_pass']}) — does not advance"
        )
        pre_head = tuple(pre_ver["max_head"])
        log(
            f"TrackB finetune {key}: Step 2 gate OK — pretrained head "
            f"L{pre_head[0]} h{pre_head[1]} "
            f"(copy_acc {pre_ver['copy_acc']:.3f}, "
            f"mass {pre_ver['max_mass']:.3f} = {pre_ver['x_baseline']:.1f}x baseline)"
        )
    else:
        pre_head = None
        log(
            f"TrackB finetune {key}: no pretrain_verify.json — "
            f"proceeding without pre-head tracking"
        )

    torch.manual_seed(seed)
    model = TinyTransformer(seed=seed, n_layers=8).to(device)
    pre_ck = torch.load(progress.results_dir() / pre_ck_rel, map_location=device)
    assert (
        pre_ck["step"] == MAX_STEPS
    ), f"pretrain ckpt at step {pre_ck['step']}, expected {MAX_STEPS}"
    model.load_state_dict(pre_ck["model"])
    log(
        f"TrackB finetune {key}: loaded pretrained checkpoint ({MAX_STEPS} steps); "
        f"optimizer FRESH (initialisation is the single variable)"
    )

    # ---- Fit-pool name rows: UNFREEZE (Item 3 intervention) -----------------
    model.names_grad_mask[:N_FIT_NAMES] = 1.0
    model.names_grad_mask[N_FIT_NAMES:] = 0.0
    log(
        f"TrackB finetune {key}: names_grad_mask — "
        f"{N_FIT_NAMES} fit-pool rows TRAINABLE, "
        f"{N_NAMES - N_FIT_NAMES} transfer rows FROZEN"
    )

    opt = make_optimizer(model, lr=1e-3, weight_decay=0.01)
    assert [pg["weight_decay"] for pg in opt.param_groups] == [
        0.01,
        0.0,
    ], f"wd groups wrong: {[pg['weight_decay'] for pg in opt.param_groups]}"

    frozen_init = model.emb_names.detach().clone()
    # Note: verify_tied_names() checks that name-row embeddings round-trip
    # through the tied unembedding. After induction pretraining, emb_main
    # rows were trained while emb_names stayed frozen, so the round-trip
    # through the full emb.T may drift. The phase5 finetune script does not
    # call verify_tied_names after loading. We log the round-trip rate for
    # diagnostic purposes but do not assert (the true freeze audit is
    # bit-exact comparison of emb_names vs frozen_init at every eval).
    with torch.no_grad():
        ids = torch.arange(
            model.emb.shape[0] - model.emb_names.shape[0],
            model.emb.shape[0],
            device=device,
        )[:1000]
        logits = model.emb[ids] @ model.emb.T
        rt_ok = (logits.argmax(dim=1) == ids).float().mean().item()
    log(
        f"TrackB finetune {key}: name round-trip rate {rt_ok:.3f} "
        f"(pretrained emb_main drift expected; freeze audit at evals is authoritative)"
    )

    # ---- Pre-training verifications -----------------------------------------
    # 1. Gradient flow check: fit rows get nonzero grad, transfer rows get zero
    test_toks = torch.randint(0, VOCAB, (4, SEQ_LEN), device=device)
    test_loss = lm_loss(model(test_toks), test_toks)
    opt.zero_grad()
    test_loss.backward()
    fit_grad = model.emb_names.grad[:N_FIT_NAMES].abs().sum().item()
    xfer_grad = model.emb_names.grad[N_FIT_NAMES:].abs().sum().item()
    opt.zero_grad()
    log(
        f"TrackB finetune {key}: gradient flow — "
        f"fit rows {fit_grad:.6f}, transfer rows {xfer_grad:.6f} "
        f"({'OK' if fit_grad > 0 and xfer_grad == 0 else 'FAIL'})"
    )
    assert fit_grad > 0 and xfer_grad == 0, "gradient flow check failed"

    # ---- Induction eval set (Step 4: per-eval head survival) ----------------
    ev_ind = I.build_eval_set(seed)
    ver0 = I.verify_induction(model, ev_ind, device)
    log(
        f"TrackB finetune {key}: step-0 induction re-verify — "
        f"copy_acc {ver0['copy_acc']:.3f}, "
        f"max mass {ver0['max_mass']:.3f} at "
        f"L{ver0['max_head'][0]} h{ver0['max_head'][1]} "
        f"(pass={ver0['pass']})"
    )
    assert ver0["pass"], "loaded checkpoint fails induction verification"

    # ---- Evaluation battery construction ------------------------------------
    # (a) Held-out fit-vocab composed (2-hop Q_P + Q_G): n=192, full orbits
    ev_comp_fit = build_eval_orbits("fit", 192, seed=9000 + seed)
    # (b) Held-out transfer-vocab composed: n=96, full orbits
    ev_comp_xfer = build_eval_orbits("transfer", 96, seed=9500 + seed)
    # (c) Single-hop aux, per type: n=64 each
    ev_aux = {
        "A_PS": build_eval_orbits("fit", 64, seed=9600 + 31 * A_PS + seed, qtok=A_PS),
        "A_SN": build_eval_orbits("fit", 64, seed=9600 + 31 * A_SN + seed, qtok=A_SN),
        "A_NS": build_eval_orbits("fit", 64, seed=9600 + 31 * A_NS + seed, qtok=A_NS),
        "A_SG": build_eval_orbits("fit", 64, seed=9600 + 31 * A_SG + seed, qtok=A_SG),
    }
    # (e) 3-hop generalisation (Q_3H, test-only): n=192, full orbits
    ev_3hop = build_eval_orbits("fit", 192, seed=9700 + seed, qtok=Q_3H)
    # (f) Per-k breakdown: n=64 per k
    ev_k3 = build_eval_orbits("fit", 64, seed=9800 + seed, force_k=K3)
    ev_k4 = build_eval_orbits("fit", 64, seed=9801 + seed, force_k=K4)
    # (g) Unseen permutation products: generators n=96, full n=96
    ev_gen = build_eval_orbits("fit", 96, seed=9850 + seed, perm_mode="generators")
    ev_full = build_eval_orbits("fit", 96, seed=9851 + seed, perm_mode="full")

    def eval_acc(ev):
        preds = masked_answer_preds(
            model, ev["tokens"], ev["answer_pos"], ev["candidates"]
        )
        return float((preds == ev["answers"]).mean())

    def eval_orbit(ev):
        preds = masked_answer_preds(
            model, ev["tokens"], ev["answer_pos"], ev["candidates"]
        )
        return orbit_metrics(preds, ev)

    def train_acc(pool, eval_rng, qtoks, n=96):
        bs = [b for b in pool if b.qtok in qtoks][:n]
        if not bs:
            return None
        gs = []
        for b in bs:
            fill = generators_for(b.k)  # eval uses generators for train-acc
            gs.append(fill[int(eval_rng.integers(len(fill)))])
        toks, apos, ans, cands = render_batch(bs, gs)
        preds = masked_answer_preds(model, toks, apos, cands)
        return float((preds == ans).mean())

    # ---- Training loop ------------------------------------------------------
    rng = np.random.default_rng((seed, 77))
    eval_rng = np.random.default_rng((seed, 88))
    pool = [sample_base(rng) for _ in range(POOL_SIZE)]

    step, history = 0, []
    ind_history = []  # per-eval induction survival
    if ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        step, history = ck["step"], ck.get("history", [])
        if "ind_history" in ck:
            ind_history = ck["ind_history"]
        log(f"TrackB finetune {key}: RESUMED at step {step}")

    head_destroyed_logged = False
    t0 = time.time()
    model.train()

    try:
        while step < MAX_STEPS:
            if step and step % POOL_REFRESH == 0:
                pool = [sample_base(rng) for _ in range(POOL_SIZE)]

            idx = rng.integers(0, len(pool), size=BATCH)
            # Training uses generators only (g12, g23)
            gs = []
            for i in idx:
                k = pool[i].k
                gens = generators_for(k)
                gs.append(gens[int(rng.integers(len(gens)))])

            toks, _, _, _ = render_batch([pool[i] for i in idx], gs)
            toks = torch.as_tensor(toks, device=device)
            for pg in opt.param_groups:
                pg["lr"] = lr_at(step)
            loss = lm_loss(model(toks), toks)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            step += 1

            if step % EVAL_EVERY == 0 or step == MAX_STEPS:
                model.eval()

                # ---- Task metrics -------------------------------------------
                # Train accuracy
                tr_aux = train_acc(pool, eval_rng, (A_PS, A_SN, A_NS, A_SG))
                tr_comp = train_acc(pool, eval_rng, (Q_P, Q_G))

                # Held-out composed
                m_comp_fit = eval_orbit(ev_comp_fit)
                m_comp_xfer = eval_orbit(ev_comp_xfer)

                # Aux breakdown
                aux_accs = {}
                for label, ev_a in ev_aux.items():
                    aux_accs[label] = eval_acc(ev_a)
                aux_min = min(aux_accs.values())

                # 3-hop
                m_3hop = eval_orbit(ev_3hop)

                # Per-k
                m_k3 = eval_orbit(ev_k3)
                m_k4 = eval_orbit(ev_k4)

                # Unseen permutations
                m_gen = eval_orbit(ev_gen)
                m_full = eval_orbit(ev_full)

                # ---- Induction head survival (Step 4) -----------------------
                ind_acc = I.copy_accuracy(model, ev_ind, device)
                ind_tab = I.induction_attention(model, ev_ind, device, limit=256)
                li, h = np.unravel_index(ind_tab.argmax(), ind_tab.shape)
                ind_max = float(ind_tab.max())
                pre_mass = float(ind_tab[pre_head]) if pre_head else None
                ind_alive = bool(
                    ind_max >= I.EDGE_ABS and ind_max >= I.EDGE_REL * I.UNIFORM_BASELINE
                )
                ind_behav = bool(ind_acc > I.BEHAV_THRESH)
                ind_rec = dict(
                    step=step,
                    copy_acc=float(ind_acc),
                    max_mass=ind_max,
                    max_head=[int(li), int(h)],
                    x_baseline=float(ind_max / I.UNIFORM_BASELINE),
                    pretrained_head_mass=pre_mass,
                    mechanistic=ind_alive,
                    behavioural=ind_behav,
                    head_table=[[float(v) for v in row] for row in ind_tab],
                )
                ind_history.append(ind_rec)

                model.train()

                # Freeze audit: transfer-pool rows must stay bit-exact (fit-pool rows
                # are meant to change — that's the intervention).
                with torch.no_grad():
                    frozen_ok = bool(
                        torch.equal(
                            model.emb_names[N_FIT_NAMES:].detach(),
                            frozen_init[N_FIT_NAMES:],
                        )
                    )
                if not frozen_ok:
                    log(
                        f"MILESTONE: TrackB finetune {key} FREEZE VIOLATION "
                        f"at step {step} — run invalid"
                    )
                    raise RuntimeError("freeze violation")

                rec = dict(
                    step=step,
                    loss=float(loss.item()),
                    train_aux=tr_aux,
                    train_composed=tr_comp,
                    # (a) held-out fit composed
                    held_composed_acc=m_comp_fit["episode_acc"],
                    held_composed_strict=m_comp_fit["strict_orbit_acc"],
                    held_composed_consistency=m_comp_fit["orbit_consistency"],
                    # (b) held-out transfer
                    transfer_acc=m_comp_xfer["episode_acc"],
                    transfer_strict=m_comp_xfer["strict_orbit_acc"],
                    transfer_consistency=m_comp_xfer["orbit_consistency"],
                    # (c) aux
                    aux_accs=aux_accs,
                    aux_min=aux_min,
                    # (e) 3-hop
                    threehop_acc=m_3hop["episode_acc"],
                    threehop_strict=m_3hop["strict_orbit_acc"],
                    threehop_consistency=m_3hop["orbit_consistency"],
                    # (f) per-k
                    k3_acc=m_k3["episode_acc"],
                    k3_strict=m_k3["strict_orbit_acc"],
                    k4_acc=m_k4["episode_acc"],
                    k4_strict=m_k4["strict_orbit_acc"],
                    # (g) unseen perms
                    gen_acc=m_gen["episode_acc"],
                    gen_strict=m_gen["strict_orbit_acc"],
                    full_acc=m_full["episode_acc"],
                    full_strict=m_full["strict_orbit_acc"],
                    perm_gap=m_gen["episode_acc"] - m_full["episode_acc"],
                    # induction
                    ind_copy_acc=ind_acc,
                    ind_max_mass=ind_max,
                    ind_max_head=[int(li), int(h)],
                    ind_pretrained_head_mass=pre_mass,
                    ind_alive=ind_alive,
                    ind_behavioural=ind_behav,
                    frozen_rows_bit_exact=frozen_ok,
                )
                history.append(rec)

                rate = step / max(time.time() - t0, 1e-9)
                log(
                    f"TrackB finetune {key}: step {step} loss {rec['loss']:.4f} "
                    f"train_aux {tr_aux if tr_aux is None else f'{tr_aux:.3f}'} "
                    f"train_comp {tr_comp if tr_comp is None else f'{tr_comp:.3f}'} "
                    f"held_comp {rec['held_composed_acc']:.3f} "
                    f"strict {rec['held_composed_strict']:.3f} "
                    f"cons {rec['held_composed_consistency']:.3f} "
                    f"xfer {rec['transfer_acc']:.3f} "
                    f"aux_min {aux_min:.3f} "
                    f"3hop {rec['threehop_acc']:.3f} "
                    f"k3 {rec['k3_acc']:.3f} k4 {rec['k4_acc']:.3f} "
                    f"gen {rec['gen_acc']:.3f} full {rec['full_acc']:.3f} "
                    f"gap {rec['perm_gap']:+.3f} "
                    f"| IND copy {ind_acc:.3f} mass {ind_max:.3f}@L{li}h{h} "
                    f"preHead {pre_mass if pre_mass else 'N/A'} "
                    f"alive={ind_alive} behav={ind_behav} "
                    f"| frozen={frozen_ok} ({rate:.1f} steps/s)"
                )

                # Branch-condition logging
                if aux_min >= 0.95:
                    log(
                        f"MILESTONE: TrackB finetune {key} BRANCH — "
                        f"aux_held_min {aux_min:.3f} >= 0.95 at step {step}"
                    )
                if not ind_alive and not head_destroyed_logged:
                    head_destroyed_logged = True
                    log(
                        f"MILESTONE: TrackB finetune {key} INDUCTION HEAD LOST "
                        f"at step {step} — max mass {ind_max:.3f} "
                        f"({ind_max / I.UNIFORM_BASELINE:.1f}x), "
                        f"pretrained-head mass {pre_mass}, "
                        f"copy_acc {ind_acc:.3f}"
                    )

                # ---- Checkpoint ---------------------------------------------
                torch.save(
                    dict(
                        model=model.state_dict(),
                        opt=opt.state_dict(),
                        step=step,
                        history=history,
                        ind_history=ind_history,
                    ),
                    ckpt_path,
                )
                progress.save_json(f"{key}/trajectory.json", history)
                progress.save_json(f"{key}/induction_survival.json", ind_history)

        # ========== FINAL VERIFICATION =======================================
        log(
            f"MILESTONE: TrackB finetune {key} finished {MAX_STEPS} steps; "
            f"running final verification battery..."
        )

        # Per-query-type breakdown
        query_breakdown = {}
        query_types = [
            ("A_PS", A_PS),
            ("A_SN", A_SN),
            ("A_NS", A_NS),
            ("A_SG", A_SG),
            ("Q_P", Q_P),
            ("Q_G", Q_G),
            ("Q_3H", Q_3H),
        ]
        for label, qt in query_types:
            ev = build_eval_orbits("fit", 96, seed=9900 + 31 * qt + seed, qtok=qt)
            preds = masked_answer_preds(
                model, ev["tokens"], ev["answer_pos"], ev["candidates"]
            )
            m = orbit_metrics(preds, ev)
            query_breakdown[label] = dict(
                accuracy=m["episode_acc"],
                strict_orbit=m["strict_orbit_acc"],
                orbit_consistency=m["orbit_consistency"],
            )
            log(
                f"TrackB breakdown {key}: {label:<6} "
                f"acc={m['episode_acc']:.3f} "
                f"strict={m['strict_orbit_acc']:.3f} "
                f"cons={m['orbit_consistency']:.3f}"
            )
        progress.save_json(f"{key}/query_breakdown.json", query_breakdown)

        # Final verification record
        final = history[-1] if history else {}
        verification = dict(
            seed=seed,
            arm=arm,
            steps=step,
            final=final,
            max_composed_held=max(h["held_composed_acc"] for h in history),
            max_composed_transfer=max(h["transfer_acc"] for h in history),
            max_aux_min=max(h["aux_min"] for h in history),
            max_threehop=max(h["threehop_acc"] for h in history),
            max_perm_gap=max(abs(h["perm_gap"]) for h in history),
            final_ind_alive=final.get("ind_alive"),
            final_ind_behav=final.get("ind_behavioural"),
            min_ind_max_mass=min(h["ind_max_mass"] for h in history),
            min_ind_copy_acc=min(h["ind_copy_acc"] for h in history),
            query_breakdown=query_breakdown,
        )
        progress.save_json(f"{key}/verification.json", verification)

        log(
            f"MILESTONE: TrackB finetune {key} DONE at {step} — "
            f"held_comp {final['held_composed_acc']:.3f} "
            f"(strict {final['held_composed_strict']:.3f}), "
            f"transfer {final['transfer_acc']:.3f}, "
            f"3hop {final['threehop_acc']:.3f}, "
            f"aux_min {final['aux_min']:.3f} | "
            f"induction copy {final['ind_copy_acc']:.3f} "
            f"mass {final['ind_max_mass']:.3f} "
            f"alive={final['ind_alive']}"
        )

        return verification

    except Exception:
        log(f"MILESTONE: TrackB finetune {key} CRASHED:\n" + traceback.format_exc())
        crash_ver = {
            "seed": seed,
            "arm": arm,
            "status": "CRASHED",
            "error": traceback.format_exc(),
        }
        progress.save_json(f"{key}/verification.json", crash_ver)
        return crash_ver
    finally:
        stop_evt.set()
        vol.commit()


@app.local_entrypoint()
def main():
    """Launch the first batch: 12 runs (seeds 0-5, both arms).

    Verified seeds from induction pretraining: 0,1,2,4,5,6,7,8,9,10,11,13
    (seeds 3 and 12 failed verification). Batch 1 = seeds 0,1,2,4,5,6.
    """
    log = print

    # First batch: 6 seeds × 2 arms = 12 runs
    seeds = [7, 8, 9, 10]
    arms = ["R", "C"]
    runs = [(s, a) for s in seeds for a in arms]

    log(f"\n{'='*60}")
    log(f"TrackB finetune BATCH 2: launching {len(runs)} runs")
    log(f"  Seeds: {seeds}")
    log(f"  Arms: {arms}")
    log(f"  Total: {len(runs)} runs (well within concurrency limit)")
    log(f"{'='*60}")

    results = list(finetune.starmap(runs))

    log(f"\nAll {len(results)} runs complete")
    for r in results:
        s, a = r.get("seed", "?"), r.get("arm", "?")
        status = r.get("status", "OK")
        log(f"  seed{s}/{a}: {status}")

    log("\nMILESTONE: TrackB finetune BATCH 2 COMPLETE")

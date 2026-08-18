"""Stage 2 training + the trained-system adapter for the discriminator.

Training: full-sequence LM loss, batch 256, AdamW lr 1e-3, cosine with
warmup, fp32. Evaluate every 500 steps; stop when held-out strict-orbit
accuracy has not improved by >0.002 over 10 consecutive evaluations, or at
max_steps. Checkpoints (model + optimizer + history) are written at every
evaluation so an interrupted Colab session resumes where it left off.

Convergence gate before an organism enters the discriminator:
  composed-query held-out accuracy >= 0.95 (except T0, whose shortcut is the
  point — T0 is evaluated on its own leaked distribution), transfer-vocabulary
  accuracy >= 0.95, abstract-role orbit consistency >= 0.95.
"""

import math
import time

import numpy as np
import torch

from shared import progress
from shared.progress import log
from trained import data as D
from trained.model import (
    N_LAYERS,
    TinyTransformer,
    lm_loss,
    make_optimizer,
    masked_answer_preds,
    verify_tied_names,
)

BATCH = 256
EVAL_EVERY = 500
PATIENCE = 10
MIN_DELTA = 0.002
# The patience criterion is satisfied automatically by any run whose
# strict-orbit accuracy sits at 0 (no improvement > MIN_DELTA is vacuously
# true), which stopped every failing run at the minimum possible step. No
# early stop may fire before this floor; the 50k cap is unchanged.
MIN_STEPS_FLOOR = 25_000
WARMUP = 500
POOL_SIZE = 8192
POOL_REFRESH = 2500


def _lr(step, max_steps, base=1e-3):
    if step < WARMUP:
        return base * (step + 1) / WARMUP
    t = (step - WARMUP) / max(1, max_steps - WARMUP)
    return base * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * min(t, 1.0))))


def _eval_model(model, ev):
    preds = masked_answer_preds(model, ev["tokens"], ev["answer_pos"], ev["candidates"])
    return D.orbit_metrics(preds, ev)


def train_organism(
    name,
    seed,
    max_steps=50_000,
    device="cpu",
    smoke=False,
    n_layers=4,
    run_key=None,
    hard_stop=False,
    weight_decay=0.01,
):
    """run_key overrides the results subdirectory (depth-sweep runs live under
    stage2/depth_sweep/...). hard_stop=True disables the patience criterion
    entirely: the run goes to max_steps, so non-convergence at max_steps is
    actually tested. weight_decay exposed per the authorised wd lever
    (default 0.01 = the value every prior run used)."""
    key = run_key or f"{name}/seed{seed}"
    ckpt_rel = f"stage2/{key}/ckpt.pt"
    ckpt_path = progress.results_dir() / ckpt_rel
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    batch = 64 if smoke else BATCH
    eval_every = max(50, min(EVAL_EVERY, max_steps // 5)) if smoke else EVAL_EVERY
    warmup = min(WARMUP, max_steps // 4)

    torch.manual_seed(seed)
    model = TinyTransformer(seed=seed, n_layers=n_layers).to(device)
    rt = verify_tied_names(model)
    log(f"stage2 {key}: tied-name round-trip verified ({rt:.3f})")
    opt = make_optimizer(model, lr=1e-3, weight_decay=weight_decay)
    log(
        f"stage2 {key}: weight_decay={weight_decay} (name-embedding rows excluded "
        f"from decay via explicit parameter group — Phase 4b freeze fix)"
    )

    step, history = 0, []
    if ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        step, history = ck["step"], ck["history"]
        log(f"stage2 {key}: RESUMED from checkpoint at step {step}")

    n_ev = 32 if smoke else 192
    ev_held = D.build_eval_orbits(name, "fit", n_ev, seed=9000 + seed)
    ev_transfer = D.build_eval_orbits(name, "transfer", n_ev // 2, seed=9500 + seed)
    rng = np.random.default_rng((seed, 77))
    eval_rng = np.random.default_rng(
        (seed, 88)
    )  # separate stream: train-acc eval must not perturb the training data order
    pool = [D.sample_base(name, rng) for _ in range(POOL_SIZE if not smoke else 512)]

    t0 = time.time()
    model.train()
    while step < max_steps:
        if step and step % POOL_REFRESH == 0:
            pool = [D.sample_base(name, rng) for _ in range(len(pool))]
        idx = rng.integers(0, len(pool), size=batch)
        gs = [D.PERMS[i] for i in rng.integers(0, len(D.PERMS), size=batch)]
        toks, _, _, _ = D.render_batch([pool[i] for i in idx], gs)
        toks = torch.as_tensor(toks, device=device)
        for pg in opt.param_groups:
            pg["lr"] = _lr(step, max_steps, base=1e-3)
        loss = lm_loss(model(toks), toks)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        step += 1

        if step % eval_every == 0 or step == max_steps:
            m_h = _eval_model(model, ev_held)
            m_t = _eval_model(model, ev_transfer)
            # train-distribution accuracy (mandated): composed queries rendered
            # from bases in the CURRENT training pool
            tr_pool = [b for b in pool if b.qtok in (D.Q_P, D.Q_G)][:96]
            tr_gs = [
                D.PERMS[i]
                for i in eval_rng.integers(0, len(D.PERMS), size=len(tr_pool))
            ]
            tr_toks, tr_apos, tr_ans, tr_cands = D.render_batch(tr_pool, tr_gs)
            tr_preds = masked_answer_preds(model, tr_toks, tr_apos, tr_cands)
            train_acc = float((tr_preds == tr_ans).mean())
            model.train()
            rec = dict(
                step=step,
                loss=float(loss.item()),
                train_composed=train_acc,
                held_episode=m_h["episode_acc"],
                held_strict=m_h["strict_orbit_acc"],
                held_consistency=m_h["orbit_consistency"],
                transfer_episode=m_t["episode_acc"],
            )
            history.append(rec)
            rate = step / max(time.time() - t0, 1e-9)
            log(
                f"stage2 {key}: step {step} loss {rec['loss']:.4f} "
                f"train {train_acc:.3f} "
                f"held acc {rec['held_episode']:.3f} strict {rec['held_strict']:.3f} "
                f"consistency {rec['held_consistency']:.3f} transfer {rec['transfer_episode']:.3f} "
                f"({rate:.1f} steps/s)"
            )
            torch.save(
                dict(
                    model=model.state_dict(),
                    opt=opt.state_dict(),
                    step=step,
                    history=history,
                ),
                ckpt_path,
            )
            progress.save_json(f"stage2/{key}/trajectory.json", history)
            # early stop: no strict-orbit improvement > MIN_DELTA across
            # PATIENCE consecutive evaluations — but never before the
            # MIN_STEPS_FLOOR (a chance-level run satisfies patience vacuously)
            if (
                len(history) > PATIENCE
                and step >= min(MIN_STEPS_FLOOR, max_steps)
                and not hard_stop
            ):
                recent = [h["held_strict"] for h in history[-(PATIENCE + 1) :]]
                if max(recent[1:]) - recent[0] <= MIN_DELTA and not smoke:
                    log(
                        f"stage2 {key}: early stop at step {step} "
                        f"(no strict-orbit improvement > {MIN_DELTA} over {PATIENCE} evals; "
                        f"floor {MIN_STEPS_FLOOR} respected)"
                    )
                    break

    final = history[-1] if history else {}
    acc_gate = 0.5 if smoke else 0.95
    gates = dict(
        composed_held=(
            final.get("held_episode", 0.0),
            name == "T0" or final.get("held_episode", 0) >= acc_gate,
        ),
        transfer=(
            final.get("transfer_episode", 0.0),
            final.get("transfer_episode", 0) >= acc_gate,
        ),
        consistency=(
            final.get("held_consistency", 0.0),
            final.get("held_consistency", 0) >= acc_gate,
        ),
    )
    converged = all(ok for _, ok in gates.values())
    ver = dict(
        organism=name,
        seed=seed,
        steps=step,
        converged=converged,
        enter_discriminator=converged,
        gates={k: dict(value=v, ok=ok) for k, (v, ok) in gates.items()},
        final=final,
        smoke=smoke,
        n_layers=n_layers,
        note="T0 composed gate uses its own leaked distribution (the shortcut is the point)",
    )
    progress.save_json(f"stage2/{key}/verification.json", ver)
    log(
        f"stage2 {key}: training done at step {step} — converged={converged} "
        + " ".join(
            f"{k}={v:.3f}({'ok' if ok else 'FAIL'})" for k, (v, ok) in gates.items()
        )
    )
    # per-query-type breakdown is part of EVERY training run (blocking
    # requirement): hop failure must be distinguishable from composition failure
    _query_breakdown_model(model, name, key, seed)
    return ver


# ---------------------------------------------------------------------------
# Per-query-type accuracy breakdown
# ---------------------------------------------------------------------------
QUERY_TYPES = [
    ("property->symbol", D.A_PS),
    ("symbol->person", D.A_SN),
    ("person->symbol", D.A_NS),
    ("symbol->guarded", D.A_SG),
    ("composed_P", D.Q_P),
    ("composed_G", D.Q_G),
]


def _query_breakdown_model(model, org, key, seed, n_bases=96):
    """Score each query type independently on held-out episodes; accuracy and
    orbit consistency per type. Written to stage2/<key>/query_breakdown.json."""
    out = {}
    for label, qt in QUERY_TYPES:
        if qt in (D.A_SG, D.Q_G) and "G" not in D.ORG_SPECS[org]["paths"]:
            continue
        ev = D.build_eval_orbits(
            org, "fit", n_bases, seed=9800 + 31 * qt + seed, qtok=qt
        )
        m = _eval_model(model, ev)
        out[label] = dict(
            accuracy=m["episode_acc"], orbit_consistency=m["orbit_consistency"]
        )
        log(
            f"breakdown {key}: {label:<18} acc={m['episode_acc']:.3f} "
            f"consistency={m['orbit_consistency']:.3f}"
        )
    progress.save_json(f"stage2/{key}/query_breakdown.json", out)
    return out


def query_breakdown(name, seed, device="cpu", run_key=None, n_layers=4):
    """Breakdown for an existing checkpoint (skips gracefully if absent)."""
    key = run_key or f"{name}/seed{seed}"
    ckpt = progress.results_dir() / f"stage2/{key}/ckpt.pt"
    if not ckpt.exists():
        log(f"breakdown {key}: no checkpoint — skipped")
        return None
    model = TinyTransformer(seed=seed, n_layers=n_layers).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device)["model"])
    model.eval()
    return _query_breakdown_model(model, name, key, seed)


# ---------------------------------------------------------------------------
# Discriminator adapter: exposes the same system interface as the synthetic
# organisms, over resid_post at the answer position (fixed in advance).
# ---------------------------------------------------------------------------
class TrainedSystem:
    N_BASES = 48

    def __init__(self, name, seed, device="cpu", n_layers=N_LAYERS):
        self.name, self.seed, self.device = name, seed, device
        self.k, self.n_layers = D.K, n_layers
        self.d = 128
        self.generators = [(1, 0, 2), (0, 2, 1)]
        self.all_perms = D.PERMS
        self.crosspath_available = "G" in D.ORG_SPECS[name]["paths"]
        # Stage 2 support test: u_i are name embeddings, so high (R - I) mass
        # in span{u_i - u_j} is the lexical-swap artifact -> direction 'lt'
        self.support_test = "lexical"
        self.model = TinyTransformer(seed=seed, n_layers=n_layers).to(device)
        sfx = "" if n_layers == 4 else f"_L{n_layers}"
        ck = torch.load(
            progress.results_dir() / f"stage2/{name}{sfx}/seed{seed}/ckpt.pt",
            map_location=device,
        )
        self.model.load_state_dict(ck["model"])
        self.model.eval()
        self._bases = {}
        self._eps = {}  # ep key -> (base, g)
        self._cache = {}  # ep key -> list of resid vectors per layer

    def bases(self, path, vocab, split):
        key = (path, vocab, split)
        if key not in self._bases:
            if path == "G" and not self.crosspath_available:
                return []
            import zlib

            seed = zlib.crc32(
                repr((self.name, self.seed) + key).encode()
            )  # stable across processes
            rng = np.random.default_rng(seed)
            bs = [
                D.sample_base(
                    self.name, rng, vocab=vocab, composed_only=True, path=path
                )
                for _ in range(self.N_BASES)
            ]
            ids = []
            for i, b in enumerate(bs):
                bid = key + (i,)
                self._eps.update({(bid, g): (b, g) for g in self.all_perms})
                ids.append(bid)
            self._bases[key] = ids
        return self._bases[key]

    def orbit(self, bid):
        return {g: (bid, g) for g in self.all_perms}

    def _episode(self, ep):
        return self._eps[ep]

    def answers(self, eps):
        return np.array([b.answer_slot(g) for b, g in map(self._episode, eps)])

    def content_labels(self, eps):
        return np.array(
            [
                (b.props[b.qi] if b.qtok == D.Q_P else b.syms[b.qi]) % 8
                for b, _ in map(self._episode, eps)
            ]
        )

    def u_vectors(self, eps, layer):
        E = self.model.emb.detach().cpu().numpy()
        return np.stack(
            [
                E[[D.NAME0 + b.names[j] for j in range(self.k)]]
                for b, _ in map(self._episode, eps)
            ]
        )

    def _render(self, eps):
        toks, apos, ans, cands = zip(
            *[D.render(b, g) for b, g in map(self._episode, eps)]
        )
        return np.stack(toks), np.array(apos), np.array(ans), np.array(cands)

    @torch.no_grad()
    def _ensure_cached(self, eps):
        missing = [ep for ep in eps if ep not in self._cache]
        for i in range(0, len(missing), 256):
            chunk = missing[i : i + 256]
            toks, apos, _, _ = self._render(chunk)
            t = torch.as_tensor(toks, device=self.device)
            _, resids = self.model(t, capture=True)
            rows = torch.arange(len(chunk), device=self.device)
            ap = torch.as_tensor(apos, device=self.device)
            for li in range(self.n_layers):
                layer_vecs = resids[li][rows, ap].cpu().numpy()
                for j, ep in enumerate(chunk):
                    self._cache.setdefault(ep, [None] * self.n_layers)[li] = layer_vecs[
                        j
                    ]

    def states(self, eps, layer):
        self._ensure_cached(eps)
        return np.stack([self._cache[ep][layer] for ep in eps])

    @torch.no_grad()
    def decode(self, eps):
        toks, apos, _, cands = self._render(eps)
        preds = masked_answer_preds(self.model, toks, apos, cands)
        return (cands == preds[:, None]).argmax(axis=1)

    @torch.no_grad()
    def decode_from(self, H, layer, eps):
        """Causal transport: replace resid_post[layer] at the answer position
        with H, run the remaining layers, decode with candidate masking."""
        preds = np.empty(len(eps), dtype=int)
        for i in range(0, len(eps), 256):
            chunk = eps[i : i + 256]
            toks, apos, _, cands = self._render(chunk)
            t = torch.as_tensor(toks, device=self.device)
            ap = torch.as_tensor(apos, device=self.device)
            hv = torch.as_tensor(
                H[i : i + 256], dtype=torch.float32, device=self.device
            )
            logits = self.model(t, patch=(layer, ap, hv))
            rows = torch.arange(len(chunk), device=self.device)
            sel = logits[rows, ap]
            cd = torch.as_tensor(cands, device=self.device)
            pred_tok = (
                cd.gather(1, sel.gather(1, cd).argmax(1, keepdim=True))
                .squeeze(1)
                .cpu()
                .numpy()
            )
            preds[i : i + 256] = (cands == pred_tok[:, None]).argmax(axis=1)
        return preds


def load_system(name, seed, device="cpu", n_layers=N_LAYERS):
    return TrainedSystem(name, seed, device=device, n_layers=n_layers)


def evaluate_unleaked(name="T0", seeds=(0, 1), device="cpu", n_bases=192):
    """Evaluate existing checkpoints on episodes WITHOUT the positional leak.
    Expected for T0: accuracy near 1/k, confirming the shortcut. If accuracy
    stays high, the organism learned the actual task despite the shortcut and
    the pressure gradient's bottom rung is not what the design assumes —
    reported prominently. (Transfer accuracy does not discriminate here: a
    positional circuit generalises to unseen names.)"""
    out = {}
    for seed in seeds:
        rel = f"stage2/{name}/seed{seed}/unleaked_eval.json"
        if progress.exists(rel):
            out[seed] = progress.load_json(rel)
            log(
                f"unleaked eval {name}/seed{seed}: recorded as FINAL "
                f"(acc={out[seed]['episode_acc']:.3f}) — not revisited"
            )
            continue
        ckpt = progress.results_dir() / f"stage2/{name}/seed{seed}/ckpt.pt"
        if not ckpt.exists():
            log(f"unleaked eval {name}/seed{seed}: no checkpoint — skipped")
            continue
        model = TinyTransformer(seed=seed).to(device)
        model.load_state_dict(torch.load(ckpt, map_location=device)["model"])
        model.eval()
        ev = D.build_eval_orbits(name, "fit", n_bases, seed=9700 + seed, leak=False)
        m = _eval_model(model, ev)
        out[seed] = m
        progress.save_json(f"stage2/{name}/seed{seed}/unleaked_eval.json", m)
        verdict = (
            "CONFIRMS positional shortcut (accuracy ~ chance without the leak)"
            if m["episode_acc"] < 0.6
            else "UNEXPECTED: high accuracy WITHOUT the leak — the organism learned the "
            "actual task; the pressure gradient's bottom rung is not what the design "
            "assumes. Report prominently."
        )
        log(
            f"unleaked eval {name}/seed{seed}: composed acc={m['episode_acc']:.3f} "
            f"orbit consistency={m['orbit_consistency']:.3f} -> {verdict}"
        )
    return out

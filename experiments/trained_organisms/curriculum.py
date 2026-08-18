"""Authorised lever 2 — curriculum: train on single-hop auxiliary queries ONLY
until held-out aux accuracy >= 0.95 (min across aux types), then introduce
composed queries (normal T1 mixture). T1, 4 layers, 3 seeds, hard 50k total
steps. Everything else fixed: batch 256, AdamW lr 1e-3 wd 0.01, cosine over
50k, eval every 500, candidate masking, full-sequence LM loss.

If phase 1 never reaches the gate within 50k, that IS the result (given
Check 2 showed aux held-out never left chance, this is a live possibility);
the run records where aux train/held accuracy plateaued.

Runs keyed stage2/curriculum/T1_cur/seed{s} on the volume.
"""

import modal
import pathlib

app = modal.App("dv3-curriculum")
vol = modal.Volume.from_name("dv3-results")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.4.1", "numpy", "scipy", "pandas", "matplotlib")
    .add_local_dir(
        str(pathlib.Path(__file__).resolve().parents[2] / "src"),
        remote_path="/root/dv3",
    )
)

MAX_STEPS = 50_000
GATE = 0.95


@app.function(image=image, gpu="A10G", timeout=60 * 60 * 8, volumes={"/results": vol})
def train_curriculum(seed: int):
    import math, os, sys, threading, time, traceback

    os.environ["DV3_RESULTS"] = "/results"
    sys.path.insert(0, "/root/dv3")
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
    from trained import data as D
    from trained.model import (
        TinyTransformer,
        lm_loss,
        masked_answer_preds,
        verify_tied_names,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    key = f"curriculum/T1_cur/seed{seed}"
    ckpt_path = progress.results_dir() / f"stage2/{key}/ckpt.pt"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    BATCH, EVAL_EVERY, POOL_SIZE, POOL_REFRESH, WARMUP = 256, 500, 8192, 2500, 500
    AUX_T1 = (D.A_PS, D.A_SN, D.A_NS)

    def lr_at(step):
        if step < WARMUP:
            return 1e-3 * (step + 1) / WARMUP
        t = (step - WARMUP) / max(1, MAX_STEPS - WARMUP)
        return 1e-3 * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * min(t, 1.0))))

    def sample_pool(rng, phase):
        if phase == 1:  # aux only, balanced across the three T1 aux types
            return [
                D.sample_base("T1", rng, force_qtok=AUX_T1[i % 3])
                for i in range(POOL_SIZE)
            ]
        return [D.sample_base("T1", rng) for _ in range(POOL_SIZE)]  # normal mixture

    torch.manual_seed(seed)
    model = TinyTransformer(seed=seed, n_layers=4).to(device)
    verify_tied_names(model)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    log(
        f"curriculum {key}: START phase 1 (aux-only) — device={device}, gate={GATE}, "
        f"wd=0.01, 4 layers, hard {MAX_STEPS}"
    )

    rng = np.random.default_rng((seed, 77))
    eval_rng = np.random.default_rng((seed, 88))

    # fixed held-out eval sets
    ev_aux = {
        qt: D.build_eval_orbits("T1", "fit", 64, seed=9600 + 31 * qt + seed, qtok=qt)
        for qt in AUX_T1
    }
    ev_comp = D.build_eval_orbits("T1", "fit", 192, seed=9000 + seed)
    ev_transfer = D.build_eval_orbits("T1", "transfer", 96, seed=9500 + seed)

    def acc(ev):
        preds = masked_answer_preds(
            model, ev["tokens"], ev["answer_pos"], ev["candidates"]
        )
        return float((preds == ev["answers"]).mean())

    def train_acc(pool, qtoks, n=96):
        bs = [b for b in pool if b.qtok in qtoks][:n]
        if not bs:
            return None
        gs = [D.PERMS[i] for i in eval_rng.integers(0, len(D.PERMS), size=len(bs))]
        toks, apos, ans, cands = D.render_batch(bs, gs)
        preds = masked_answer_preds(model, toks, apos, cands)
        return float((preds == ans).mean())

    phase, phase_switch_step = 1, None
    step, history = 0, []
    if ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        step, history, phase = ck["step"], ck["history"], ck["phase"]
        phase_switch_step = ck.get("phase_switch_step")
        log(f"curriculum {key}: RESUMED at step {step} (phase {phase})")

    pool = sample_pool(rng, phase)
    t0 = time.time()
    model.train()
    try:
        while step < MAX_STEPS:
            if step and step % POOL_REFRESH == 0:
                pool = sample_pool(rng, phase)
            idx = rng.integers(0, len(pool), size=BATCH)
            gs = [D.PERMS[i] for i in rng.integers(0, len(D.PERMS), size=BATCH)]
            toks, _, _, _ = D.render_batch([pool[i] for i in idx], gs)
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
                aux_held = {D.token_name(qt): acc(ev_aux[qt]) for qt in AUX_T1}
                aux_min = min(aux_held.values())
                comp_held = acc(ev_comp)
                tr_aux = train_acc(pool, AUX_T1)
                tr_comp = train_acc(pool, (D.Q_P,)) if phase == 2 else None
                model.train()
                rec = dict(
                    step=step,
                    phase=phase,
                    loss=float(loss.item()),
                    aux_held=aux_held,
                    aux_min=aux_min,
                    train_aux=tr_aux,
                    train_composed=tr_comp,
                    held_episode=comp_held,
                    transfer_episode=acc(ev_transfer),
                )
                history.append(rec)
                rate = step / max(time.time() - t0, 1e-9)
                log(
                    f"curriculum {key}: step {step} P{phase} loss {rec['loss']:.3f} "
                    f"aux_held_min {aux_min:.3f} ({aux_held}) train_aux "
                    f"{tr_aux if tr_aux is None else round(tr_aux, 3)} "
                    f"composed_held {comp_held:.3f} ({rate:.1f} steps/s)"
                )
                if phase == 1 and aux_min >= GATE:
                    phase = 2
                    phase_switch_step = step
                    pool = sample_pool(rng, phase)
                    log(
                        f"MILESTONE: curriculum {key} PHASE GATE PASSED at step {step} "
                        f"(aux held-out min {aux_min:.3f} >= {GATE}) — composed queries introduced"
                    )
                torch.save(
                    dict(
                        model=model.state_dict(),
                        opt=opt.state_dict(),
                        step=step,
                        history=history,
                        phase=phase,
                        phase_switch_step=phase_switch_step,
                    ),
                    ckpt_path,
                )
                progress.save_json(f"stage2/{key}/trajectory.json", history)

        final = history[-1]
        result = dict(
            seed=seed,
            phase_reached=phase,
            phase_switch_step=phase_switch_step,
            final=final,
        )
        progress.save_json(f"stage2/{key}/curriculum_result.json", result)
        if phase == 1:
            log(
                f"MILESTONE: curriculum seed{seed} — PHASE 1 NEVER PASSED the {GATE} gate in "
                f"{MAX_STEPS} steps (final aux held min {final['aux_min']:.3f}, "
                f"train_aux {final['train_aux']}) — aux queries do not become learnable even "
                f"with 100% of the gradient. That is the result."
            )
        else:
            log(
                f"MILESTONE: curriculum seed{seed} — gate passed at {phase_switch_step}; "
                f"final composed held {final['held_episode']:.3f}, transfer "
                f"{final['transfer_episode']:.3f}"
            )
        return result
    except Exception:
        log(f"MILESTONE: CURRICULUM seed{seed} CRASHED:\n" + traceback.format_exc())
        raise
    finally:
        stop_evt.set()
        vol.commit()


@app.function(image=image, timeout=1800, volumes={"/results": vol})
def finalize():
    import os, sys

    os.environ["DV3_RESULTS"] = "/results"
    sys.path.insert(0, "/root/dv3")
    os.chdir("/root/dv3")
    from shared import progress
    from shared.progress import log

    summary = {}
    for s in (0, 1, 2):
        rel = f"stage2/curriculum/T1_cur/seed{s}/curriculum_result.json"
        if progress.exists(rel):
            summary[f"seed{s}"] = progress.load_json(rel)
    progress.save_json("stage2/curriculum_summary.json", summary)
    log(
        "MILESTONE: CURRICULUM SUMMARY: "
        + "; ".join(
            f"seed{s}: phase={v['phase_reached']} switch={v['phase_switch_step']} "
            f"aux_min={v['final']['aux_min']:.3f} composed={v['final']['held_episode']:.3f}"
            for s, v in ((k[-1], v) for k, v in sorted(summary.items()))
        )
    )
    vol.commit()
    return summary


@app.local_entrypoint()
def main():
    results = list(train_curriculum.map([0, 1, 2]))
    print("curriculum:", results)
    print("finalize:", finalize.remote())

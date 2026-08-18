"""Track B induction pretraining — BATCH 2.
Seeds 6–10, both Arm R (VOCAB=3154) and Arm C (VOCAB=878) = 10 runs.

8-layer TinyTransformers (d_model=128) on the Phase 5 v2 induction corpus,
14 seeds, A10G GPUs. 50k steps per seed, full verification battery at end.

Usage:
    modal run --detach phase10/trackB/induction_pretrain_batch2.py
"""

import modal
import pathlib

app = modal.App("dv3-trackB-batch2")
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

# BATCH 2: seeds 6–10, both arms
SEEDS = [6, 7, 8, 9, 10]
ARMS = {
    "armR": 3154,  # Large pools — role-rewarding
    "armC": 878,  # Small name pool — retrieval-sufficient
}

# Pre-computed v2 corpus reference hashes (verified at script write time, 2026-08-09).
V2_INDUCTION_SOURCE_SHA256 = (
    "a7ba38451c4b31327b2957ae0285c4796912c4158523925d1c2a68ab5ae39edd"
)
V2_MODEL_SOURCE_SHA256 = (
    "e7927870fc5353a3f0888263e425c8ae42f12c851e1ff4afced758a8785f9079"
)
# Reference batch at seed 9999 using original VOCAB=1045 (deterministic numpy RNG).
V2_REFERENCE_BATCH_SHA256 = (
    "08622548408f78ec57ed93d3f6c7ee0cbd12e8d43b383e0f6466addc697eeb80"
)

# Per-arm reference batch hashes — recomputed with arm-specific VOCAB.
# These are placeholder values; MUST be recomputed before launch by running
# the reference-batch hash snippet with each arm's VOCAB.
ARM_REFERENCE_BATCH_SHA256 = {
    "armR": "PLACEHOLDER_RECOMPUTE_BEFORE_LAUNCH",
    "armC": "PLACEHOLDER_RECOMPUTE_BEFORE_LAUNCH",
}


@app.function(image=image, gpu="A10G", timeout=60 * 60 * 12, volumes={"/results": vol})
def pretrain(seed: int, arm: str) -> dict:
    import hashlib, math, os, sys, threading, time, traceback

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

    vocab = ARMS[arm]

    # ---------- Monkey-patch VOCAB before importing induction/model ----------
    import trained.data

    trained.data.VOCAB = vocab

    import numpy as np
    import torch
    from shared import progress
    from shared.progress import log
    from trained import induction as I
    from trained.model import (
        TinyTransformer,
        lm_loss,
        make_optimizer,
        verify_tied_names,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    key = f"phase10/trackB/induction/{arm}/pretrain/seed{seed}"
    ckpt_path = progress.results_dir() / f"{key}/ckpt.pt"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    BATCH, EVAL_EVERY, WARMUP = 256, 500, 500

    # ---------- corpus hash verification ----------
    log(
        f"TrackB batch2 {arm} seed{seed}: verifying v2 corpus hashes (VOCAB={vocab})..."
    )

    with open("/root/dv3/trained/induction.py", "rb") as f:
        induction_hash = hashlib.sha256(f.read()).hexdigest()
    assert (
        induction_hash == V2_INDUCTION_SOURCE_SHA256
    ), f"induction.py hash mismatch: {induction_hash}"

    with open("/root/dv3/trained/model.py", "rb") as f:
        model_hash = hashlib.sha256(f.read()).hexdigest()
    assert model_hash == V2_MODEL_SOURCE_SHA256, f"model.py hash mismatch: {model_hash}"

    # Check reference batch — uses arm-specific hash
    arm_ref_hash = ARM_REFERENCE_BATCH_SHA256[arm]
    if arm_ref_hash != "PLACEHOLDER_RECOMPUTE_BEFORE_LAUNCH":
        rng_check = np.random.default_rng(9999)
        seqs_ref, _ = I.sample_batch(rng_check, 256)
        ref_hash = hashlib.sha256(seqs_ref.tobytes()).hexdigest()
        assert (
            ref_hash == arm_ref_hash
        ), f"reference batch hash mismatch: {ref_hash} != {arm_ref_hash}"
        log(f"TrackB batch2 {arm} seed{seed}: corpus hash verification PASSED")
    else:
        log(
            f"TrackB batch2 {arm} seed{seed}: SKIPPING reference batch hash "
            f"(placeholder — recompute before launch)"
        )

    # ---------- LR schedule (verbatim from Phase 5) ----------
    def lr_at(step):
        if step < WARMUP:
            return 1e-3 * (step + 1) / WARMUP
        t = (step - WARMUP) / max(1, MAX_STEPS - WARMUP)
        return 1e-3 * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * min(t, 1.0))))

    # ---------- model init ----------
    torch.manual_seed(seed)
    model = TinyTransformer(seed=seed, n_layers=8).to(device)
    rt = verify_tied_names(model)
    log(
        f"TrackB batch2 {arm} seed{seed}: tied-name round-trip verified ({rt:.3f}), "
        f"VOCAB={vocab}, 8 layers, d_model=128, 4 heads, d_ff=512, fp32"
    )
    opt = make_optimizer(model, lr=1e-3, weight_decay=0.01)
    assert [pg["weight_decay"] for pg in opt.param_groups] == [
        0.01,
        0.0,
    ], f"wd groups wrong: {[pg['weight_decay'] for pg in opt.param_groups]}"
    log(
        f"TrackB batch2 {arm} seed{seed}: START — induction corpus v2, "
        f"batch {BATCH}, AdamW lr 1e-3 wd 0.01 (name rows wd=0.0, frozen), "
        f"cosine/{MAX_STEPS} warmup {WARMUP}, hard {MAX_STEPS}"
    )

    rng = np.random.default_rng((seed, 77))
    ev_seqs = I.build_eval_set(seed)
    frozen_init = model.emb_names.detach().clone()

    step, history = 0, []
    if ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        step, history = ck["step"], ck["history"]
        log(f"TrackB batch2 {arm} seed{seed}: RESUMED at step {step}")

    t0 = time.time()
    model.train()
    try:
        while step < MAX_STEPS:
            seqs, _ = I.sample_batch(rng, BATCH)
            toks = torch.as_tensor(seqs, device=device)
            for pg in opt.param_groups:
                pg["lr"] = lr_at(step)
            loss = lm_loss(model(toks), toks)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            step += 1

            if step % EVAL_EVERY == 0 or step == MAX_STEPS:
                acc = I.copy_accuracy(model, ev_seqs, device)
                tab = I.induction_attention(model, ev_seqs, device, limit=256)
                li, h = np.unravel_index(tab.argmax(), tab.shape)
                mx = float(tab.max())
                with torch.no_grad():
                    frozen_ok = bool(torch.equal(model.emb_names.detach(), frozen_init))
                model.train()
                rec = dict(
                    step=step,
                    loss=float(loss.item()),
                    copy_acc=acc,
                    max_ind_mass=mx,
                    max_ind_head=[int(li), int(h)],
                    x_baseline=float(mx / I.UNIFORM_BASELINE),
                    frozen_rows_bit_exact=frozen_ok,
                )
                history.append(rec)
                rate = step / max(time.time() - t0, 1e-9)
                log(
                    f"TrackB batch2 {arm} seed{seed}: step {step} loss {rec['loss']:.4f} "
                    f"copy_acc {acc:.3f} max_ind_mass {mx:.3f} "
                    f"(L{li} h{h}, {rec['x_baseline']:.1f}x baseline) "
                    f"frozen_exact={frozen_ok} ({rate:.1f} steps/s)"
                )
                if not frozen_ok:
                    log(
                        f"MILESTONE: TrackB batch2 {arm} seed{seed} FREEZE VIOLATION "
                        f"at step {step}"
                    )
                    raise RuntimeError("freeze violation")
                torch.save(
                    dict(
                        model=model.state_dict(),
                        opt=opt.state_dict(),
                        step=step,
                        history=history,
                    ),
                    ckpt_path,
                )
                progress.save_json(f"{key}/trajectory.json", history)

        # ========== FULL VERIFICATION BATTERY ==========
        log(
            f"MILESTONE: TrackB batch2 {arm} seed{seed} finished {MAX_STEPS} steps; "
            f"running full verification battery..."
        )

        ver = I.verify_induction(model, ev_seqs, device)

        # Off-period content probe
        log(
            f"TrackB batch2 {arm} seed{seed}: running off-period content probe "
            f"(periods 11, 13, 40)..."
        )
        off_period_masses = {}
        off_period_accs = {}
        for L_test in [11, 13, 40]:
            rng_op = np.random.default_rng((seed, 70000 + L_test))
            r = rng_op.random((1024, vocab - 1))
            prefixes = np.argsort(r, axis=1)[:, :L_test] + 1
            op_Ls = np.full(1024, L_test, dtype=np.int64)
            idx = np.arange(I.IND_LEN)[None, :] % op_Ls[:, None]
            op_seqs = np.take_along_axis(prefixes, idx, axis=1).astype(np.int64)
            op_ev = dict(
                seqs=op_seqs,
                Ls=op_Ls,
                probes=I.probe_index_arrays(op_Ls),
            )
            op_acc = I.copy_accuracy(model, op_ev, device, batch=256)
            op_tab = I.induction_attention(model, op_ev, device, batch=128, limit=None)
            op_li, op_h = np.unravel_index(op_tab.argmax(), op_tab.shape)
            op_mx = float(op_tab.max())
            off_period_masses[str(L_test)] = dict(
                max_mass=op_mx,
                max_head=[int(op_li), int(op_h)],
                x_baseline=float(op_mx / I.UNIFORM_BASELINE),
            )
            off_period_accs[str(L_test)] = float(op_acc)
            log(
                f"TrackB batch2 {arm} seed{seed}: off-period L={L_test} "
                f"copy_acc={op_acc:.3f} max_mass={op_mx:.3f} "
                f"at L{op_li} h{op_h} ({op_mx / I.UNIFORM_BASELINE:.1f}x baseline)"
            )

        ver.update(
            seed=seed,
            arm=arm,
            vocab=vocab,
            steps=step,
            final_loss=history[-1]["loss"],
            n_layers=8,
            off_period_masses=off_period_masses,
            off_period_accs=off_period_accs,
            off_period_warning=bool(
                all(
                    v["max_mass"] <= 3 * I.UNIFORM_BASELINE
                    for v in off_period_masses.values()
                )
            ),
        )
        progress.save_json(f"{key}/pretrain_verify.json", ver)

        adv = "ADVANCES" if ver["pass"] else "DOES NOT ADVANCE"
        status = "PASS" if ver["pass"] else "FAILED"
        log(
            f"MILESTONE: TrackB batch2 {arm} seed{seed} DONE at {step} — "
            f"copy_acc {ver['copy_acc']:.3f} "
            f"(behavioural {'PASS' if ver['behavioural_pass'] else 'FAIL'}), "
            f"max mass {ver['max_mass']:.3f} at L{ver['max_head'][0]} "
            f"h{ver['max_head'][1]} ({ver['x_baseline']:.1f}x baseline; "
            f"mechanistic {'PASS' if ver['mechanistic_pass'] else 'FAIL'}) "
            f"-> seed {adv}"
        )
        if not ver["pass"]:
            log(
                f"MILESTONE: TrackB batch2 {arm} seed{seed} {status} — checkpoint "
                f"saved for diagnosis at {key}/ckpt.pt"
            )
        return ver

    except Exception:
        log(
            f"MILESTONE: TrackB batch2 {arm} seed{seed} CRASHED:\n"
            + traceback.format_exc()
        )
        crash_ver = {
            "seed": seed,
            "arm": arm,
            "status": "CRASHED",
            "pass": False,
            "error": traceback.format_exc(),
            "behavioural_pass": False,
            "mechanistic_pass": False,
        }
        progress.save_json(f"{key}/pretrain_verify.json", crash_ver)
        return crash_ver
    finally:
        stop_evt.set()
        vol.commit()


@app.function(image=image, volumes={"/results": vol})
def finalize_summary(all_results_json: str):
    """Write summary.json to the volume (runs on Modal CPU — no GPU needed)."""
    import json
    from shared import progress

    all_results = json.loads(all_results_json)
    passing = [r for r in all_results if r.get("pass")]
    failing = [
        r for r in all_results if not r.get("pass") and r.get("status") != "CRASHED"
    ]
    crashed = [r for r in all_results if r.get("status") == "CRASHED"]

    summary = {
        "batch": 2,
        "track": "B",
        "phase": "induction_pretrain",
        "architecture": "8L_d128_fp32",
        "seeds_launched": SEEDS,
        "arms_launched": list(ARMS.keys()),
        "n_runs": len(SEEDS) * len(ARMS),
        "n_passing": len(passing),
        "n_failing": len(failing),
        "n_crashed": len(crashed),
        "passing_seeds": [r["seed"] for r in passing],
        "failing_seeds": [r["seed"] for r in failing],
        "crashed_seeds": [r["seed"] for r in crashed],
        "per_seed_details": {},
    }
    for r in all_results:
        s = r.get("seed", "?")
        a = r.get("arm", "?")
        summary["per_seed_details"][f"{s}/{a}"] = {
            "pass": r.get("pass", False),
            "copy_acc": r.get("copy_acc"),
            "max_mass": r.get("max_mass"),
            "max_head": r.get("max_head"),
            "x_baseline": r.get("x_baseline"),
            "off_period_masses": r.get("off_period_masses", {}),
        }

    path = progress.save_json("phase10/trackB/induction/batch2/summary.json", summary)
    vol.commit()
    print(f"\nMILESTONE: TrackB batch2 induction pretrain SUMMARY written to {path}")
    print(
        f"  {len(passing)}/{len(SEEDS) * len(ARMS)} passed, {len(failing)} failed, "
        f"{len(crashed)} crashed"
    )
    return summary


@app.local_entrypoint()
def main():
    import json

    log = print

    # Build seed-arm pairs: seeds 6-10 × both arms = 10 runs
    runs = [(s, a) for s in SEEDS for a in ARMS]

    log(f"\n{'='*60}")
    log(
        f"TrackB BATCH 2: launching {len(runs)} runs "
        f"(seeds {SEEDS}, arms {list(ARMS.keys())})"
    )
    log(f"{'='*60}")
    for s, a in runs:
        log(f"  seed={s}  arm={a}  VOCAB={ARMS[a]}")

    all_results = list(pretrain.starmap(runs))
    log(f"All {len(all_results)} runs complete")

    log("\n" + "=" * 60)
    log("Batch 2 complete — writing summary...")
    summary = finalize_summary.remote(json.dumps(all_results, default=str))
    log("\n" + json.dumps(summary, indent=2, default=str))

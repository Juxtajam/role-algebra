"""Modal runner for discriminator_validation_v3.

Runs the full notebook pipeline (Stage 1 synthetic suite -> Stage 2 trained
transformers) on a Modal GPU container. Results + checkpoints persist to a
Modal Volume ('dv3-results'), so the run is resumable exactly like the
notebook's Drive-backed design: re-running skips finished work and resumes
training from checkpoints.
"""

import threading
import time

import modal
import pathlib

app = modal.App("dv3-discriminator-validation")

vol = modal.Volume.from_name("dv3-results", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.4.1", "numpy", "scipy", "pandas", "matplotlib")
    .add_local_dir(
        str(pathlib.Path(__file__).resolve().parents[2] / "src"),
        remote_path="/root/dv3",
    )
)

SEEDS = [0, 1, 2]
MAX_STEPS = 50_000
NULL_REPS = 12


@app.function(image=image, gpu="A10G", timeout=60 * 60 * 20, volumes={"/results": vol})
def run_all():
    import os
    import sys
    import traceback

    os.environ["DV3_RESULTS"] = "/results"
    os.environ["MPLBACKEND"] = "Agg"
    sys.path.insert(0, "/root/dv3")
    os.chdir("/root/dv3")

    # commit the volume periodically so progress.log / checkpoints are
    # visible from outside and survive preemption
    stop = threading.Event()

    def committer():
        while not stop.is_set():
            time.sleep(60)
            try:
                vol.commit()
            except Exception:
                pass

    threading.Thread(target=committer, daemon=True).start()

    import torch
    from shared.progress import log

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(
        f"MODAL RUN START — device={device} ({torch.cuda.get_device_name(0) if device=='cuda' else 'cpu'})"
    )

    try:
        # ---- calibration (frozen) -------------------------------------
        from shared.calibrate import calibrate_stage1
        from shared.run import make_synth_system

        cfg1 = calibrate_stage1(make_synth_system, null_reps=NULL_REPS)
        log(f"MILESTONE: Stage 1 calibration frozen (lambda={cfg1['lambda']})")

        # ---- Stage 1 ---------------------------------------------------
        from shared.run import run_stage1, stage1_passed

        s1 = run_stage1()
        log(
            f"MILESTONE: Stage 1 {'PASSED' if s1['passed'] else 'FAILED'} — "
            f"verdicts={s1['verdicts']} floor={s1['snr_sensitivity_floor']}"
        )
        vol.commit()
        if not stage1_passed():
            log("MILESTONE: STAGE 1 FAILED — Stage 2 must not run (spec). Stopping.")
            return "stage1_failed"

        # ---- data audit -------------------------------------------------
        from trained.data import audit_episodes

        for org in ("T0", "T1", "T2", "T3"):
            failures = audit_episodes(org, n=2000)
            log(f"MILESTONE: data audit {org}: {len(failures)} failures")
            if failures:
                log(
                    f"MILESTONE: DATA FAULT in {org} — stopping before training: {failures[:5]}"
                )
                return "data_fault"

        # ---- Stage 2 training -------------------------------------------
        from shared.run import run_stage2_training

        verifications, t3_ok = run_stage2_training(SEEDS, MAX_STEPS, device=device)
        conv = {k: v.get("converged") for k, v in verifications.items()}
        log(f"MILESTONE: Stage 2 training done — converged={conv} t3_ok={t3_ok}")
        vol.commit()

        # ---- analyses (always run; cheap) ---------------------------------
        from shared.report import (
            transition_report,
            hop_vs_composition,
            plot_trajectories,
        )

        for org in ("T0", "T1", "T2", "T3"):
            transition_report(org, SEEDS, threshold=0.40)
        from trained.train import evaluate_unleaked, query_breakdown

        evaluate_unleaked("T0", seeds=SEEDS, device=device)
        for org in ("T1", "T2", "T3"):
            for s in SEEDS:
                query_breakdown(org, s, device=device)
        branch = hop_vs_composition(("T1", "T2", "T3"), SEEDS)
        log(f"MILESTONE: hop-vs-composition branch verdicts: {branch}")
        try:
            plot_trajectories(SEEDS)
        except Exception as e:
            log(f"plot_trajectories failed (non-fatal): {e}")

        # ---- conditional depth sweep --------------------------------------
        from shared.run import run_depth_sweep

        sweep = run_depth_sweep(SEEDS, device=device)
        if sweep is not None:
            log(f"MILESTONE: depth sweep summary: {sweep}")
        vol.commit()

        # ---- Stage 2 discriminator (gated on T3) ---------------------------
        from shared import progress

        t3_ok = any(
            progress.exists(f"stage2/T3/seed{s}/verification.json")
            and progress.load_json(f"stage2/T3/seed{s}/verification.json")["converged"]
            for s in SEEDS
        )
        if not t3_ok:
            log(
                "MILESTONE: T3 did not converge on any seed — Stage 2 discriminator does "
                "not run (spec). Training trajectories are the deliverable."
            )
        else:
            from shared.run import run_stage2_discriminator

            stage2_results = run_stage2_discriminator(SEEDS, device=device)
            verdicts = {k: r.get("verdict") for k, r in stage2_results.items()}
            log(f"MILESTONE: Stage 2 discriminator done — verdicts={verdicts}")

        # ---- final report ----------------------------------------------------
        from shared import report

        report.write_verdict_matrix(SEEDS)
        dr = report.dose_response(SEEDS)
        try:
            report.plot_dose_response(dr)
        except Exception as e:
            log(f"plot_dose_response failed (non-fatal): {e}")
        report.final_report(SEEDS)
        log("MILESTONE: FINAL REPORT WRITTEN — run complete")
        return "done"
    except Exception:
        log("MILESTONE: RUN CRASHED:\n" + traceback.format_exc())
        raise
    finally:
        stop.set()
        vol.commit()


@app.local_entrypoint()
def main():
    print(run_all.remote())

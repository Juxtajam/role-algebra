"""Authorised lever 1 — weight decay: T1, 4 layers, 3 seeds, hard 50k,
wd in {0.1, 1.0}. The wd=0.01 arm is the existing depth_sweep/T1_L4 runs
(identical protocol: 4 layers, hard 50k, seeds 0-2, wd=0.01) — reused, not
rerun; declared in the report. Curriculum lever is held fixed (not applied).

train.py now also logs train-distribution composed accuracy at every eval
(mandated by Check 2) from a separate rng stream so the training data order
is unchanged.
"""

import modal
import pathlib

app = modal.App("dv3-wd-sweep")
vol = modal.Volume.from_name("dv3-results")

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


@app.function(image=image, gpu="A10G", timeout=60 * 60 * 8, volumes={"/results": vol})
def sweep_wd(wd: float):
    import os, sys, threading, time, traceback

    os.environ["DV3_RESULTS"] = "/results"
    os.environ["MPLBACKEND"] = "Agg"
    sys.path.insert(0, "/root/dv3")
    os.chdir("/root/dv3")

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
    from shared import progress
    from shared.progress import log
    from trained.train import train_organism

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tag = str(wd).replace(".", "p")
    log(f"WD SWEEP wd={wd} START — device={device}")
    results = {}
    try:
        for s in SEEDS:
            rk = f"wd_sweep/T1_wd{tag}/seed{s}"
            ver_rel = f"stage2/{rk}/verification.json"
            if progress.exists(ver_rel):
                ver = progress.load_json(ver_rel)
                log(
                    f"wd sweep {rk}: already done (final acc "
                    f"{ver['final'].get('held_episode', 0):.3f})"
                )
            else:
                ver = train_organism(
                    "T1",
                    s,
                    max_steps=MAX_STEPS,
                    device=device,
                    n_layers=4,
                    run_key=rk,
                    hard_stop=True,
                    weight_decay=wd,
                )
            results[rk] = dict(
                held=ver["final"].get("held_episode", 0.0),
                train=ver["final"].get("train_composed", None),
            )
            vol.commit()
        log(f"MILESTONE: wd sweep wd={wd} complete — {results}")
    except Exception:
        log(f"MILESTONE: WD SWEEP wd={wd} CRASHED:\n" + traceback.format_exc())
        raise
    finally:
        stop.set()
        vol.commit()
    return results


@app.function(image=image, timeout=3600, volumes={"/results": vol})
def finalize():
    import os, sys

    os.environ["DV3_RESULTS"] = "/results"
    sys.path.insert(0, "/root/dv3")
    os.chdir("/root/dv3")
    from shared import progress
    from shared.progress import log
    from shared.report import transition_report

    summary = {}
    # wd=0.01 arm = existing depth_sweep/T1_L4 runs (declared reuse)
    for s in (0, 1, 2):
        rel = f"stage2/depth_sweep/T1_L4/seed{s}/verification.json"
        if progress.exists(rel):
            v = progress.load_json(rel)
            summary[f"wd0.01(reused L4)/seed{s}"] = dict(
                held=v["final"].get("held_episode", 0.0), steps=v["steps"]
            )
    for tag, wd in (("0p1", 0.1), ("1p0", 1.0)):
        transition_report(f"wd_sweep/T1_wd{tag}", (0, 1, 2))
        for s in (0, 1, 2):
            rel = f"stage2/wd_sweep/T1_wd{tag}/seed{s}/verification.json"
            if progress.exists(rel):
                v = progress.load_json(rel)
                summary[f"wd{wd}/seed{s}"] = dict(
                    held=v["final"].get("held_episode", 0.0),
                    train=v["final"].get("train_composed"),
                    steps=v["steps"],
                    converged=v["converged"],
                )
    progress.save_json("stage2/wd_sweep_summary.json", summary)
    log(f"MILESTONE: WD SWEEP SUMMARY: {summary}")
    vol.commit()
    return summary


@app.local_entrypoint()
def main():
    results = list(sweep_wd.map([0.1, 1.0]))
    print("wd sweep:", results)
    print("finalize:", finalize.remote())

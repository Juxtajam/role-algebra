"""7.3 — Depth sweep: T1 at 4/6/8 layers x 3 seeds, hard 50k steps.

Preconditions resolved: aux-query scoring audited clean
(candidate masks correct, chance 1/3 as assumed, solver 0/2000 on aux
episodes, masked==unmasked accuracy), so chance-level aux performance is
genuine and the hop-vs-composition branch is UNAVAILABLE, not hop_failure.
This is written to the volume as a declared resolution; the depth sweep's
refusal gate is bypassed with that declaration (the design explicitly
directs proceeding to 7.3).

Parallelism: one container per depth (3 seeds serial inside each).
hard_stop=True: patience disabled, every run goes the full 50k.
"""

import modal
import pathlib

app = modal.App("dv3-depth-sweep")
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


def _setup():
    import os, sys

    os.environ["DV3_RESULTS"] = "/results"
    os.environ["MPLBACKEND"] = "Agg"
    sys.path.insert(0, "/root/dv3")
    os.chdir("/root/dv3")


@app.function(image=image, gpu="A10G", timeout=60 * 60 * 10, volumes={"/results": vol})
def sweep_depth(n_layers: int):
    import threading, time, traceback

    _setup()
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
    log(f"DEPTH SWEEP L={n_layers} START — device={device}")
    results = {}
    try:
        for s in SEEDS:
            rk = f"depth_sweep/T1_L{n_layers}/seed{s}"
            ver_rel = f"stage2/{rk}/verification.json"
            if progress.exists(ver_rel):
                ver = progress.load_json(ver_rel)
                log(
                    f"depth sweep {rk}: already done (final acc "
                    f"{ver['final'].get('held_episode', 0):.3f})"
                )
            else:
                ver = train_organism(
                    "T1",
                    s,
                    max_steps=MAX_STEPS,
                    device=device,
                    n_layers=n_layers,
                    run_key=rk,
                    hard_stop=True,
                )
            results[rk] = ver["final"].get("held_episode", 0.0)
            vol.commit()
        log(
            f"MILESTONE: depth sweep L={n_layers} complete — final composed acc: {results}"
        )
    except Exception:
        log(f"MILESTONE: DEPTH SWEEP L={n_layers} CRASHED:\n" + traceback.format_exc())
        raise
    finally:
        stop.set()
        vol.commit()
    return results


@app.function(image=image, timeout=3600, volumes={"/results": vol})
def record_resolution():
    _setup()
    from shared import progress
    from shared.progress import log

    resolution = dict(
        verdict="unavailable",
        prior_verdict="hop_failure (all T1 seeds)",
        basis="7.1 audit: candidate masks correct (3 same-type candidates, chance 1/3), "
        "labels/solver clean on 2000 aux episodes per type per organism, "
        "masked==unmasked accuracy, realised aux proportions 10-13% per type. "
        "Chance-level aux performance is genuine, so aux queries are not serving "
        "their diagnostic purpose; hop-vs-composition is UNAVAILABLE, not hop_failure. "
        "Depth sweep proceeds.",
        design_fault_recorded="T0 leakage was meant for the composed task but is inherited "
        "by person-answer auxiliary queries (symbol->person at 1.00 "
        "via position in T0).",
    )
    progress.save_json("stage2/hop_vs_composition_resolution.json", resolution)
    log(
        "MILESTONE: hop-vs-composition RESOLVED as UNAVAILABLE (audit clean, aux genuinely "
        "at chance) — proceeding to depth sweep"
    )
    vol.commit()
    return "recorded"


@app.function(image=image, timeout=3600, volumes={"/results": vol})
def finalize():
    """Aggregate after all depths finish: transition reports + summary + plots."""
    _setup()
    from shared import progress
    from shared.progress import log
    from shared.report import transition_report

    summary = {}
    for L in (4, 6, 8):
        transition_report(f"depth_sweep/T1_L{L}", SEEDS)
        for s in SEEDS:
            rel = f"stage2/depth_sweep/T1_L{L}/seed{s}/verification.json"
            if progress.exists(rel):
                v = progress.load_json(rel)
                summary[f"L{L}/seed{s}"] = dict(
                    final_acc=v["final"].get("held_episode", 0.0),
                    consistency=v["final"].get("held_consistency", 0.0),
                    transfer=v["final"].get("transfer_episode", 0.0),
                    steps=v["steps"],
                    converged=v["converged"],
                )
    progress.save_json("stage2/depth_sweep_summary.json", summary)
    best8 = max(
        (summary.get(f"L8/seed{s}", {}).get("final_acc", 0.0) for s in SEEDS),
        default=0.0,
    )
    if best8 >= 0.95:
        verdict = (
            f"DEPTH SWEEP RESULT: 8 layers reaches {best8:.3f} composed accuracy — "
            "adopt depth 8 and rerun T0/T2/T3 at that depth"
        )
    elif best8 < 0.4:
        verdict = (
            f"DEPTH SWEEP RESULT: 8 layers still at chance ({best8:.3f}) at 50k — "
            "depth line closed; report and stop"
        )
    else:
        verdict = f"DEPTH SWEEP RESULT: 8 layers partial ({best8:.3f}) — report as-is"
    log(f"MILESTONE: {verdict}")
    log(f"MILESTONE: depth sweep summary: {summary}")
    vol.commit()
    return summary


@app.local_entrypoint()
def main():
    record_resolution.remote()
    results = list(sweep_depth.map([4, 6, 8]))
    print("sweep results:", results)
    print("finalize:", finalize.remote())

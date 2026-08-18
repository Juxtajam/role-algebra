"""Orchestration: Stage 1 (synthetic suite, gates Stage 2) and Stage 2
(trained transformers, dose-response). Every unit of work is guarded by its
results file, so re-running a cell resumes instead of repeating."""

from shared import calibrate
from shared import discriminator as disc
from shared import progress
from shared.progress import log

SNRS = (10, 3, 1)
ORGANISM_ORDER = ("S-role", "S-shared", "S-retrieval", "S-position")


def make_synth_system(name, snr):
    from synth.model import Frame
    from synth.organisms import ORGANISMS

    if not hasattr(make_synth_system, "_frame"):
        make_synth_system._frame = Frame()
    return ORGANISMS[name](make_synth_system._frame, snr)


def _verdict_table(results):
    from synth.organisms import REQUIRED

    lines = [f"{'organism':<14}" + "".join(f"SNR {s:<12}" for s in SNRS)]
    for name in ORGANISM_ORDER:
        row = f"{name:<14}"
        for snr in SNRS:
            r = results.get(f"{name}@{snr}")
            if r is None:
                row += f"{'—':<16}"
                continue
            mark = ""
            want = REQUIRED.get(name, {}).get(snr)
            if want is not None:
                mark = " pass" if r["verdict"] == want else " fail"
            row += f"{r['verdict']:<12}{mark:<4}"
        lines.append(row)
    return "\n".join(lines)


def run_stage1(force=False):
    cfg_all = progress.load_json(calibrate.THRESHOLDS_PATH)
    cfg1 = cfg_all["stage1"]
    results = {}
    for name in ORGANISM_ORDER:
        for snr in SNRS:
            key = f"{name}@{snr}"
            out_path = f"stage1/{key}/discriminator.json"
            if progress.exists(out_path) and not force:
                results[key] = progress.load_json(out_path)
                log(f"stage1 {key}: cached (verdict={results[key]['verdict']})")
                continue
            system = make_synth_system(name, snr)
            acc = system.self_decode_accuracy()
            progress.save_json(
                f"stage1/{key}/self_accuracy.json",
                dict(accuracy=acc, required=0.99, ok=acc >= 0.99),
            )
            log(f"stage1 {key}: self-decode accuracy {acc:.4f} (gate >= 0.99)")
            if acc < 0.99:
                results[key] = dict(
                    verdict="self_decode_failed", score=0.0, self_accuracy=acc
                )
                progress.save_json(out_path, results[key])
                continue
            run_cfg = dict(
                layer=cfg1["runs"][key]["layer"],
                thresholds=cfg1["thresholds"],
                **{"lambda": cfg1["lambda"]},
            )
            r = disc.run_frozen(system, run_cfg, run_id=f"stage1 {key}")
            r["self_accuracy"] = acc
            r["first_decodable"] = cfg1["runs"][key]["first_decodable"]
            results[key] = r
            progress.save_json(out_path, r)
            log("stage1 verdicts so far:\n" + _verdict_table(results))

    from synth.organisms import REQUIRED

    failures = [
        f"{n}@{s}: got {results[f'{n}@{s}']['verdict']}, need {want}"
        for n, snr_map in REQUIRED.items()
        for s, want in snr_map.items()
        if results[f"{n}@{s}"]["verdict"] != want
    ]
    # sensitivity floor: lowest SNR at which S-role is still detected
    detected = [s for s in SNRS if results[f"S-role@{s}"]["verdict"] == "H_role"]
    floor = min(detected) if detected else None
    summary = dict(
        passed=not failures,
        failures=failures,
        snr_sensitivity_floor=floor,
        s_shared_verdicts={s: results[f"S-shared@{s}"]["verdict"] for s in SNRS},
        verdicts={k: r["verdict"] for k, r in results.items()},
        scores={k: r["score"] for k, r in results.items()},
    )
    progress.save_json("stage1/summary.json", summary)
    log("stage1 final verdict table:\n" + _verdict_table(results))
    log(
        f"stage1 {'PASSED' if summary['passed'] else 'FAILED'}"
        + (
            f" — S-role sensitivity floor: SNR {floor}"
            if floor
            else " — S-role never detected"
        )
    )
    if failures:
        for f in failures:
            log(f"stage1 gate failure: {f}")
    log(
        f"S-shared verdicts (reported as a stated limitation, not gated): {summary['s_shared_verdicts']}"
    )
    return summary


def stage1_passed():
    return (
        progress.exists("stage1/summary.json")
        and progress.load_json("stage1/summary.json")["passed"]
    )


# ---------------------------------------------------------------------------
# Stage 2 — runs only if Stage 1 passed
# ---------------------------------------------------------------------------
TRAINING_ORGANISMS = ("T0", "T1", "T2", "T3")  # ordered by role-reuse pressure


def _stale_verification(ver, max_steps, floor):
    """A non-converged run that stopped before the minimum-steps floor was
    killed by the vacuous early-stop criterion (chance-level runs satisfy
    patience automatically) — its non-convergence result is invalid and the
    run must continue from its checkpoint."""
    return (
        not ver.get("converged")
        and not ver.get("smoke")
        and ver.get("steps", 0) < min(floor, max_steps)
    )


def depth_suffix(n_layers):
    """Runs at non-default depth live under suffixed keys (T1_L8/seed0) so a
    depth adoption re-keys the whole grid instead of clobbering 4-layer runs."""
    return "" if n_layers == 4 else f"_L{n_layers}"


def _train_orgs(names, seeds, max_steps, device, smoke, verifications, n_layers=4):
    from trained.train import MIN_STEPS_FLOOR, train_organism

    for name in names:
        for seed in seeds:
            key = f"{name}{depth_suffix(n_layers)}/seed{seed}"
            ver_path = f"stage2/{key}/verification.json"
            if progress.exists(ver_path):
                ver = progress.load_json(ver_path)
                if not _stale_verification(ver, max_steps, MIN_STEPS_FLOOR):
                    verifications[key] = ver
                    log(
                        f"stage2 {key}: training already verified (converged={ver['converged']})"
                    )
                    continue
                log(
                    f"stage2 {key}: prior run stopped at step {ver.get('steps')} before the "
                    f"{MIN_STEPS_FLOOR}-step floor — invalid early stop, resuming from checkpoint "
                    f"(declared in the report)"
                )
                (progress.results_dir() / ver_path).unlink()
            verifications[key] = train_organism(
                name,
                seed,
                max_steps=max_steps,
                device=device,
                smoke=smoke,
                n_layers=n_layers,
                run_key=key,
            )


def run_stage2_training(seeds, max_steps, device="cpu", smoke=False, n_layers=4):
    """Train (or resume) organism x seed with diagnostic ordering: T0 and T1
    first. T1 (24-name pool, single path) is the simplest organism in the
    suite — note the recorded design-premise correction: bindings are
    resampled per episode, so nothing is memorisable at any pool size and T1
    differs from T2 only in name-embedding gradient density (flagged in the
    report, not redesigned). If every T1 seed remains at chance through
    max_steps, that points to a data or supervision fault (audit the data,
    report, stop) — do NOT search hyperparameters, and do not spend compute
    on T2/T3. Runs whose non-convergence predates the early-stop floor are
    invalid and are automatically continued from their checkpoints.
    T3 non-convergence halts the Stage 2 discriminator (spec)."""
    sfx = depth_suffix(n_layers)
    verifications = {}
    _train_orgs(("T0", "T1"), seeds, max_steps, device, smoke, verifications, n_layers)
    t1_left_chance = any(
        verifications.get(f"T1{sfx}/seed{s}", {})
        .get("final", {})
        .get("held_episode", 0)
        > 0.40
        for s in seeds
    )
    if not (t1_left_chance or smoke):
        log(
            "STAGE 2 STOPPED before T2/T3: every T1 seed remained at chance through "
            f"{max_steps} steps. A distribution solvable by memorisation over 24 names "
            "failing here points to a data or supervision fault, not task difficulty — "
            "run the data audit cell and report. Do not search hyperparameters."
        )
        return verifications, False
    _train_orgs(("T2", "T3"), seeds, max_steps, device, smoke, verifications, n_layers)
    t3_ok = any(
        verifications.get(f"T3{sfx}/seed{s}", {}).get("converged") for s in seeds
    )
    if not t3_ok:
        log(
            "STAGE 2 HALTED: no T3 seed converged. Reporting trajectories and stopping "
            "(spec: do not search hyperparameters further)."
        )
    return verifications, t3_ok


def run_depth_sweep(
    seeds, device="cpu", org="T1", depths=(4, 6, 8), max_steps=50_000, force=False
):
    """Task 3, conditional on task 2: train T1 at 4/6/8 layers, 3 seeds each,
    all other hyperparameters unchanged, max_steps as a HARD stop (patience
    disabled — non-convergence at 50k has never actually been tested; the
    25k floor stays in the main training path). Refuses to run unless the
    breakdown shows a composition failure (hop failure means depth is the
    wrong lever)."""
    from shared.report import transition_report
    from trained.train import train_organism

    branch_rel = "stage2/hop_vs_composition.json"
    if not force:
        if not progress.exists(branch_rel):
            log(
                "depth sweep REFUSED: run the query-breakdown and branch cells first "
                "(task order is mandatory)."
            )
            return None
        verdicts = progress.load_json(branch_rel)
        if any(v == "hop_failure" for v in verdicts.values()):
            log(
                "depth sweep REFUSED: breakdown shows a hop failure — more layers will "
                "not help a model that cannot do one hop. Inspect episode dumps."
            )
            return None
        if not any(v == "composition_failure" for v in verdicts.values()):
            log(
                "depth sweep REFUSED: breakdown does not show the composition-failure "
                "pattern (pass force=True to override; that is a protocol deviation)."
            )
            return None
    summary = {}
    for L in depths:
        for s in seeds:
            rk = f"depth_sweep/{org}_L{L}/seed{s}"
            ver_rel = f"stage2/{rk}/verification.json"
            if progress.exists(ver_rel):
                ver = progress.load_json(ver_rel)
                log(
                    f"depth sweep {rk}: already done (final acc "
                    f"{ver['final'].get('held_episode', 0):.3f})"
                )
            else:
                ver = train_organism(
                    org,
                    s,
                    max_steps=max_steps,
                    device=device,
                    n_layers=L,
                    run_key=rk,
                    hard_stop=True,
                )
            summary[rk] = ver["final"].get("held_episode", 0.0)
        transition_report(f"depth_sweep/{org}_L{L}", seeds)
    best8 = max(
        (summary.get(f"depth_sweep/{org}_L8/seed{s}", 0.0) for s in seeds), default=0.0
    )
    if best8 >= 0.95:
        log(
            f"DEPTH SWEEP RESULT: 8 layers reaches {best8:.3f} composed accuracy on {org} "
            "— adopt depth 8 and rerun T0/T2/T3 at that depth "
            "(run_stage2_training with n_layers=8 keys)."
        )
    elif best8 < 0.4:
        log(
            f"DEPTH SWEEP RESULT: 8 layers still at chance ({best8:.3f}) at {max_steps} "
            "steps — the depth line is closed; report and stop."
        )
    else:
        log(f"DEPTH SWEEP RESULT: 8 layers partial ({best8:.3f}) — report as-is.")
    progress.save_json("stage2/depth_sweep_summary.json", summary)
    return summary


def run_stage2_discriminator(seeds, device="cpu", force=False, n_layers=4):
    from trained.train import load_system

    results = {}
    for name in TRAINING_ORGANISMS:
        for seed in seeds:
            key = f"{name}{depth_suffix(n_layers)}/seed{seed}"
            out_path = f"stage2/{key}/discriminator.json"
            ver_path = f"stage2/{key}/verification.json"
            if not progress.exists(ver_path):
                log(f"stage2 {key}: no verification — skipping discriminator")
                continue
            ver = progress.load_json(ver_path)
            if not ver.get("enter_discriminator", False):
                log(
                    f"stage2 {key}: did not meet the convergence gate — recorded, no discriminator run"
                )
                results[key] = dict(verdict="not_converged", score=None)
                continue
            if progress.exists(out_path) and not force:
                results[key] = progress.load_json(out_path)
                log(f"stage2 {key}: cached (verdict={results[key]['verdict']})")
                continue
            system = load_system(name, seed, device=device, n_layers=n_layers)
            cfg = calibrate.calibrate_system(system, key)
            run_cfg = dict(
                layer=cfg["layer"],
                thresholds=cfg["thresholds"],
                **{"lambda": cfg["lambda"]},
            )
            r = disc.run_frozen(system, run_cfg, run_id=f"stage2 {key}")
            results[key] = r
            progress.save_json(out_path, r)
            _interim_dose_response(results)
    return results


def _interim_dose_response(results):
    """Print partial dose-response after every finished run so progress is
    visible long before the full grid completes."""
    rows = [
        (k, r.get("score"))
        for k, r in sorted(results.items())
        if r.get("score") is not None
    ]
    if rows:
        log(
            "interim dose-response (score by organism/seed):\n"
            + "\n".join(f"  {k:<12} score={s:.3f}" for k, s in rows)
        )

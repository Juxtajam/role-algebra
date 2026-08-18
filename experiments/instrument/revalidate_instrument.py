"""Revalidate the instrument from scratch.

The whole programme depends on the discriminator DETECTING role structure when
it is present. This re-runs the actual frozen discriminator code on FRESHLY
DRAWN synthetic organisms (several independent noise seeds) with the frozen
thresholds, and checks that the verdicts reproduce the claim that the
instrument returns the correct answer on systems with known ground truth:

  S-role      -> H_role       (SNR 10, 3; floor at SNR 1)
  S-shared    -> H_role       (a genuine role rep, E2)
  S-retrieval -> H_retrieval
  S-position  -> H_retrieval

Does NOT touch the stored results (reads the frozen thresholds, computes fresh
verdicts in memory). Writes results/revalidation/stage1_revalidation.json.
"""

import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from synth.model import Frame  # noqa: E402
from synth.organisms import ORGANISMS, REQUIRED  # noqa: E402
from shared import discriminator as disc  # noqa: E402

SNRS = (10, 3, 1)
ORG_ORDER = ("S-role", "S-shared", "S-retrieval", "S-position")
N_SEEDS = 5  # independent noise draws per (organism, SNR)


def main():
    cfg1 = json.load(open(ROOT / "results/instrument/calibration/thresholds.json"))["stage1"]
    frame = Frame()  # the fixed shared frame (same as the programme)
    print(
        f"frozen lambda={cfg1['lambda']}; thresholds for "
        f"{len(cfg1['thresholds'])} metrics; {len(cfg1['runs'])} per-run layers"
    )

    out = {"n_seeds": N_SEEDS, "frozen_lambda": cfg1["lambda"], "results": {}}
    print(
        f"\n{'organism':<13} {'SNR':>4} | verdicts across "
        f"{N_SEEDS} fresh noise seeds | required | reproduces?"
    )
    all_ok = True
    for name in ORG_ORDER:
        for snr in SNRS:
            key = f"{name}@{snr}"
            layer = cfg1["runs"][key]["layer"]
            run_cfg = dict(
                layer=layer, thresholds=cfg1["thresholds"], **{"lambda": cfg1["lambda"]}
            )
            verdicts, scores, self_accs = [], [], []
            for seed in range(N_SEEDS):
                org = ORGANISMS[name](frame, snr, seed=seed)
                acc = org.self_decode_accuracy()
                self_accs.append(round(acc, 4))
                if acc < 0.99:
                    verdicts.append("self_decode_failed")
                    continue
                r = disc.run_frozen(org, run_cfg, run_id="")
                verdicts.append(r["verdict"])
                scores.append(round(r["score"], 3))
            want = REQUIRED.get(name, {}).get(snr)
            modal_v = max(set(verdicts), key=verdicts.count)
            reproduces = (want is None) or all(v == want for v in verdicts)
            if want is not None and not reproduces:
                all_ok = False
            out["results"][key] = dict(
                required=want,
                verdicts=verdicts,
                modal_verdict=modal_v,
                scores=scores,
                self_accuracy=self_accs,
                reproduces=bool(reproduces if want is not None else None),
            )
            vs = "/".join(v.replace("H_", "") for v in verdicts)
            flag = (
                "REPRODUCES"
                if want is None
                else ("REPRODUCES pass" if reproduces else "MISMATCH fail")
            )
            print(
                f"{name:<13} {snr:>4} | {vs:<38} | {str(want).replace('H_',''):<12} | {flag}"
            )

    # sensitivity floor from the fresh runs
    floor = None
    for s in SNRS:
        vs = out["results"][f"S-role@{s}"]["verdicts"]
        if all(v == "H_role" for v in vs):
            floor = s
    out["snr_sensitivity_floor_fresh"] = floor
    out["all_required_reproduce"] = all_ok
    (ROOT / "results/instrument/revalidation").mkdir(parents=True, exist_ok=True)
    json.dump(
        out, open(ROOT / "results/instrument/revalidation/stage1_revalidation.json", "w"), indent=1
    )

    print(f"\nsensitivity floor (fresh, all-seeds S-role=H_role): SNR {floor}")
    print(
        f"REVALIDATION: {'PASS — instrument reproduces E1 on fresh draws' if all_ok else 'FAIL — verdicts do not reproduce'}"
    )
    print(
        "S-shared (E2: a genuine role rep, expected H_role):",
        out["results"]["S-shared@10"]["modal_verdict"],
        out["results"]["S-shared@3"]["modal_verdict"],
    )
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()

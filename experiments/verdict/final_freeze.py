"""Phase 8A-final — Step 1: FREEZE the run-2 task set + scoring code.

Verifies local == volume byte-exact, then computes the content hash:
  sha256 over the concatenation of "relpath:sha256(file)\n" lines, files in
  sorted relpath order, over:
    tasks/{gate,disc}_{P,G}_{fit,transfer}.jsonl   (8 episode files)
    tasks/manifest.json  tasks/name_pools.json     (episode-set metadata)
    phase8a_modal.py                               (scoring code + thresholds)
    phase8a_generate_tasks.py                      (frozen generation procedure)
Writes phase8a_final_freeze.json (per-file hashes + combined content hash).
"""

import hashlib
import json
import pathlib
import sys

CODE_DIR = (
    pathlib.Path(__file__).resolve().parent
)  # experiments/ (hashed code lives here)
ROOT = CODE_DIR.parent  # repository root
LOCAL_TASKS = ROOT / "results/verdict/gate/tasks"
VOL = pathlib.Path("/tmp/freeze_check")

TASK_FILES = sorted(
    [
        f"{k}_{p}_{v}.jsonl"
        for k in ("gate", "disc")
        for p in ("P", "G")
        for v in ("fit", "transfer")
    ]
    + ["manifest.json", "name_pools.json"]
)
CODE_FILES = ["phase8a_generate_tasks.py", "phase8a_modal.py"]


def sha(p):
    return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()


def main():
    # 1. local vs volume byte-equality
    mismatches = []
    for f in TASK_FILES:
        if sha(LOCAL_TASKS / f) != sha(VOL / "tasks" / f):
            mismatches.append(f"tasks/{f}")
    for f in CODE_FILES:
        if sha(CODE_DIR / f) != sha(VOL / f):
            mismatches.append(f)
    if mismatches:
        print("BYTE MISMATCH local vs dv3-results volume:", mismatches)
        sys.exit(1)
    print(
        f"local == volume byte-exact: {len(TASK_FILES)} task files + "
        f"{len(CODE_FILES)} code files"
    )

    # 2. content hash
    entries = {}
    for f in TASK_FILES:
        entries[f"tasks/{f}"] = sha(LOCAL_TASKS / f)
    for f in CODE_FILES:
        entries[f] = sha(CODE_DIR / f)
    blob = "".join(f"{k}:{v}\n" for k, v in sorted(entries.items()))
    content_hash = hashlib.sha256(blob.encode()).hexdigest()

    # 3. episode counts sanity (from manifest, cross-checked against files)
    manifest = json.load(open(LOCAL_TASKS / "manifest.json"))
    counts = {}
    for f in TASK_FILES:
        if f.endswith(".jsonl"):
            counts[f] = sum(1 for _ in open(LOCAL_TASKS / f))
    expected = {"gate": 600, "disc": 1800}
    for f, n in counts.items():
        assert n == expected[f.split("_")[0]], (f, n)

    freeze = dict(
        phase="8A-final",
        frozen_from="run-2 task set (dv3-results:phase8a/tasks/) + scoring code",
        hash_definition=(
            "sha256 over concatenation of 'relpath:sha256(file)\\n' "
            "lines, sorted by relpath"
        ),
        files=entries,
        content_hash=content_hash,
        episode_counts=counts,
        gate_thresholds=dict(
            episode_acc=0.95, strict_orbit=0.90, orbit_consistency=0.95
        ),
        manifest_master_seed=manifest["master_seed"],
        verified_local_equals_volume=True,
    )
    out = ROOT / "results/verdict/gate/phase8a_final_freeze.json"
    json.dump(freeze, open(out, "w"), indent=2)
    print("CONTENT HASH:", content_hash)
    print("written:", out)


if __name__ == "__main__":
    main()

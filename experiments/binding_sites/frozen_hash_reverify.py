"""Phase 9 STAGE 1 (local prep) — Task 1: re-verify the frozen episode-set
content hash with the EXACT 8A-final procedure.

Hash manifest source: results/phase8a/phase8a_final_freeze.json, produced by
phase8a_final_freeze.py. Definition reused verbatim: sha256 over the
concatenation of 'relpath:sha256(file)\\n' lines, sorted by relpath, over:
  tasks/{gate,disc}_{P,G}_{fit,transfer}.jsonl   (8 episode files)
  tasks/manifest.json  tasks/name_pools.json     (episode-set metadata)
  phase8a_generate_tasks.py                      (frozen generation procedure)
  phase8a_modal.py                               (scoring code + thresholds)
(File list and sha()/blob construction copied byte-for-byte in spirit from
phase8a_final_freeze.py; also byte-compares local files against the
dv3-results volume copies fetched to /tmp/p9_freeze_check.)

Writes results/phase9/frozen_hash_reverify.json.
"""

import hashlib
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
LOCAL_TASKS = ROOT / "results/verdict/gate/tasks"
VOL = pathlib.Path("/tmp/p9_freeze_check")
OUT = ROOT / "results/binding_sites"
OUT.mkdir(parents=True, exist_ok=True)

EXPECTED = "84f2e54d85d6e8aa4c1474b608bef5ab69babe54353ef0ef2702d9f6ed38baef"

# --- EXACT file list from phase8a_final_freeze.py ---
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
    entries = {}
    vol_match = {}
    for f in TASK_FILES:
        entries[f"tasks/{f}"] = sha(LOCAL_TASKS / f)
        vol_match[f"tasks/{f}"] = entries[f"tasks/{f}"] == sha(VOL / "tasks" / f)
    for f in CODE_FILES:
        entries[f] = sha(ROOT / f)
        vol_match[f] = entries[f] == sha(VOL / f)

    # --- EXACT hash construction from phase8a_final_freeze.py ---
    blob = "".join(f"{k}:{v}\n" for k, v in sorted(entries.items()))
    h = hashlib.sha256(blob.encode()).hexdigest()

    ok = (h == EXPECTED) and all(vol_match.values())
    rec = dict(
        phase="9-stage1-prep",
        step="frozen-set content-hash re-verification (8A-final procedure)",
        hash_definition=(
            "sha256 over concatenation of 'relpath:sha256(file)"
            "\\n' lines, sorted by relpath"
        ),
        manifest_source="results/verdict/gate/phase8a_final_freeze.json",
        expected=EXPECTED,
        recomputed=h,
        match=h == EXPECTED,
        local_equals_volume=vol_match,
        all_local_equal_volume=all(vol_match.values()),
        all_ok=bool(ok),
        files=entries,
    )
    json.dump(rec, open(OUT / "frozen_hash_reverify.json", "w"), indent=2)
    print(f"recomputed: {h}")
    print(f"expected:   {EXPECTED}")
    print(f"match={h == EXPECTED}  local==volume: {all(vol_match.values())}")
    print("written:", OUT / "frozen_hash_reverify.json")
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()

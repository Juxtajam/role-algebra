"""Phase 9 — 9A-prep step 1: RE-VERIFY the frozen episode-set content hash
before the caching session.

Same hash definition as phase8a_final_freeze.py (sha256 over concatenation of
'relpath:sha256(file)\\n' lines, sorted by relpath) over the 8 episode files +
2 metadata files + 2 code files. Also re-verifies local == dv3-results volume
byte-exact (volume files fetched to /tmp/p9_freeze_check by the caller).

Writes results/phase9/frozen_set_reverify.json.
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
VOL = pathlib.Path("/tmp/p9_freeze_check")
OUT = ROOT / "results/binding_sites"
OUT.mkdir(parents=True, exist_ok=True)

EXPECTED = "84f2e54d85d6e8aa4c1474b608bef5ab69babe54353ef0ef2702d9f6ed38baef"

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
        vol_match[f"tasks/{f}"] = sha(LOCAL_TASKS / f) == sha(VOL / "tasks" / f)
    for f in CODE_FILES:
        entries[f] = sha(CODE_DIR / f)
        vol_match[f] = sha(CODE_DIR / f) == sha(VOL / f)
    blob = "".join(f"{k}:{v}\n" for k, v in sorted(entries.items()))
    h = hashlib.sha256(blob.encode()).hexdigest()
    ok = h == EXPECTED and all(vol_match.values())
    rec = dict(
        phase="9A-prep",
        step="frozen-set re-verification",
        expected=EXPECTED,
        recomputed=h,
        match=h == EXPECTED,
        local_equals_volume=vol_match,
        all_ok=bool(ok),
        files=entries,
    )
    json.dump(rec, open(OUT / "frozen_set_reverify.json", "w"), indent=2)
    print(f"recomputed: {h}")
    print(f"expected:   {EXPECTED}")
    print(f"match={h == EXPECTED}  local==volume: {all(vol_match.values())}")
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()

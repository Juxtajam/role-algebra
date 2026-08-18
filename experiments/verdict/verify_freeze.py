"""Phase 8C — Step 0: verify frozen content hash (volume bytes) + local copies."""

import hashlib, pathlib, json

root = pathlib.Path("/tmp/phase8c_freeze")
tdir = root / "tasks"
if (tdir / "tasks").exists():
    tdir = tdir / "tasks"
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
EXPECTED = "84f2e54d85d6e8aa4c1474b608bef5ab69babe54353ef0ef2702d9f6ed38baef"


def sha(p):
    return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()


entries = {}
for f in TASK_FILES:
    entries[f"tasks/{f}"] = sha(tdir / f)
for f in CODE_FILES:
    entries[f] = sha(root / f)
blob = "".join(f"{k}:{v}\n" for k, v in sorted(entries.items()))
h = hashlib.sha256(blob.encode()).hexdigest()
print("VOLUME CONTENT HASH:", h)
print("MATCH:", h == EXPECTED)

loc = pathlib.Path(__file__).resolve().parents[2]
mism = []
for f in TASK_FILES:
    if sha(loc / "results/verdict/gate/tasks" / f) != entries[f"tasks/{f}"]:
        mism.append(f)
for f in CODE_FILES:
    if sha(loc / f) != entries[f]:
        mism.append(f)
print("local-vs-volume mismatches:", mism or "NONE")

out = dict(
    volume_content_hash=h,
    matches_frozen=h == EXPECTED,
    local_mismatches=mism,
    files=entries,
)
json.dump(
    out,
    open(
        str(
            pathlib.Path(__file__).resolve().parents[2]
            / "results/verdict/discriminator_hash_verify.json"
        ),
        "w",
    ),
    indent=2,
)
assert h == EXPECTED and not mism
print("STEP 0 (content hash): PASS")

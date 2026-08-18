"""Phase 9 STAGE 1 (local prep) — Task 2b: content hash over the
joint-permutation episode set + its generation code.

Same hash definition as the 8A-final freeze (phase8a_final_freeze.py):
sha256 over the concatenation of 'relpath:sha256(file)\\n' lines, sorted by
relpath, over:
  tasks_joint/joint_{P,G}_{fit,transfer}.jsonl   (4 episode files)
  tasks_joint/manifest.json  tasks_joint/README.md
  code/phase9_generate_joint.py                  (joint generation code)
  phase8a_generate_tasks.py                      (frozen module it imports)

Writes results/phase9/joint_hash.json.
"""

import hashlib
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[2]
P9 = ROOT / "results/binding_sites"

FILES = {
    "tasks_joint/joint_P_fit.jsonl": P9 / "tasks_joint/joint_P_fit.jsonl",
    "tasks_joint/joint_P_transfer.jsonl": P9 / "tasks_joint/joint_P_transfer.jsonl",
    "tasks_joint/joint_G_fit.jsonl": P9 / "tasks_joint/joint_G_fit.jsonl",
    "tasks_joint/joint_G_transfer.jsonl": P9 / "tasks_joint/joint_G_transfer.jsonl",
    "tasks_joint/manifest.json": P9 / "tasks_joint/manifest.json",
    "tasks_joint/README.md": P9 / "tasks_joint/README.md",
    "code/phase9_generate_joint.py": P9 / "code/phase9_generate_joint.py",
    "phase8a_generate_tasks.py": ROOT / "phase8a_generate_tasks.py",
}


def sha(p):
    return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()


def main():
    entries = {rel: sha(fp) for rel, fp in FILES.items()}
    blob = "".join(f"{k}:{v}\n" for k, v in sorted(entries.items()))
    h = hashlib.sha256(blob.encode()).hexdigest()
    counts = {
        rel: sum(1 for _ in open(FILES[rel]))
        for rel in entries
        if rel.endswith(".jsonl")
    }
    rec = dict(
        phase="9-stage1-prep",
        set="joint-permutation",
        hash_definition=(
            "sha256 over concatenation of 'relpath:sha256(file)"
            "\\n' lines, sorted by relpath (8A-final freeze "
            "definition, phase8a_final_freeze.py)"
        ),
        files=entries,
        content_hash=h,
        episode_counts=counts,
        n_bases_per_cell=150,
        k=3,
        perms_per_base=6,
        frozen_set_hash="84f2e54d85d6e8aa4c1474b608bef5ab69babe54353ef0ef2702d9f6ed38baef",
    )
    json.dump(rec, open(P9 / "joint_hash.json", "w"), indent=2)
    print("JOINT CONTENT HASH:", h)
    print("episode counts:", counts)
    print("written:", P9 / "joint_hash.json")


if __name__ == "__main__":
    main()

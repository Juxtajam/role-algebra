"""Phase 9 Item 2 (local prep) — content hash over the base-draw-control
(in-place) episode set + its generation code.

Same hash definition as the 8A-final freeze and joint_hash.json
(phase8a_final_freeze.py): sha256 over the concatenation of
'relpath:sha256(file)\\n' lines, sorted by relpath, over:
  tasks_inplace_control/inplace_G_{fit,transfer}.jsonl  (2 episode files)
  tasks_inplace_control/manifest.json  tasks_inplace_control/README.md
  code/phase9_generate_inplace_control.py               (generation code)
  phase8a_generate_tasks.py                             (frozen module it imports)

Writes results/phase9/inplace_control_hash.json.
"""

import hashlib
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[2]
P9 = ROOT / "results/binding_sites"

FILES = {
    "tasks_inplace_control/inplace_G_fit.jsonl": P9
    / "tasks_inplace_control/inplace_G_fit.jsonl",
    "tasks_inplace_control/inplace_G_transfer.jsonl": P9
    / "tasks_inplace_control/inplace_G_transfer.jsonl",
    "tasks_inplace_control/manifest.json": P9 / "tasks_inplace_control/manifest.json",
    "tasks_inplace_control/README.md": P9 / "tasks_inplace_control/README.md",
    "code/phase9_generate_inplace_control.py": P9
    / "code/phase9_generate_inplace_control.py",
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
    joint_hash = json.load(open(P9 / "joint_hash.json"))["content_hash"]
    rec = dict(
        phase="9-item2-prep",
        set="inplace-base-draw-control",
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
        base_source=(
            "same G base problems as the joint set (master seed "
            "20260808, same rng stream; equality asserted per "
            "base_id at generation, see tasks_inplace_control/"
            "manifest.json base_identity_check)"
        ),
        joint_set_hash=joint_hash,
        frozen_set_hash="84f2e54d85d6e8aa4c1474b608bef5ab69babe54353ef0ef2702d9f6ed38baef",
    )
    json.dump(rec, open(P9 / "inplace_control_hash.json", "w"), indent=2)
    print("INPLACE-CONTROL CONTENT HASH:", h)
    print("episode counts:", counts)
    print("written:", P9 / "inplace_control_hash.json")


if __name__ == "__main__":
    main()

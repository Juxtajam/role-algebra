"""Phase 9 STAGE 1 (local prep) — Task 3 driver: compute the pre-registered
position manifest for BOTH episode sets (frozen 84f2e54d... set: gate + disc
files; joint-permutation set) with the finder in phase9_positions.py
(imported — the exact code the 9A session will import).

Writes results/phase9/position_manifest.json:
  - position-class definitions (verbatim from phase9_positions docstring)
  - per-episode position counts (P=11, G=12) and totals per set/file
  - per-episode token indices for every episode of both sets, keyed
    set/file/line, on the unpadded rendered prompt (target tokeniser
    Qwen/Qwen2.5-72B-Instruct @ 495f3936), plus rendered_len for the
    session-side left-padding offset.
"""

import json
import pathlib
import sys

from transformers import AutoTokenizer

ROOT = pathlib.Path(__file__).resolve().parents[2]
P9 = ROOT / "results/binding_sites"
sys.path.insert(0, str(P9 / "code"))
import position_finder as pos  # noqa: E402

TARGET_MODEL = "Qwen/Qwen2.5-72B-Instruct"
TARGET_REVISION = "495f39366efef23836d0cfae4fbe635880d2be31"

FROZEN_FILES = [
    f"{k}_{p}_{v}.jsonl"
    for k in ("gate", "disc")
    for p in ("P", "G")
    for v in ("fit", "transfer")
]
JOINT_FILES = [f"joint_{p}_{v}.jsonl" for p in ("P", "G") for v in ("fit", "transfer")]


def main():
    tok = AutoTokenizer.from_pretrained(TARGET_MODEL, revision=TARGET_REVISION)
    assert tok.is_fast  # offset mapping required

    manifest = dict(
        phase="9-stage1-prep",
        target_tokeniser=dict(model=TARGET_MODEL, revision=TARGET_REVISION),
        finder_code="results/binding_sites/code/phase9_positions.py",
        rendering=(
            "chat template (system+user), add_generation_prompt=True,"
            " enable_thinking=False, + forced 'Answer:' prefix — "
            "identical to phase8a_final_modal.render"
        ),
        position_classes=dict(
            carry_entity=(
                "final token of the entity-mention (person name) "
                "span in each CARRY (holds) clause; k=3 per "
                "episode"
            ),
            fact_final=(
                "final token of each fact clause (token containing "
                "the clause-final '.'); 6 clauses path P, 7 path G"
            ),
            query_arg=(
                "final token of the query-argument mention: {mark} "
                "in path P, {sym} in path G; 1 per episode"
            ),
            answer=(
                "last token of the rendered prompt ('Answer:' forced "
                "prefix) — 8A/8C answer position, continuity; 1 per "
                "episode"
            ),
        ),
        per_episode_position_count=dict(P=11, G=12),
        padding_note=(
            "indices are into the UNPADDED rendered prompt; under "
            "left padding the in-session index = index + "
            "(padded_len - rendered_len)"
        ),
        sets={},
        episodes={},
    )

    grand_positions = 0
    grand_eps = 0
    for set_name, folder, files in (
        ("frozen", ROOT / "results/verdict/gate/tasks", FROZEN_FILES),
        ("joint", P9 / "tasks_joint", JOINT_FILES),
    ):
        set_positions = 0
        set_eps = 0
        file_stats = {}
        for fn in files:
            recs = [json.loads(l) for l in open(folder / fn)]
            n_pos_file = 0
            for i, rec in enumerate(recs):
                p = pos.episode_positions(rec, tok)
                key = f"{set_name}/{fn}/{i}"
                manifest["episodes"][key] = p
                n_pos_file += (
                    len(p["carry_entity"])
                    + len(p["fact_final"])
                    + len(p["query_arg"])
                    + len(p["answer"])
                )
            file_stats[fn] = dict(episodes=len(recs), positions=n_pos_file)
            set_positions += n_pos_file
            set_eps += len(recs)
            print(f"{set_name}/{fn}: {len(recs)} eps, {n_pos_file} positions")
        manifest["sets"][set_name] = dict(
            source=str(folder.relative_to(ROOT)),
            files=file_stats,
            episodes=set_eps,
            positions=set_positions,
        )
        grand_positions += set_positions
        grand_eps += set_eps

    manifest["total_episodes"] = grand_eps
    manifest["total_positions"] = grand_positions
    out = P9 / "position_manifest.json"
    json.dump(manifest, open(out, "w"))
    print(f"TOTAL: {grand_eps} episodes, {grand_positions} positions")
    print("written:", out)


if __name__ == "__main__":
    main()

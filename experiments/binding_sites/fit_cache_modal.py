"""P-only caching session. Launch is manual; the frozen hash is verified
in-session and artifacts are checked on the filesystem before results are read.

Derived from the 9A session script with three changes ONLY:
  1. No behavioural gate step (9A ran it; joint P cells passed 1.000/1.000/
     1.000 both vocabs — phase9/gate/gate_results.json).
  2. File lists restricted to path P: frozen {gate,disc}_P_{fit,transfer}
     (4 files, 600+600+1800+1800 = 4800 eps) + joint_P_{fit,transfer}
     (2 files, 900+900 = 1800 eps). 6600 episodes total, 11 positions each.
  3. Output dir phase9/acts_P/ (checksums.json there).

Everything else (hash verification before model load, render/positions/
layers, memmap streaming, per-file sha256 with re-read verification) is
copied verbatim from phase9a_modal.py.
"""

import hashlib
import json
import os
import sys
import threading
import time

import modal

APP = modal.App("dv3-phase9-item3-cache")
vol = modal.Volume.from_name("dv3-results")
hf_vol = modal.Volume.from_name("hf-models", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.4.1",
        "transformers==4.51.3",
        "accelerate",
        "numpy",
        "safetensors",
        "sentencepiece",
        "hf_transfer",
    )
    .env({"HF_HOME": "/hf", "HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

MODEL = "Qwen/Qwen2.5-72B-Instruct"
REVISION = "495f39366efef23836d0cfae4fbe635880d2be31"
RES = "/results"
FROZEN_HASH = "84f2e54d85d6e8aa4c1474b608bef5ab69babe54353ef0ef2702d9f6ed38baef"
GPU = "A100-80GB:2"
USD_PER_HOUR = 5.00

# frozen-set hash file list (phase8a_final_freeze.py, verbatim from 9A script)
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

# P-ONLY cache lists (change 2)
FROZEN_CACHE_FILES = [
    f"{k}_P_{v}.jsonl" for k in ("gate", "disc") for v in ("fit", "transfer")
]
JOINT_CACHE_FILES = [f"joint_P_{v}.jsonl" for v in ("fit", "transfer")]

ACTS_DIR = "acts_P"


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] MILESTONE phase9item3cache: {msg}"
    print(line, flush=True)
    with open(f"{RES}/progress.log", "a") as f:
        f.write(line + "\n")


def sha_file(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def verify_frozen_hash():
    entries = {}
    for f in TASK_FILES:
        entries[f"tasks/{f}"] = sha_file(f"{RES}/phase8a/tasks/{f}")
    for f in CODE_FILES:
        entries[f] = sha_file(f"{RES}/phase8a/{f}")
    blob = "".join(f"{k}:{v}\n" for k, v in sorted(entries.items()))
    h = hashlib.sha256(blob.encode()).hexdigest()
    assert h == FROZEN_HASH, f"FROZEN HASH MISMATCH: {h} != {FROZEN_HASH}"
    return h


def verify_joint_hash():
    jh = json.load(open(f"{RES}/phase9/joint_hash.json"))
    expected = jh["content_hash"]
    relmap = {
        "tasks_joint/joint_P_fit.jsonl": f"{RES}/phase9/tasks_joint/joint_P_fit.jsonl",
        "tasks_joint/joint_P_transfer.jsonl": f"{RES}/phase9/tasks_joint/joint_P_transfer.jsonl",
        "tasks_joint/joint_G_fit.jsonl": f"{RES}/phase9/tasks_joint/joint_G_fit.jsonl",
        "tasks_joint/joint_G_transfer.jsonl": f"{RES}/phase9/tasks_joint/joint_G_transfer.jsonl",
        "tasks_joint/manifest.json": f"{RES}/phase9/tasks_joint/manifest.json",
        "tasks_joint/README.md": f"{RES}/phase9/tasks_joint/README.md",
        "code/phase9_generate_joint.py": f"{RES}/phase9/code/phase9_generate_joint.py",
        "phase8a_generate_tasks.py": f"{RES}/phase8a/phase8a_generate_tasks.py",
    }
    assert set(relmap) == set(jh["files"]), (
        "joint hash file list mismatch",
        sorted(jh["files"]),
    )
    entries = {rel: sha_file(fp) for rel, fp in relmap.items()}
    blob = "".join(f"{k}:{v}\n" for k, v in sorted(entries.items()))
    h = hashlib.sha256(blob.encode()).hexdigest()
    assert h == expected, f"JOINT HASH MISMATCH: {h} != {expected}"
    return h


@APP.function(
    image=image,
    gpu=GPU,
    volumes={RES: vol, "/hf": hf_vol},
    timeout=10 * 3600,
    memory=131072,
)
def run_session():
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    t_session0 = time.time()
    os.makedirs(f"{RES}/phase9/{ACTS_DIR}", exist_ok=True)
    stop = threading.Event()

    def committer():
        while not stop.wait(120):
            vol.commit()

    threading.Thread(target=committer, daemon=True).start()

    # ---------------- hashes + frozen 9B config, before the model ----------
    hf_ = verify_frozen_hash()
    log(f"frozen content hash VERIFIED: {hf_}")
    hj = verify_joint_hash()
    log(f"joint content hash VERIFIED: {hj}")
    cfg_sha = sha_file(f"{RES}/phase9/item3_committed_config.json")
    cfg_sha_recorded = (
        open(f"{RES}/phase9/item3_committed_config.sha256").read().split()[0]
    )
    assert cfg_sha == cfg_sha_recorded, "9B CONFIG SHA MISMATCH"
    log(f"frozen 9B config present, sha VERIFIED: {cfg_sha}")

    sys.path.insert(0, f"{RES}/phase9/code")
    import position_finder as posmod

    pman = json.load(open(f"{RES}/phase9/position_manifest.json"))
    lset = json.load(open(f"{RES}/phase9/layer_set.json"))
    LAYERS = lset["layers"]
    log(
        f"pre-registered: {lset['count']} layers; caching P-only "
        f"({len(FROZEN_CACHE_FILES)} frozen + {len(JOINT_CACHE_FILES)} joint files)"
    )

    # ---------------- model ----------------
    log(f"loading {MODEL}@{REVISION[:8]} bf16 across {GPU}")
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        revision=REVISION,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map="auto",
        low_cpu_mem_usage=True,
    ).eval()
    hf_vol.commit()
    load_s = time.time() - t0
    n_layers = model.config.num_hidden_layers
    d_model = model.config.hidden_size
    assert d_model == 8192 and n_layers == 80
    assert max(LAYERS) < n_layers
    log(f"model loaded in {load_s:.0f}s: {n_layers} layers, d_model={d_model}")

    def render(rec):
        msgs = [
            {"role": "system", "content": rec["system"]},
            {"role": "user", "content": rec["user"]},
        ]
        s = tok.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        return s + "Answer:"

    def load_file(volpath):
        return [json.loads(l) for l in open(volpath)]

    def in_session_positions(rec, key):
        p_man = pman["episodes"][key]
        p_new = posmod.episode_positions(rec, tok)
        assert p_new == p_man, ("POSITION MISMATCH in-session", key)
        flat = (
            p_man["carry_entity"]
            + p_man["fact_final"]
            + p_man["query_arg"]
            + p_man["answer"]
        )
        return flat, p_man["rendered_len"]

    @torch.no_grad()
    def cache_file(recs, set_name, fn, bs=8):
        stem = fn[: -len(".jsonl")]
        k0 = f"{set_name}/{fn}/0"
        e0 = pman["episodes"][k0]
        n_pos = (
            len(e0["carry_entity"])
            + len(e0["fact_final"])
            + len(e0["query_arg"])
            + len(e0["answer"])
        )
        out_path = f"{RES}/phase9/{ACTS_DIR}/{set_name}_{stem}.npy"
        acts = np.lib.format.open_memmap(
            out_path,
            mode="w+",
            dtype=np.float16,
            shape=(len(recs), n_pos, len(LAYERS), d_model),
        )
        for i in range(0, len(recs), bs):
            chunk = recs[i : i + bs]
            batch = [render(r) for r in chunk]
            enc = tok(batch, return_tensors="pt", padding=True).to(model.device)
            out = model(**enc, output_hidden_states=True, use_cache=False)
            padded_len = enc.input_ids.shape[1]
            hs = torch.stack(
                [out.hidden_states[l + 1] for l in LAYERS], dim=1
            )  # (b, L, seq, d)
            for j, rec in enumerate(chunk):
                key = f"{set_name}/{fn}/{i + j}"
                flat, rlen = in_session_positions(rec, key)
                off = padded_len - rlen
                assert off >= 0
                cols = [off + t for t in flat]
                assert len(cols) == n_pos
                acts[i + j] = (
                    hs[j][:, cols, :].transpose(0, 1).to(torch.float16).cpu().numpy()
                )
            del out, hs
            if (i // bs) % 25 == 0:
                log(f"  acts {i + len(chunk)}/{len(recs)} -> {out_path}")
                acts.flush()
        acts.flush()
        del acts
        vol.commit()
        return out_path, n_pos

    checksums = {}
    for set_name, folder, files in (
        ("frozen", f"{RES}/phase8a/tasks", FROZEN_CACHE_FILES),
        ("joint", f"{RES}/phase9/tasks_joint", JOINT_CACHE_FILES),
    ):
        for fn in files:
            recs = load_file(f"{folder}/{fn}")
            op, n_pos = cache_file(recs, set_name, fn)
            log(
                f"cached {set_name}/{fn}: {len(recs)} eps x {n_pos} pos "
                f"x {len(LAYERS)} layers"
            )

    for fname in sorted(os.listdir(f"{RES}/phase9/{ACTS_DIR}")):
        if not fname.endswith(".npy"):
            continue
        fp = f"{RES}/phase9/{ACTS_DIR}/{fname}"
        h1 = sha_file(fp)
        arr = np.load(fp)
        assert arr.ndim == 4
        assert arr.shape[2] == len(LAYERS) and arr.shape[3] == d_model
        assert arr.dtype == np.float16
        h2 = sha_file(fp)
        assert h1 == h2
        checksums[fname] = dict(sha256=h1, shape=list(arr.shape), dtype=str(arr.dtype))
        del arr
    with open(f"{RES}/phase9/{ACTS_DIR}/checksums.json", "w") as f:
        json.dump(
            dict(
                frozen_hash=hf_,
                joint_hash=hj,
                committed_config_sha=cfg_sha,
                model=MODEL,
                revision=REVISION,
                d_model=d_model,
                layers=LAYERS,
                act_dtype="float16",
                axis_order="(episode, position, layer, d_model)",
                position_order=(
                    "carry_entity + fact_final + " "query_arg + answer, manifest order"
                ),
                files=checksums,
            ),
            f,
            indent=2,
        )
    vol.commit()
    log(f"checksums written + re-read-verified for {len(checksums)} files")

    wall_h = (time.time() - t_session0) / 3600
    meta = dict(
        model=MODEL,
        revision=REVISION,
        gpu=GPU,
        scope="P-only",
        frozen_hash=hf_,
        joint_hash=hj,
        committed_config_sha=cfg_sha,
        cached=True,
        n_layers_cached=len(LAYERS),
        layers=LAYERS,
        d_model=d_model,
        act_dtype="float16",
        model_load_seconds=round(load_s, 1),
        wall_hours=round(wall_h, 3),
        est_cost_usd=round(wall_h * USD_PER_HOUR, 2),
        usd_per_hour=USD_PER_HOUR,
    )
    with open(f"{RES}/phase9/item3_cache_session_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    stop.set()
    vol.commit()
    log(f"session complete: {wall_h:.2f} h, est ${wall_h * USD_PER_HOUR:.2f}")
    return {"checksums_files": len(checksums), "session": meta}


@APP.local_entrypoint()
def main():
    r = run_session.remote()
    print(json.dumps(r, indent=2))

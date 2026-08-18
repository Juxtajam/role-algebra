"""Phase 10 section F — cache 72B attention-output and MLP-output at the
answer position, via forward hooks on the self_attn and mlp sub-modules.

The residual-stream caches (8C/9) hold only resid_post; attn_out and mlp_out
are sub-block tensors (resid_post[l]-resid_post[l-1] = attn_out+mlp_out,
inseparable), so this needs a fresh session. Answer position only, layers
[40,48,56,60,61,64,68,72], cells disc_{P_fit,P_transfer,G_fit}, fp16.

Verifies the frozen content hash in-session (abort on mismatch) before any
capture; writes per-file sha256 + re-read verification. Config frozen at
phase10/nonlinear/committed_config_F.json (sha 03058547...).

Launch:  modal run --detach experiments/phase10_F_cache_modal.py
Output:  dv3-results:phase10/nonlinear/actsF/{cell}_{attn,mlp}.npy + checksums.json
"""

import hashlib
import json
import os
import threading
import time

import modal

APP = modal.App("dv3-phase10-F-cache")
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
LAYERS = [40, 48, 56, 60, 61, 64, 68, 72]
CELLS = ["disc_P_fit", "disc_P_transfer", "disc_G_fit"]
ACTS_DIR = "phase10/nonlinear/actsF"

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


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] MILESTONE phase10Fcache: {msg}"
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


@APP.function(
    image=image,
    gpu=GPU,
    volumes={RES: vol, "/hf": hf_vol},
    timeout=6 * 3600,
    memory=131072,
)
def run_session():
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    t0s = time.time()
    os.makedirs(f"{RES}/{ACTS_DIR}", exist_ok=True)
    stop = threading.Event()
    threading.Thread(
        target=lambda: [vol.commit() for _ in iter(lambda: stop.wait(120), True)],
        daemon=True,
    ).start()

    hf_ = verify_frozen_hash()
    log(f"frozen content hash VERIFIED: {hf_}")
    cfg_sha = sha_file(f"{RES}/phase10/nonlinear/committed_config_F.json")
    log(f"section-F config sha: {cfg_sha}")

    log(f"loading {MODEL}@{REVISION[:8]} bf16 on {GPU}")
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
    n_layers, d_model = model.config.num_hidden_layers, model.config.hidden_size
    assert d_model == 8192 and n_layers == 80 and max(LAYERS) < n_layers
    log(f"model loaded in {time.time()-t0:.0f}s: {n_layers}L d={d_model}")

    # ---- forward hooks on self_attn and mlp of the target layers ----
    grabbed = {}

    def mk_hook(kind, li):
        def hook(mod, inp, out):
            t = out[0] if isinstance(out, tuple) else out
            grabbed[(kind, li)] = t.detach()

        return hook

    handles = []
    layer_mods = model.model.layers
    for li in LAYERS:
        handles.append(
            layer_mods[li].self_attn.register_forward_hook(mk_hook("attn", li))
        )
        handles.append(layer_mods[li].mlp.register_forward_hook(mk_hook("mlp", li)))

    def render(rec):
        msgs = [
            {"role": "system", "content": rec["system"]},
            {"role": "user", "content": rec["user"]},
        ]
        s = tok.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        return s + "Answer:"

    @torch.no_grad()
    def cache_cell(cell, bs=8):
        recs = [json.loads(l) for l in open(f"{RES}/phase8a/tasks/{cell}.jsonl")]
        n = len(recs)
        attn = np.lib.format.open_memmap(
            f"{RES}/{ACTS_DIR}/{cell}_attn.npy",
            mode="w+",
            dtype=np.float16,
            shape=(n, len(LAYERS), d_model),
        )
        mlp = np.lib.format.open_memmap(
            f"{RES}/{ACTS_DIR}/{cell}_mlp.npy",
            mode="w+",
            dtype=np.float16,
            shape=(n, len(LAYERS), d_model),
        )
        for i in range(0, n, bs):
            chunk = recs[i : i + bs]
            enc = tok([render(r) for r in chunk], return_tensors="pt", padding=True).to(
                model.device
            )
            grabbed.clear()
            model(**enc, use_cache=False)
            # answer position = last column (left padding -> real tokens at right)
            for slot, kind in ((attn, "attn"), (mlp, "mlp")):
                st = torch.stack(
                    [grabbed[(kind, li)][:, -1, :] for li in LAYERS], dim=1
                )  # (b, L, d)
                slot[i : i + len(chunk)] = st.to(torch.float16).cpu().numpy()
            if (i // bs) % 25 == 0:
                log(f"  {cell} {i+len(chunk)}/{n}")
                attn.flush()
                mlp.flush()
        attn.flush()
        mlp.flush()
        del attn, mlp
        vol.commit()

    for cell in CELLS:
        t1 = time.time()
        cache_cell(cell)
        log(f"cached {cell} in {time.time()-t1:.0f}s")
    for h in handles:
        h.remove()

    # ---- checksums + re-read verification ----
    checks = {}
    for cell in CELLS:
        for kind in ("attn", "mlp"):
            fp = f"{RES}/{ACTS_DIR}/{cell}_{kind}.npy"
            arr = np.load(fp, mmap_mode="r")
            h1 = sha_file(fp)
            checks[f"{cell}_{kind}"] = dict(sha256=h1, shape=list(arr.shape))
    meta = dict(
        model=MODEL,
        revision=REVISION,
        gpu=GPU,
        layers=LAYERS,
        cells=CELLS,
        position="answer(last col, left pad)",
        d_model=d_model,
        act_dtype="float16",
        objects=["attn_out(self_attn output)", "mlp_out(mlp output)"],
        frozen_hash=hf_,
        config_F_sha=cfg_sha,
        session_seconds=round(time.time() - t0s, 1),
        files=checks,
    )
    with open(f"{RES}/{ACTS_DIR}/checksums.json", "w") as f:
        json.dump(meta, f, indent=2)
    vol.commit()
    log(f"checksums written for {len(checks)} files; session {time.time()-t0s:.0f}s")
    stop.set()
    return meta


@APP.local_entrypoint()
def main():
    print("Launching section-F cache session (attn_out + mlp_out @ answer position)...")
    meta = run_session.remote()
    print(json.dumps(meta, indent=1)[:600])
    print("MILESTONE: section-F cache COMPLETE")

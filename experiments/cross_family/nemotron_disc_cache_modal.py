"""Track A cross-family MECHANISM — cache Nemotron-70B answer-position
activations on the frozen k=3 disc episodes, to run the 8C discriminator on a
second family. Nemotron passed the k=3 behavioural gate (E17, after D10), so
it is eligible under the gate-first rule.

Verifies the frozen content hash 84f2e54d in-session (abort on mismatch),
renders with the Llama chat template (+ "Answer:", D3 BOS handling native),
generates disc predictions with the D10 normalisation (strip markdown *), and
caches resid_post at the answer position for the 8C reporting layers, all four
disc cells (P/G x fit/transfer) -> C1 and C2 both available.

Hardware: 2xH100-80GB (operator directive 2026-08-15). 141 GB bf16 needs
2x80 GB; H100 throughput chosen for a caching-heavy session (7200 disc
episodes x generate + cache across 12 layers).

Launch: modal run --detach phase10/trackA/nemotron_disc_cache_modal.py
"""

import hashlib
import json
import os
import threading
import time

import modal

APP = modal.App("dv3-trackA-nemotron-disc-cache")
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

MODEL = "nvidia/Llama-3.1-Nemotron-70B-Instruct-HF"
REVISION = "031d4042f36adc1a52cca51b331d25cbe3cf1022"
RES = "/results"
FROZEN_HASH = "84f2e54d85d6e8aa4c1474b608bef5ab69babe54353ef0ef2702d9f6ed38baef"
GPU = "H100:2"
REPORT_LAYERS = [0, 8, 16, 24, 32, 40, 48, 56, 61, 64, 72, 79]
DISC_CELLS = [("P", "fit"), ("P", "transfer"), ("G", "fit"), ("G", "transfer")]
ACTS_DIR = "phase10/trackA/nemotron_acts"

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


def log(m):
    line = f"[{time.strftime('%H:%M:%S')}] MILESTONE nemodisc: {m}"
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
    assert h == FROZEN_HASH, f"FROZEN HASH MISMATCH {h}"
    return h


def orbit_metrics(recs, preds):
    import collections
    import numpy as np

    bybase = collections.defaultdict(list)
    for r, p in zip(recs, preds):
        bybase[r["base_id"]].append((r, p))
    ep, strict = [], []
    for _, items in sorted(bybase.items()):
        oks = [p == r["answer"] for r, p in items]
        ep.extend(oks)
        strict.append(all(oks))
    return float(np.mean(ep)), float(np.mean(strict))


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

    log(f"loading {MODEL}@{REVISION[:8]} bf16 on {GPU}")
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        revision=REVISION,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map="auto",
        low_cpu_mem_usage=True,
    ).eval()
    hf_vol.commit()
    n_layers, d = model.config.num_hidden_layers, model.config.hidden_size
    log(f"loaded in {time.time()-t0:.0f}s: {n_layers}L d={d}")
    assert max(REPORT_LAYERS) < n_layers

    def render(rec):
        msgs = [
            {"role": "system", "content": rec["system"]},
            {"role": "user", "content": rec["user"]},
        ]
        return (
            tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            + "Answer:"
        )

    def load(kind, path, vocab):
        return [
            json.loads(l)
            for l in open(f"{RES}/phase8a/tasks/{kind}_{path}_{vocab}.jsonl")
        ]

    @torch.no_grad()
    def generate(recs, bs=24):
        preds = []
        for i in range(0, len(recs), bs):
            batch = [render(r) for r in recs[i : i + bs]]
            enc = tok(batch, return_tensors="pt", padding=True).to(model.device)
            out = model.generate(
                **enc,
                max_new_tokens=8,
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
                pad_token_id=tok.pad_token_id,
            )
            for j in range(len(batch)):
                text = tok.decode(
                    out[j, enc.input_ids.shape[1] :], skip_special_tokens=True
                )
                w = text.strip().split()
                preds.append(w[0].strip(".,!*") if w else "")  # D10 normalisation
            if (i // bs) % 20 == 0:
                log(f"  gen {i+len(batch)}/{len(recs)}")
        return preds

    @torch.no_grad()
    def cache(recs, out_path, bs=8):
        acts = np.lib.format.open_memmap(
            out_path,
            mode="w+",
            dtype=np.float16,
            shape=(len(recs), len(REPORT_LAYERS), d),
        )
        for i in range(0, len(recs), bs):
            batch = [render(r) for r in recs[i : i + bs]]
            enc = tok(batch, return_tensors="pt", padding=True).to(model.device)
            out = model(**enc, output_hidden_states=True, use_cache=False)
            hs = torch.stack([out.hidden_states[l + 1] for l in REPORT_LAYERS], dim=1)
            acts[i : i + len(batch)] = hs[:, :, -1, :].to(torch.float16).cpu().numpy()
            if (i // bs) % 50 == 0:
                log(f"  acts {i+len(batch)}/{len(recs)} -> {out_path.split('/')[-1]}")
                acts.flush()
        acts.flush()
        del acts
        vol.commit()

    checks = {}
    for path, vocab in DISC_CELLS:
        recs = load("disc", path, vocab)
        preds = generate(recs)
        json.dump(preds, open(f"{RES}/{ACTS_DIR}/preds_disc_{path}_{vocab}.json", "w"))
        acc, strict = orbit_metrics(recs, preds)
        log(f"disc {path}/{vocab}: acc={acc:.4f} strict={strict:.4f}")
        op = f"{RES}/{ACTS_DIR}/disc_{path}_{vocab}.npy"
        cache(recs, op)
        arr = np.load(op, mmap_mode="r")
        checks[f"disc_{path}_{vocab}"] = dict(
            sha256=sha_file(op),
            shape=list(arr.shape),
            episode_acc=acc,
            strict_orbit=strict,
        )
        vol.commit()
    meta = dict(
        model=MODEL,
        revision=REVISION,
        gpu=GPU,
        k=3,
        layers=REPORT_LAYERS,
        position="answer(last col, left pad)",
        d_model=d,
        frozen_hash=hf_,
        session_seconds=round(time.time() - t0s, 1),
        files=checks,
    )
    json.dump(meta, open(f"{RES}/{ACTS_DIR}/checksums.json", "w"), indent=2)
    vol.commit()
    log(f"cached {len(checks)} disc cells; session {time.time()-t0s:.0f}s")
    stop.set()
    return meta


@APP.local_entrypoint()
def main():
    print("Launching Nemotron disc caching (2xH100)...")
    m = run_session.remote()
    print(json.dumps(m, indent=1)[:800])
    print("MILESTONE: Nemotron disc cache COMPLETE")

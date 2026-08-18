"""Track C — k=4 (S_4) behavioural gate on Qwen2.5-72B, + conditional disc
activation caching (answer position, 12 reporting layers) if the gate passes.

Verifies the k=4 content hash in-session (abort on mismatch) before generating.
Gate thresholds are the frozen k=3 bars (episode acc >= 0.95, strict-orbit
>= 0.90, orbit-consistency >= 0.95) — noting strict-orbit is now over 24
permutations, an inherently harder bar; raw numbers reported regardless.

Launch:  modal run --detach phase10/trackC/k4_gate_modal.py
Outputs: dv3-results:phase10/trackC/{gate_results.json, preds_*.json,
         actsF? no -> acts_k4/{disc_*}_answer.npy, checksums.json}
"""

import hashlib
import json
import os
import threading
import time

import modal

APP = modal.App("dv3-trackC-k4-gate")
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
TASKS = f"{RES}/phase10/trackC/tasks_k4"
GPU = "A100-80GB:2"
K = 4
GATE = dict(episode_acc=0.95, strict_orbit=0.90, orbit_consistency=0.95)
REPORT_LAYERS = [0, 8, 16, 24, 32, 40, 48, 56, 61, 64, 72, 79]
DISC_CELLS = [("P", "fit"), ("P", "transfer"), ("G", "fit")]
CELLS = [("P", "fit"), ("P", "transfer"), ("G", "fit"), ("G", "transfer")]


def log(m):
    line = f"[{time.strftime('%H:%M:%S')}] MILESTONE trackCk4: {m}"
    print(line, flush=True)
    with open(f"{RES}/progress.log", "a") as f:
        f.write(line + "\n")


def sha_file(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def verify_content_hash():
    man = json.load(open(f"{TASKS}/manifest.json"))
    fh = {}
    for name in man["file_sha256"]:
        path = (
            f"{RES}/phase10/trackC/generate_k4.py"
            if name == "generate_k4.py"
            else f"{TASKS}/{name}"
        )
        fh[name] = sha_file(path)
    blob = "".join(f"{k}:{v}\n" for k, v in sorted(fh.items()))
    h = hashlib.sha256(blob.encode()).hexdigest()
    assert h == man["content_hash"], f"K4 HASH MISMATCH {h} != {man['content_hash']}"
    return h


def orbit_metrics(recs, preds):
    import collections
    import numpy as np

    bybase = collections.defaultdict(list)
    for r, p in zip(recs, preds):
        bybase[r["base_id"]].append((r, p))
    ep, strict, cons = [], [], []
    for _, items in sorted(bybase.items()):
        oks = [p == r["answer"] for r, p in items]
        ep.extend(oks)
        strict.append(all(oks))
        names = items[0][0]["base"]["names"]
        cons.append(
            any(all(p == names[r["g"][s]] for r, p in items) for s in range(K))
        )  # K-aware
    return float(np.mean(ep)), float(np.mean(strict)), float(np.mean(cons))


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
    stop = threading.Event()
    threading.Thread(
        target=lambda: [vol.commit() for _ in iter(lambda: stop.wait(120), True)],
        daemon=True,
    ).start()

    ch = verify_content_hash()
    log(f"k=4 content hash VERIFIED: {ch}")

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
    n_layers, d = model.config.num_hidden_layers, model.config.hidden_size
    log(f"loaded in {time.time()-t0:.0f}s: {n_layers}L d={d}")

    def render(rec):
        s = tok.apply_chat_template(
            [
                {"role": "system", "content": rec["system"]},
                {"role": "user", "content": rec["user"]},
            ],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        return s + "Answer:"

    def load(kind, path, vocab):
        return [json.loads(l) for l in open(f"{TASKS}/{kind}_{path}_{vocab}.jsonl")]

    @torch.no_grad()
    def generate(recs, bs=24):
        preds = []
        for i in range(0, len(recs), bs):
            batch = [render(r) for r in recs[i : i + bs]]
            enc = tok(batch, return_tensors="pt", padding=True).to("cuda")
            out = model.generate(
                **enc,
                max_new_tokens=8,
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
                pad_token_id=tok.pad_token_id or tok.eos_token_id,
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
            enc = tok(batch, return_tensors="pt", padding=True).to("cuda")
            out = model(**enc, output_hidden_states=True, use_cache=False)
            hs = torch.stack([out.hidden_states[l + 1] for l in REPORT_LAYERS], dim=1)
            acts[i : i + len(batch)] = hs[:, :, -1, :].to(torch.float16).cpu().numpy()
            if (i // bs) % 50 == 0:
                log(f"  acts {i+len(batch)}/{len(recs)} -> {out_path}")
                acts.flush()
        acts.flush()
        del acts
        vol.commit()

    # ---- gate ----
    os.makedirs(f"{RES}/phase10/trackC", exist_ok=True)
    results = {
        "model": MODEL,
        "revision": REVISION,
        "k": K,
        "content_hash": ch,
        "gate_thresholds": GATE,
        "strict_orbit_note": "over 24 permutations (S_4)",
        "cells": {},
    }
    all_pass = True
    for path, vocab in CELLS:
        recs = load("gate", path, vocab)
        t1 = time.time()
        preds = generate(recs)
        json.dump(
            preds, open(f"{RES}/phase10/trackC/preds_gate_{path}_{vocab}.json", "w")
        )
        acc, strict, cons = orbit_metrics(recs, preds)
        passed = (
            acc >= GATE["episode_acc"]
            and strict >= GATE["strict_orbit"]
            and cons >= GATE["orbit_consistency"]
        )
        all_pass = all_pass and passed
        results["cells"][f"{path}/{vocab}"] = dict(
            episode_acc=acc,
            strict_orbit=strict,
            orbit_consistency=cons,
            passed=passed,
            n_episodes=len(recs),
            seconds=round(time.time() - t1, 1),
        )
        log(
            f"GATE {path}/{vocab}: acc={acc:.4f} strict={strict:.4f} "
            f"cons={cons:.4f} -> {'PASS' if passed else 'FAIL'}"
        )
        vol.commit()
    results["gate_all_pass"] = all_pass
    json.dump(results, open(f"{RES}/phase10/trackC/gate_results.json", "w"), indent=2)

    # ---- conditional disc caching ----
    if all_pass:
        log("gate PASSED -> caching disc activations (answer pos, 12 layers)")
        os.makedirs(f"{RES}/phase10/trackC/acts_k4", exist_ok=True)
        checks = {}
        for path, vocab in DISC_CELLS:
            recs = load("disc", path, vocab)
            preds = generate(recs)
            json.dump(
                preds, open(f"{RES}/phase10/trackC/preds_disc_{path}_{vocab}.json", "w")
            )
            acc, strict, cons = orbit_metrics(recs, preds)
            log(f"disc {path}/{vocab}: acc={acc:.4f} strict={strict:.4f}")
            op = f"{RES}/phase10/trackC/acts_k4/disc_{path}_{vocab}.npy"
            cache(recs, op)
            arr = np.load(op, mmap_mode="r")
            checks[f"disc_{path}_{vocab}"] = dict(
                sha256=sha_file(op),
                shape=list(arr.shape),
                episode_acc=acc,
                strict_orbit=strict,
            )
        meta = dict(
            model=MODEL,
            revision=REVISION,
            k=K,
            layers=REPORT_LAYERS,
            position="answer(last col, left pad)",
            d_model=d,
            content_hash=ch,
            files=checks,
        )
        json.dump(
            meta, open(f"{RES}/phase10/trackC/acts_k4/checksums.json", "w"), indent=2
        )
        vol.commit()
        log(f"disc caching done ({len(checks)} files)")
    else:
        log("gate did NOT pass all cells -> no disc caching (report the gate table)")

    log(f"session done in {time.time()-t0s:.0f}s; all_pass={all_pass}")
    stop.set()
    return results


@APP.local_entrypoint()
def main():
    print("Launching k=4 gate on Qwen2.5-72B...")
    r = run_session.remote()
    print(json.dumps(r, indent=1)[:1200])
    print("MILESTONE: k=4 gate session COMPLETE")

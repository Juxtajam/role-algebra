"""Base-draw control, single rented Modal session. Launch is manual.

Evaluate in-place orbits over the same new G base problems used in the joint
set (both vocabularies), identical thresholds and bootstrap protocol. One cell
pair. Write phase9/gate/inplace_control.json. This determines whether the
joint-set consistency drop is permutation-driven or base-sampling.

Model: Qwen/Qwen2.5-72B-Instruct @ 495f3936, bf16, 2x A100-80GB — identical
to 9A / 8A-final.

Harness reused VERBATIM from phase9a_modal.py: log/sha_file, in-session
content-hash verification before the model loads, frozen scoring
(orbit_metrics + GATE loaded from the frozen phase8a_modal.py volume bytes
via importlib), boot_cis (bootstrap 95% CIs over bases, 10,000 resamples,
seed 20260807, copied verbatim from phase8a_final_modal.py via
phase9a_modal.py), render/generate (chat template, add_generation_prompt,
enable_thinking=False, forced "Answer:" prefix, greedy, max_new_tokens=8,
first whitespace-delimited word, exact match), thresholds 0.95/0.90/0.95.

Session order (hard-coded):
  1. Re-verify the CONTROL-SET content hash against the volume bytes
     (results/phase9/inplace_control_hash.json definition + file list;
     the volume must hold phase9/tasks_inplace_control/*,
     phase9/code/phase9_generate_inplace_control.py and the frozen
     phase8a/phase8a_generate_tasks.py before launch). Any mismatch aborts
     before the model loads.
  2. Evaluate ONLY the two control cells (inplace_control G/fit,
     G/transfer): greedy decode, frozen thresholds, n=150 bases per cell,
     full six-member orbits, bootstrap CIs — protocol identical to 9A.
  3. Write gate table to phase9/gate/inplace_control.json and per-episode
     predictions to phase9/gate/preds_inplace_control_G_{fit,transfer}.json
     + phase9/gate/episodes_inplace_control_G_{fit,transfer}.csv, committed
     to the volume BEFORE the session exits. NO activation caching in this
     session.
"""

import csv
import hashlib
import importlib.util
import json
import os
import threading
import time

import modal

APP = modal.App("dv3-phase9-item2")
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
CELLS = [("G", "fit"), ("G", "transfer")]  # one cell pair, per instruction
N_BOOT = 10_000
BOOT_SEED = 20260807
GPU = "A100-80GB:2"
USD_PER_HOUR = 5.00


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] MILESTONE phase9item2: {msg}"
    print(line, flush=True)
    with open(f"{RES}/progress.log", "a") as f:
        f.write(line + "\n")


def sha_file(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def verify_control_hash():
    """inplace_control_hash.json definition, volume bytes under phase9/."""
    ch = json.load(open(f"{RES}/phase9/inplace_control_hash.json"))
    expected = ch["content_hash"]
    relmap = {
        # relpaths in inplace_control_hash.json -> volume locations
        "tasks_inplace_control/inplace_G_fit.jsonl": f"{RES}/phase9/tasks_inplace_control/inplace_G_fit.jsonl",
        "tasks_inplace_control/inplace_G_transfer.jsonl": f"{RES}/phase9/tasks_inplace_control/inplace_G_transfer.jsonl",
        "tasks_inplace_control/manifest.json": f"{RES}/phase9/tasks_inplace_control/manifest.json",
        "tasks_inplace_control/README.md": f"{RES}/phase9/tasks_inplace_control/README.md",
        "code/phase9_generate_inplace_control.py": f"{RES}/phase9/code/phase9_generate_inplace_control.py",
        "phase8a_generate_tasks.py": f"{RES}/phase8a/phase8a_generate_tasks.py",
    }
    assert set(relmap) == set(ch["files"]), (
        "control hash file list mismatch",
        sorted(ch["files"]),
    )
    entries = {rel: sha_file(fp) for rel, fp in relmap.items()}
    blob = "".join(f"{k}:{v}\n" for k, v in sorted(entries.items()))
    h = hashlib.sha256(blob.encode()).hexdigest()
    assert h == expected, f"CONTROL HASH MISMATCH: {h} != {expected}"
    return h


def load_frozen_scoring():
    """orbit_metrics + GATE from the frozen phase8a_modal.py volume bytes
    (identical mechanism to phase9a_modal.py / phase8a_final_modal.py)."""
    spec = importlib.util.spec_from_file_location(
        "frozen_phase8a_modal", f"{RES}/phase8a/phase8a_modal.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.orbit_metrics, mod.GATE


def boot_cis(recs, preds, n_boot=N_BOOT, seed=BOOT_SEED):
    """COPIED VERBATIM from phase9a_modal.py (itself verbatim from
    phase8a_final_modal.py): bootstrap 95% CIs over base problems."""
    import collections
    import numpy as np

    bybase = collections.defaultdict(list)
    for r, p in zip(recs, preds):
        bybase[r["base_id"]].append((r, p))
    bases = sorted(bybase)
    ep = np.array([np.mean([p == r["answer"] for r, p in bybase[b]]) for b in bases])
    st = np.array(
        [all(p == r["answer"] for r, p in bybase[b]) for b in bases], dtype=float
    )
    cons = []
    for b in bases:
        items = bybase[b]
        names = items[0][0]["base"]["names"]
        cons.append(any(all(p == names[r["g"][s]] for r, p in items) for s in range(3)))
    co = np.array(cons, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(bases), size=(n_boot, len(bases)))
    out = {}
    for key, arr in (
        ("episode_acc", ep),
        ("strict_orbit", st),
        ("orbit_consistency", co),
    ):
        stats = arr[idx].mean(axis=1)
        out[key] = [float(np.percentile(stats, q)) for q in (2.5, 97.5)]
    return out


@APP.function(
    image=image,
    gpu=GPU,
    volumes={RES: vol, "/hf": hf_vol},
    timeout=4 * 3600,
    memory=131072,
)
def run_session():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    t_session0 = time.time()
    os.makedirs(f"{RES}/phase9/gate", exist_ok=True)
    stop = threading.Event()

    def committer():
        while not stop.wait(120):
            vol.commit()

    threading.Thread(target=committer, daemon=True).start()

    # ---------------- step 1: control-set hash, before the model -----------
    hc = verify_control_hash()
    log(f"inplace-control content hash VERIFIED: {hc}")
    orbit_metrics, GATE = load_frozen_scoring()
    log(f"frozen scoring loaded: thresholds={GATE}")

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
    log(
        f"model loaded in {load_s:.0f}s: {n_layers} layers, "
        f"d_model={d_model}, dtype=bfloat16"
    )

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
                pad_token_id=tok.pad_token_id or tok.eos_token_id,
            )
            for j in range(len(batch)):
                text = tok.decode(
                    out[j, enc.input_ids.shape[1] :], skip_special_tokens=True
                )
                words = text.strip().split()
                preds.append(words[0].strip(".,!") if words else "")
            if (i // bs) % 5 == 0:
                log(f"  gen {i + len(batch)}/{len(recs)}")
        return preds

    # ------------- step 2: the two control cells, nothing else -------------
    gate_res = {
        "model": MODEL,
        "revision": REVISION,
        "gpu": GPU,
        "control_hash": hc,
        "gate_thresholds": GATE,
        "n_boot": N_BOOT,
        "boot_seed": BOOT_SEED,
        "set": "inplace-base-draw-control",
        "purpose": (
            "in-place orbits over the SAME new G base "
            "problems as the joint set (both vocabularies); "
            "comparison/interpretation "
            "happens offline"
        ),
        "cells": {},
    }
    all_pass = True
    for path, vocab in CELLS:
        recs = load_file(
            f"{RES}/phase9/tasks_inplace_control/" f"inplace_{path}_{vocab}.jsonl"
        )
        assert len(recs) == 900 and len(recs) // 6 == 150
        t0 = time.time()
        preds = generate(recs)
        acc, strict, cons = orbit_metrics(recs, preds)
        cis = boot_cis(recs, preds)
        passed = (
            acc >= GATE["episode_acc"]
            and strict >= GATE["strict_orbit"]
            and cons >= GATE["orbit_consistency"]
        )
        all_pass &= passed
        gate_res["cells"][f"{path}/{vocab}"] = dict(
            episode_acc=acc,
            strict_orbit=strict,
            orbit_consistency=cons,
            cis=cis,
            passed=bool(passed),
            n_bases=len(recs) // 6,
            n_eps=len(recs),
            seconds=round(time.time() - t0, 1),
        )
        with open(
            f"{RES}/phase9/gate/" f"preds_inplace_control_{path}_{vocab}.json", "w"
        ) as f:
            json.dump(preds, f)
        # per-episode records
        with open(
            f"{RES}/phase9/gate/" f"episodes_inplace_control_{path}_{vocab}.csv",
            "w",
            newline="",
        ) as f:
            w = csv.writer(f)
            w.writerow(["base_id", "ep_idx", "g", "answer", "pred", "correct"])
            for k, (r, p) in enumerate(zip(recs, preds)):
                w.writerow(
                    [
                        r["base_id"],
                        k,
                        "".join(map(str, r["g"])),
                        r["answer"],
                        p,
                        int(p == r["answer"]),
                    ]
                )
        vol.commit()
        log(
            f"CONTROL GATE {path}/{vocab}: acc={acc:.4f} "
            f"CI=[{cis['episode_acc'][0]:.3f},{cis['episode_acc'][1]:.3f}] "
            f"strict={strict:.4f} consist={cons:.4f} "
            f"threshold_met={passed}"
        )

    gate_res["all_cells_met_thresholds"] = bool(all_pass)
    wall_h = (time.time() - t_session0) / 3600
    gate_res["session"] = dict(
        model_load_seconds=round(load_s, 1),
        wall_hours=round(wall_h, 3),
        est_cost_usd=round(wall_h * USD_PER_HOUR, 2),
        usd_per_hour=USD_PER_HOUR,
    )
    with open(f"{RES}/phase9/gate/inplace_control.json", "w") as f:
        json.dump(gate_res, f, indent=2)
    stop.set()
    vol.commit()
    log(
        f"control gate table + preds committed to phase9/gate/ "
        f"(inplace_control.json); session complete: {wall_h:.2f} h, "
        f"est ${wall_h * USD_PER_HOUR:.2f}"
    )
    return gate_res


@APP.local_entrypoint()
def main():
    r = run_session.remote()
    print(json.dumps(r, indent=2))

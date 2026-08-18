"""Phase 9A caching session, single rented Modal session. Launch is manual.

Model: Qwen/Qwen2.5-72B-Instruct @ 495f3936, bf16, 2x A100-80GB — identical
to 8A-final.

Session order (hard-coded):
  1. Re-verify BOTH content hashes against the volume bytes:
       frozen set  84f2e54d... (8A-final definition + file list)
       joint set   (results/phase9/joint_hash.json definition + file list)
     Any mismatch aborts before the model loads.
  2. JOINT-PERMUTATION BEHAVIOURAL GATE FIRST, all 4 joint cells:
     greedy decode, frozen thresholds 0.95/0.90/0.95, n=150 bases per cell,
     full six-member orbits, bootstrap 95% CIs over bases (10,000 resamples,
     seed 20260807) — protocol identical to 8A-final (orbit_metrics + GATE
     loaded from the frozen phase8a_modal.py bytes on the volume via
     importlib; boot_cis copied verbatim from phase8a_final_modal.py).
     Gate results + per-episode predictions written to the volume
     (phase9/gate/) and committed BEFORE any caching.
  3. If ANY cell fails: NO caching. Behavioural
     non-equivariance under joint permutation is itself the finding —
     predictions and metrics are stored, the session ends.
  4. If all cells pass: cache resid_post fp16 at the PRE-REGISTERED
     positions (phase9/position_manifest.json, indices re-verified
     in-session against the finder code phase9/code/phase9_positions.py)
     x the PRE-REGISTERED layers (phase9/layer_set.json), for BOTH sets
     (frozen 12 files + joint 4 files), streamed to the volume under
     phase9/acts/ as one .npy per episode file, shape
     (n_eps, n_positions_per_ep, n_layers, 8192). Per-file sha256 written
     to phase9/acts/checksums.json after an in-session RE-READ verification
     (re-load, shape check, re-hash stable).

Conventions identical to 8A-final: chat template (system+user),
add_generation_prompt=True, enable_thinking=False (no-op on Qwen2.5
template), forced "Answer:" prefix; greedy, max_new_tokens=8, first
whitespace-delimited word, exact match; hidden_states[l+1] = resid_post of
layer l; left padding, unpadded position index + (padded_len - unpadded_len).
"""

import csv
import hashlib
import importlib.util
import json
import os
import sys
import threading
import time

import modal

APP = modal.App("dv3-phase9a")
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
CELLS = [(p, v) for p in ("P", "G") for v in ("fit", "transfer")]
N_BOOT = 10_000
BOOT_SEED = 20260807
GPU = "A100-80GB:2"
USD_PER_HOUR = 5.00

# frozen-set hash file list (phase8a_final_freeze.py, verbatim)
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

# frozen episode files to cache (relative to phase8a/tasks/)
FROZEN_CACHE_FILES = [
    f"{k}_{p}_{v}.jsonl"
    for k in ("gate", "disc")
    for p in ("P", "G")
    for v in ("fit", "transfer")
]
JOINT_CACHE_FILES = [
    f"joint_{p}_{v}.jsonl" for p in ("P", "G") for v in ("fit", "transfer")
]


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] MILESTONE phase9a: {msg}"
    print(line, flush=True)
    with open(f"{RES}/progress.log", "a") as f:
        f.write(line + "\n")


def sha_file(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def verify_frozen_hash():
    """8A-final definition, volume bytes."""
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
    """joint_hash.json definition, volume bytes under phase9/."""
    jh = json.load(open(f"{RES}/phase9/joint_hash.json"))
    expected = jh["content_hash"]
    relmap = {
        # relpaths in joint_hash.json -> volume locations
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


def load_frozen_scoring():
    """orbit_metrics + GATE from the frozen phase8a_modal.py volume bytes
    (identical mechanism to phase8a_final_modal.py)."""
    spec = importlib.util.spec_from_file_location(
        "frozen_phase8a_modal", f"{RES}/phase8a/phase8a_modal.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.orbit_metrics, mod.GATE


def boot_cis(recs, preds, n_boot=N_BOOT, seed=BOOT_SEED):
    """COPIED VERBATIM from phase8a_final_modal.py (8A-final protocol):
    bootstrap 95% CIs over base problems."""
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
    timeout=10 * 3600,
    memory=131072,
)
def run_session():
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    t_session0 = time.time()
    os.makedirs(f"{RES}/phase9/gate", exist_ok=True)
    stop = threading.Event()

    def committer():
        while not stop.wait(120):
            vol.commit()

    threading.Thread(target=committer, daemon=True).start()

    # ---------------- step 1: BOTH hashes, before the model ----------------
    hf_ = verify_frozen_hash()
    log(f"frozen content hash VERIFIED: {hf_}")
    hj = verify_joint_hash()
    log(f"joint content hash VERIFIED: {hj}")
    orbit_metrics, GATE = load_frozen_scoring()
    log(f"frozen scoring loaded: thresholds={GATE}")

    # pre-registered positions + layers (prep artifacts, volume bytes)
    sys.path.insert(0, f"{RES}/phase9/code")
    import position_finder as posmod

    pman = json.load(open(f"{RES}/phase9/position_manifest.json"))
    lset = json.load(open(f"{RES}/phase9/layer_set.json"))
    LAYERS = lset["layers"]
    log(
        f"pre-registered: {lset['count']} layers, "
        f"{pman['total_positions']} positions over "
        f"{pman['total_episodes']} episodes"
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

    # ------------- step 2: JOINT BEHAVIOURAL GATE, before any caching ------
    gate_res = {
        "model": MODEL,
        "revision": REVISION,
        "gpu": GPU,
        "frozen_hash": hf_,
        "joint_hash": hj,
        "gate_thresholds": GATE,
        "n_boot": N_BOOT,
        "boot_seed": BOOT_SEED,
        "set": "joint-permutation",
        "cells": {},
    }
    all_pass = True
    for path, vocab in CELLS:
        recs = load_file(f"{RES}/phase9/tasks_joint/joint_{path}_{vocab}.jsonl")
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
        with open(f"{RES}/phase9/gate/preds_joint_{path}_{vocab}.json", "w") as f:
            json.dump(preds, f)
        # per-episode records
        with open(
            f"{RES}/phase9/gate/episodes_joint_{path}_{vocab}.csv", "w", newline=""
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
            f"JOINT GATE {path}/{vocab}: acc={acc:.4f} "
            f"CI=[{cis['episode_acc'][0]:.3f},{cis['episode_acc'][1]:.3f}] "
            f"strict={strict:.4f} consist={cons:.4f} "
            f"threshold_met={passed}"
        )

    gate_res["all_cells_met_thresholds"] = bool(all_pass)
    with open(f"{RES}/phase9/gate/gate_results.json", "w") as f:
        json.dump(gate_res, f, indent=2)
    vol.commit()
    log(
        f"JOINT GATE all-cells-met-thresholds: {all_pass} "
        f"(gate results + preds committed to phase9/gate/)"
    )

    # ------------- step 3: abort caching on any cell failure ---------------
    if not all_pass:
        log(
            "a joint cell did not meet thresholds — NO caching "
            "(behavioural non-equivariance under joint "
            "permutation ends the session; predictions + metrics are on "
            "the volume for the offline report)"
        )
        wall_h = (time.time() - t_session0) / 3600
        meta = dict(
            model=MODEL,
            revision=REVISION,
            gpu=GPU,
            frozen_hash=hf_,
            joint_hash=hj,
            cached=False,
            wall_hours=round(wall_h, 3),
            est_cost_usd=round(wall_h * USD_PER_HOUR, 2),
        )
        with open(f"{RES}/phase9/session_meta.json", "w") as f:
            json.dump(meta, f, indent=2)
        stop.set()
        vol.commit()
        log(f"session complete (no caching): {wall_h:.2f} h")
        return gate_res | {"session": meta}

    # ------------- step 4: cache positions x layers, BOTH sets -------------
    log("all joint cells met thresholds — caching positions x layers for " "BOTH sets")
    os.makedirs(f"{RES}/phase9/acts", exist_ok=True)
    layer_idx = np.array(LAYERS)

    def in_session_positions(rec, key):
        """Manifest indices, re-verified in-session against the finder."""
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
        # n_pos from the first episode's manifest entry (11 path P, 12 path G)
        k0 = f"{set_name}/{fn}/0"
        e0 = pman["episodes"][k0]
        n_pos = (
            len(e0["carry_entity"])
            + len(e0["fact_final"])
            + len(e0["query_arg"])
            + len(e0["answer"])
        )
        out_path = f"{RES}/phase9/acts/{set_name}_{stem}.npy"
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
            # hidden_states[l+1] = resid_post of layer l
            hs = torch.stack(
                [out.hidden_states[l + 1] for l in LAYERS], dim=1
            )  # (b, L, seq, d)
            for j, rec in enumerate(chunk):
                key = f"{set_name}/{fn}/{i + j}"
                flat, rlen = in_session_positions(rec, key)
                off = padded_len - rlen  # left padding
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

    # per-file sha256, verified by RE-READING before session end
    for fname in sorted(os.listdir(f"{RES}/phase9/acts")):
        if not fname.endswith(".npy"):
            continue
        fp = f"{RES}/phase9/acts/{fname}"
        h1 = sha_file(fp)
        arr = np.load(fp)  # re-read: loadable, shape sane
        assert arr.ndim == 4
        assert arr.shape[2] == len(LAYERS) and arr.shape[3] == d_model
        assert arr.dtype == np.float16
        h2 = sha_file(fp)  # re-read hash: stable on disk
        assert h1 == h2
        checksums[fname] = dict(sha256=h1, shape=list(arr.shape), dtype=str(arr.dtype))
        del arr
    with open(f"{RES}/phase9/acts/checksums.json", "w") as f:
        json.dump(
            dict(
                frozen_hash=hf_,
                joint_hash=hj,
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
        frozen_hash=hf_,
        joint_hash=hj,
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
    with open(f"{RES}/phase9/session_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    stop.set()
    vol.commit()
    log(f"session complete: {wall_h:.2f} h, est ${wall_h * USD_PER_HOUR:.2f}")
    return gate_res | {"session": meta}


@APP.local_entrypoint()
def main():
    r = run_session.remote()
    print(json.dumps(r, indent=2))

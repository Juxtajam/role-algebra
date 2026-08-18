"""Phase 8A-final — single rented Modal session: gate battery on
Qwen2.5-72B-Instruct bf16 (2x A100-80GB) + conditional activation caching.

FROZEN INPUTS (content hash verified in-session before anything runs):
  dv3-results:phase8a/tasks/*            run-2 episode files (byte-exact)
  dv3-results:phase8a/phase8a_modal.py   scoring code (orbit_metrics, GATE)
  dv3-results:phase8a/phase8a_generate_tasks.py
  CONTENT_HASH below = sha256 over sorted 'relpath:sha256(file)' lines
  (phase8a_final_freeze.py). Any mismatch aborts the session.

This file is the RUNNER for the scaled model only. It does not redefine
scoring: orbit_metrics and the gate thresholds are loaded from the frozen
phase8a_modal.py bytes on the volume via importlib. Episodes, wording,
thresholds unchanged. Answer-position / rendering / decoding conventions
identical to run 2 (chat template, enable_thinking=False kwarg — a no-op on
the Qwen2.5 template — forced "Answer:" prefix, greedy, max_new_tokens=8,
first whitespace-delimited word, exact match).

Outputs -> dv3-results:phase8a_final/ (run-2 phase8a/ prefix untouched):
  gate_results.json      metrics + bootstrap CIs, per cell
  preds_gate_{P,G}_{fit,transfer}.json
  episodes_gate_{P,G}_{fit,transfer}.csv   per-episode records
  acts/ (+ checksums.json)  ONLY if all cells pass
  session_meta.json      model revision, d_model, layers, dtype, timing

Pass/fail branching per the committed config
"""

import csv
import hashlib
import importlib.util
import json
import os
import threading
import time

import modal

APP = modal.App("dv3-phase8a-final")
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
CONTENT_HASH = "84f2e54d85d6e8aa4c1474b608bef5ab69babe54353ef0ef2702d9f6ed38baef"
CELLS = [(p, v) for p in ("P", "G") for v in ("fit", "transfer")]
N_BOOT = 10_000
BOOT_SEED = 20260807
GPU = "A100-80GB:2"
USD_PER_HOUR = 5.00  # 2 x A100-80GB @ $2.50/h (modal.com/pricing, 2026-08-07)

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
    line = f"[{time.strftime('%H:%M:%S')}] MILESTONE phase8a_final: {msg}"
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
    assert h == CONTENT_HASH, f"FROZEN HASH MISMATCH: {h} != {CONTENT_HASH}"
    return h


def load_frozen_scoring():
    """orbit_metrics + GATE from the frozen phase8a_modal.py bytes."""
    spec = importlib.util.spec_from_file_location(
        "frozen_phase8a_modal", f"{RES}/phase8a/phase8a_modal.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.orbit_metrics, mod.GATE


def boot_cis(recs, preds, n_boot=N_BOOT, seed=BOOT_SEED):
    """Bootstrap 95% CIs over base problems (run-2 / recheck methodology)."""
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


def classify(rec, pred):
    """Run-2 error-category decomposition (G cells), verbatim logic."""
    if pred == rec["answer"]:
        return "correct"
    if pred == rec.get("inner_name"):
        return "inner_clause"
    if pred == rec.get("third_name"):
        return "opposite_endpoint"
    return "other"


@APP.function(
    image=image, gpu=GPU, volumes={RES: vol, "/hf": hf_vol}, timeout=10 * 3600
)
def run_session():
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    t_session0 = time.time()
    os.makedirs(f"{RES}/phase8a_final", exist_ok=True)
    stop = threading.Event()

    def committer():
        while not stop.wait(120):
            vol.commit()

    threading.Thread(target=committer, daemon=True).start()

    # ---------------- frozen-input verification ----------------
    h = verify_frozen_hash()
    log(f"frozen content hash VERIFIED: {h}")
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
    log(
        f"model loaded in {load_s:.0f}s: {n_layers} layers, "
        f"d_model={d_model}, dtype=bfloat16, "
        f"map={set(model.hf_device_map.values())}"
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

    def load_cell(kind, path, vocab):
        fp = f"{RES}/phase8a/tasks/{kind}_{path}_{vocab}.jsonl"
        return [json.loads(l) for l in open(fp)]

    # harness sanity (shapes/rendering only — no accuracy pre-check)
    r0 = load_cell("gate", "G", "fit")[0]
    s0 = render(r0)
    print("RENDERED PROMPT:\n", s0, flush=True)
    enc0 = tok([s0], return_tensors="pt").to(model.device)
    with torch.no_grad():
        m0 = model(**enc0, output_hidden_states=True, use_cache=False)
    assert len(m0.hidden_states) == n_layers + 1
    assert m0.hidden_states[1].shape[-1] == d_model
    del m0, enc0
    log("harness sanity: rendering + hidden_states shapes OK")

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

    @torch.no_grad()
    def cache_acts(recs, out_path, bs=8):
        acts = np.zeros((len(recs), n_layers, d_model), dtype=np.float16)
        for i in range(0, len(recs), bs):
            batch = [render(r) for r in recs[i : i + bs]]
            enc = tok(batch, return_tensors="pt", padding=True).to(model.device)
            out = model(**enc, output_hidden_states=True, use_cache=False)
            # hidden_states: (n_layers+1) x (b, seq, d); [l+1] = resid_post
            # of layer l. Left padding -> answer position = last column.
            cols = [
                hs[:, -1, :].to(torch.float16).cpu() for hs in out.hidden_states[1:]
            ]
            acts[i : i + len(batch)] = torch.stack(cols, dim=1).numpy()
            del out, cols
            if (i // bs) % 25 == 0:
                log(f"  acts {i + len(batch)}/{len(recs)} -> {out_path}")
        np.save(out_path, acts)
        vol.commit()

    def write_episode_csv(cell_name, recs, preds):
        fp = f"{RES}/phase8a_final/episodes_gate_{cell_name}.csv"
        with open(fp, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(
                [
                    "base_id",
                    "ep_idx",
                    "g",
                    "qpos_or_qi",
                    "answer",
                    "pred",
                    "correct",
                    "category",
                ]
            )
            for k, (r, p) in enumerate(zip(recs, preds)):
                qp = r.get("qpos", r["base"].get("qi"))
                cat = (
                    classify(r, p)
                    if r["path"] == "G"
                    else ("correct" if p == r["answer"] else "wrong")
                )
                w.writerow(
                    [
                        r["base_id"],
                        k,
                        "".join(map(str, r["g"])),
                        qp,
                        r["answer"],
                        p,
                        int(p == r["answer"]),
                        cat,
                    ]
                )

    # ---------------- gate battery (pre-registered, frozen) ----------------
    results = {
        "model": MODEL,
        "revision": REVISION,
        "gpu": GPU,
        "content_hash": h,
        "gate_thresholds": GATE,
        "d_model": d_model,
        "n_layers": n_layers,
        "dtype": "bfloat16",
        "n_boot": N_BOOT,
        "boot_seed": BOOT_SEED,
        "cells": {},
    }
    all_pass = True
    for path, vocab in CELLS:
        recs = load_cell("gate", path, vocab)
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
        import collections

        cats = (
            dict(collections.Counter(classify(r, p) for r, p in zip(recs, preds)))
            if path == "G"
            else None
        )
        results["cells"][f"{path}/{vocab}"] = dict(
            episode_acc=acc,
            strict_orbit=strict,
            orbit_consistency=cons,
            cis=cis,
            passed=bool(passed),
            n_bases=len(recs) // 6,
            n_eps=len(recs),
            categories=cats,
            seconds=round(time.time() - t0, 1),
        )
        with open(f"{RES}/phase8a_final/preds_gate_{path}_{vocab}.json", "w") as f:
            json.dump(preds, f)
        write_episode_csv(f"{path}_{vocab}", recs, preds)
        vol.commit()
        log(
            f"GATE {path}/{vocab}: acc={acc:.4f} "
            f"CI=[{cis['episode_acc'][0]:.3f},{cis['episode_acc'][1]:.3f}] "
            f"strict={strict:.4f} consist={cons:.4f} "
            f"-> {'PASS' if passed else 'FAIL'}"
        )

    results["gate_passed"] = bool(all_pass)
    with open(f"{RES}/phase8a_final/gate_results.json", "w") as f:
        json.dump(results, f, indent=2)
    vol.commit()
    log(f"GATE OVERALL: {'PASS' if all_pass else 'FAIL'}")

    # ---------------- conditional caching (step 3.3) ----------------
    checksums = {}
    if all_pass:
        log("all cells passed — caching activations (gate + disc sets)")
        os.makedirs(f"{RES}/phase8a_final/acts", exist_ok=True)
        for kind in ("gate", "disc"):
            for path, vocab in CELLS:
                recs = load_cell(kind, path, vocab)
                op = f"{RES}/phase8a_final/acts/{kind}_{path}_{vocab}.npy"
                cache_acts(recs, op)
                log(f"cached {kind}/{path}/{vocab}: {len(recs)} eps")
        # disc predictions (episode-level correctness for 8C filtering;
        # NOT discriminator fitting)
        for path, vocab in CELLS:
            recs = load_cell("disc", path, vocab)
            preds = generate(recs)
            acc, strict, cons = orbit_metrics(recs, preds)
            with open(f"{RES}/phase8a_final/preds_disc_{path}_{vocab}.json", "w") as f:
                json.dump(preds, f)
            log(
                f"disc {path}/{vocab}: acc={acc:.4f} strict={strict:.4f} "
                f"consist={cons:.4f}"
            )
            vol.commit()
        # per-file checksums, verified by RE-READING before session end
        for fn in sorted(os.listdir(f"{RES}/phase8a_final/acts")):
            fp = f"{RES}/phase8a_final/acts/{fn}"
            h1 = sha_file(fp)
            arr = np.load(fp)  # re-read: loadable, shape sane
            assert arr.shape[1] == n_layers and arr.shape[2] == d_model
            h2 = sha_file(fp)  # re-read hash: stable on disk
            assert h1 == h2
            checksums[fn] = dict(sha256=h1, shape=list(arr.shape), dtype=str(arr.dtype))
            del arr
        with open(f"{RES}/phase8a_final/acts/checksums.json", "w") as f:
            json.dump(
                dict(
                    content_hash=h,
                    model=MODEL,
                    revision=REVISION,
                    d_model=d_model,
                    n_layers=n_layers,
                    act_dtype="float16",
                    files=checksums,
                ),
                f,
                indent=2,
            )
        vol.commit()
        log(f"checksums written + verified for {len(checksums)} act files")
    else:
        log(
            "some cell failed — NO caching (per protocol); metric tables, "
            "per-episode CSVs and preds are saved for the offline audit"
        )

    wall_h = (time.time() - t_session0) / 3600
    meta = dict(
        model=MODEL,
        revision=REVISION,
        gpu=GPU,
        content_hash=h,
        d_model=d_model,
        n_layers=n_layers,
        dtype="bfloat16",
        model_load_seconds=round(load_s, 1),
        wall_hours=round(wall_h, 3),
        est_cost_usd=round(wall_h * USD_PER_HOUR, 2),
        usd_per_hour=USD_PER_HOUR,
        gate_passed=bool(all_pass),
    )
    with open(f"{RES}/phase8a_final/session_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    stop.set()
    vol.commit()
    log(f"session complete: {wall_h:.2f} h, est ${wall_h * USD_PER_HOUR:.2f}")
    return results | {"session": meta}


@APP.local_entrypoint()
def main():
    r = run_session.remote()
    print(json.dumps(r, indent=2))

"""Phase 8A — Modal session: behavioural gate on Qwen3-32B bf16 + conditional
activation caching (same session).

- Model: Qwen/Qwen3-32B, bf16, NO quantisation (recorded confound), one
  A100-80GB.
- Gate (pre-registered, frozen before launch): per path x vocab cell —
  episode accuracy >= 0.95, strict-orbit >= 0.90, abstract-role orbit
  consistency >= 0.95; all four cells must pass (both paths, both vocabs).
- If gate passes: cache resid_post at the answer position, EVERY layer, for
  all gate episodes + the discriminator set, to dv3-results:phase8a/acts/.
- If gate fails: no caching; predictions are stored either way for the
  error-category decomposition (offline).

Answer position convention (recorded): prompts are rendered with the Qwen3
chat template, enable_thinking=False, and a forced assistant prefix
"Answer:". The answer position is the LAST prompt token (the position whose
next-token prediction is the answer's single space-prefixed name token).
Decoding is greedy, max_new_tokens=8; episode accuracy compares the first
whitespace-delimited generated word to the target name (exact match).
"""

import json
import os
import threading
import time

import modal

APP = modal.App("dv3-phase8a")
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
    )
    .env({"HF_HOME": "/hf"})
)

MODEL = "Qwen/Qwen3-32B"
RES = "/results"
GATE = dict(episode_acc=0.95, strict_orbit=0.90, orbit_consistency=0.95)
CELLS = [(p, v) for p in ("P", "G") for v in ("fit", "transfer")]


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] MILESTONE phase8a: {msg}"
    print(line, flush=True)
    with open(f"{RES}/progress.log", "a") as f:
        f.write(line + "\n")


def orbit_metrics(recs, preds):
    """Episode acc, strict-orbit, abstract-role orbit consistency."""
    import collections

    bybase = collections.defaultdict(list)
    for r, p in zip(recs, preds):
        bybase[r["base_id"]].append((r, p))
    ep_acc, strict, consist = [], [], []
    for b, items in sorted(bybase.items()):
        oks = [p == r["answer"] for r, p in items]
        ep_acc.extend(oks)
        strict.append(all(oks))
        names = items[0][0]["base"]["names"]
        cons = any(all(p == names[r["g"][s]] for r, p in items) for s in range(3))
        consist.append(cons)
    import numpy as np

    return (float(np.mean(ep_acc)), float(np.mean(strict)), float(np.mean(consist)))


@APP.function(
    image=image, gpu="A100-80GB", volumes={RES: vol, "/hf": hf_vol}, timeout=6 * 3600
)
def smoke():
    """12-episode harness check (render/decode/activation shapes) before the
    full session. Uses the same code paths; results NOT part of the gate."""
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL)
    tok.padding_side = "left"
    model = (
        AutoModelForCausalLM.from_pretrained(
            MODEL,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
            low_cpu_mem_usage=True,
        )
        .cuda()
        .eval()
    )
    hf_vol.commit()
    recs = [json.loads(l) for l in open(f"{RES}/phase8a/tasks/gate_G_fit.jsonl")][:12]
    msgs = [
        {"role": "system", "content": recs[0]["system"]},
        {"role": "user", "content": recs[0]["user"]},
    ]
    s = (
        tok.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        + "Answer:"
    )
    print("RENDERED PROMPT:\n", s)
    enc = tok([s], return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model.generate(
            **enc,
            max_new_tokens=8,
            do_sample=False,
            pad_token_id=tok.pad_token_id or tok.eos_token_id,
        )
        text = tok.decode(out[0, enc.input_ids.shape[1] :], skip_special_tokens=True)
        print("GEN:", repr(text), "TARGET:", recs[0]["answer"])
        m = model(**enc, output_hidden_states=True, use_cache=False)
        print("hidden_states:", len(m.hidden_states), m.hidden_states[1].shape)
    preds = []
    for r in recs:
        mm = [
            {"role": "system", "content": r["system"]},
            {"role": "user", "content": r["user"]},
        ]
        ss = (
            tok.apply_chat_template(
                mm, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
            + "Answer:"
        )
        ee = tok([ss], return_tensors="pt").to("cuda")
        with torch.no_grad():
            oo = model.generate(
                **ee,
                max_new_tokens=8,
                do_sample=False,
                pad_token_id=tok.pad_token_id or tok.eos_token_id,
            )
        tt = (
            tok.decode(oo[0, ee.input_ids.shape[1] :], skip_special_tokens=True)
            .strip()
            .split()
        )
        preds.append(tt[0].strip(".,!") if tt else "")
    acc = float(np.mean([p == r["answer"] for p, r in zip(preds, recs)]))
    print(
        "smoke acc (12 eps, NOT the gate):",
        acc,
        list(zip(preds, [r["answer"] for r in recs]))[:6],
    )
    return dict(acc=acc, preds=preds)


@APP.function(
    image=image, gpu="A100-80GB", volumes={RES: vol, "/hf": hf_vol}, timeout=6 * 3600
)
def run_session():
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    os.makedirs(f"{RES}/phase8a", exist_ok=True)
    stop = threading.Event()

    def committer():
        while not stop.wait(120):
            vol.commit()

    threading.Thread(target=committer, daemon=True).start()

    log("loading tokenizer + model (bf16, no quantisation)")
    tok = AutoTokenizer.from_pretrained(MODEL)
    tok.padding_side = "left"
    model = (
        AutoModelForCausalLM.from_pretrained(
            MODEL,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
            low_cpu_mem_usage=True,
        )
        .cuda()
        .eval()
    )
    hf_vol.commit()
    n_layers = model.config.num_hidden_layers
    log(f"model loaded: {n_layers} layers, d_model={model.config.hidden_size}")

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
                words = text.strip().split()
                preds.append(words[0].strip(".,!") if words else "")
            if (i // bs) % 10 == 0:
                log(f"  gen {i + len(batch)}/{len(recs)}")
        return preds

    @torch.no_grad()
    def cache_acts(recs, out_path, bs=8):
        d = model.config.hidden_size
        acts = np.zeros((len(recs), n_layers, d), dtype=np.float16)
        for i in range(0, len(recs), bs):
            batch = [render(r) for r in recs[i : i + bs]]
            enc = tok(batch, return_tensors="pt", padding=True).to("cuda")
            out = model(**enc, output_hidden_states=True, use_cache=False)
            # hidden_states: (n_layers+1) x (b, seq, d); [l+1] = resid_post l.
            # left padding -> answer position is the last column for every row
            hs = torch.stack(out.hidden_states[1:], dim=1)  # (b, L, seq, d)
            acts[i : i + len(batch)] = hs[:, :, -1, :].to(torch.float16).cpu().numpy()
            if (i // bs) % 25 == 0:
                log(f"  acts {i + len(batch)}/{len(recs)} -> {out_path}")
        np.save(out_path, acts)
        vol.commit()

    # ---------------- gate ----------------
    results = {"model": MODEL, "gate_thresholds": GATE, "cells": {}}
    all_pass = True
    for path, vocab in CELLS:
        recs = load_cell("gate", path, vocab)
        t0 = time.time()
        preds = generate(recs)
        acc, strict, cons = orbit_metrics(recs, preds)
        passed = (
            acc >= GATE["episode_acc"]
            and strict >= GATE["strict_orbit"]
            and cons >= GATE["orbit_consistency"]
        )
        all_pass &= passed
        results["cells"][f"{path}/{vocab}"] = dict(
            episode_acc=acc,
            strict_orbit=strict,
            orbit_consistency=cons,
            passed=bool(passed),
            n_bases=len(recs) // 6,
            n_eps=len(recs),
            seconds=round(time.time() - t0, 1),
        )
        with open(f"{RES}/phase8a/preds_gate_{path}_{vocab}.json", "w") as f:
            json.dump(preds, f)
        vol.commit()
        log(
            f"GATE {path}/{vocab}: acc={acc:.4f} strict={strict:.4f} "
            f"consist={cons:.4f} -> {'PASS' if passed else 'FAIL'}"
        )

    results["gate_passed"] = bool(all_pass)
    with open(f"{RES}/phase8a/gate_results.json", "w") as f:
        json.dump(results, f, indent=2)
    vol.commit()
    log(f"GATE OVERALL: {'PASS' if all_pass else 'FAIL'}")

    # ---------------- conditional caching ----------------
    if all_pass:
        log("gate PASSED — caching activations (gate + discriminator sets)")
        os.makedirs(f"{RES}/phase8a/acts", exist_ok=True)
        for kind in ("gate", "disc"):
            for path, vocab in CELLS:
                recs = load_cell(kind, path, vocab)
                out_path = f"{RES}/phase8a/acts/{kind}_{path}_{vocab}.npy"
                cache_acts(recs, out_path)
                log(f"cached {kind}/{path}/{vocab}: {len(recs)} episodes")
        # disc predictions too (episode-level correctness for 8C filtering)
        for path, vocab in CELLS:
            recs = load_cell("disc", path, vocab)
            preds = generate(recs)
            acc, strict, cons = orbit_metrics(recs, preds)
            with open(f"{RES}/phase8a/preds_disc_{path}_{vocab}.json", "w") as f:
                json.dump(preds, f)
            log(
                f"disc {path}/{vocab}: acc={acc:.4f} strict={strict:.4f} "
                f"consist={cons:.4f}"
            )
            vol.commit()
    else:
        log(
            "gate FAILED — no caching (per protocol); predictions stored "
            "for error-category decomposition"
        )

    stop.set()
    vol.commit()
    log("session complete")
    return results


@APP.local_entrypoint()
def main():
    r = run_session.remote()
    print(json.dumps(r, indent=2))

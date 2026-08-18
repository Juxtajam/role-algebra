"""Phase 8C — Condition 5: patch-and-continue, ONE GPU session, model loads
once ( instruction C). Qwen2.5-72B-Instruct @ 495f3936 bf16, 2x A100-80GB.

PRE-COMPUTED INPUTS (uploaded to dv3-results:phase8c/ before launch):
  cond5_patch_vectors.npy  (7200, 8192) fp16 — full patch set (real fit on
      P/fit TEST + 10 shuffled + 1 identity null fits on CAL), computed on
      CPU from cached activations by phase8c_cond5_prepare.py.
  cond5_manifest.json      item -> (fit_id, generator, eval_row, src_row)
  committed_config.json/.sha256  frozen config (hash re-verified in-session)

PROCEDURE (frozen in committed_config.condition5 before any test read):
  For each item: run episode eval_row of disc_P_transfer with resid_post at
  the frozen patch layer (hidden_states[L+1] convention: the OUTPUT of
  decoder layer index L) REPLACED at the answer position (last prompt
  token, left padding) by the patch vector (cast bf16), continue the
  remaining layers and greedy-decode (8A rendering/decoding verbatim).
  agree = pred_patch == pred_nat, pred_nat = stored 8A-final greedy preds.

Harness checks (mechanics only, no thresholds): (a) unpatched regeneration
of 24 episodes must reproduce stored preds; (b) self-patch (episode's own
cached fp16 activation) on 24 episodes must reproduce stored preds.

Outputs -> dv3-results:phase8c/cond5_results.json (per-item agree bits).
"""

import hashlib
import json
import os
import threading
import time

import modal

APP = modal.App("dv3-phase8c-cond5")
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
GPU = "A100-80GB:2"
USD_PER_HOUR = 5.00
CONFIG_SHA = "d9e0def3cb03b204aa2545344b5d595282fe1657df335cd3e98881fce9a8d2cb"
PATCH_SHA = "c2201c3fa8da18fa293c75eeeb80b7fc8b92f89dcb91dd8ab0dd30d2b57b1a08"


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] MILESTONE phase8c_cond5: {msg}"
    print(line, flush=True)
    with open(f"{RES}/progress.log", "a") as f:
        f.write(line + "\n")


@APP.function(image=image, gpu=GPU, volumes={RES: vol, "/hf": hf_vol}, timeout=4 * 3600)
def run_session():
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    t_session0 = time.time()
    stop = threading.Event()

    def committer():
        while not stop.wait(120):
            vol.commit()

    threading.Thread(target=committer, daemon=True).start()

    # ---------- frozen-input verification ----------
    def sha_file(p):
        return hashlib.sha256(open(p, "rb").read()).hexdigest()

    cfg_txt = open(f"{RES}/phase8c/committed_config.json").read()
    h = hashlib.sha256(cfg_txt.encode()).hexdigest()
    assert h == CONFIG_SHA, f"config hash mismatch {h}"
    cfg = json.loads(cfg_txt)
    LAYER = cfg["patch_layer"]
    hp = sha_file(f"{RES}/phase8c/cond5_patch_vectors.npy")
    assert hp == PATCH_SHA, f"patch vectors hash mismatch {hp}"
    log(
        f"frozen inputs verified: config {h[:12]}, patches {hp[:12]}, "
        f"patch layer {LAYER}"
    )

    manifest = json.load(open(f"{RES}/phase8c/cond5_manifest.json"))
    V = np.load(f"{RES}/phase8c/cond5_patch_vectors.npy")
    items = manifest["items"]
    assert len(items) == len(V) == manifest["n_items"]
    recs = [json.loads(l) for l in open(f"{RES}/phase8a/tasks/disc_P_transfer.jsonl")]
    nat = json.load(open(f"{RES}/phase8a_final/preds_disc_P_transfer.json"))

    # ---------- model ----------
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
    log(f"model loaded in {time.time()-t0:.0f}s")

    def render(rec):
        msgs = [
            {"role": "system", "content": rec["system"]},
            {"role": "user", "content": rec["user"]},
        ]
        s = tok.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        return s + "Answer:"

    # ---------- patch hook on decoder layer index LAYER ----------
    layer_mod = model.model.layers[LAYER]
    patch_state = {"vec": None, "fired": False}

    def hook(mod, inputs, output):
        if patch_state["vec"] is not None and not patch_state["fired"]:
            hs = output[0]
            if hs.shape[1] > 1:  # prefill only
                hs[:, -1, :] = patch_state["vec"].to(hs.device, hs.dtype)
                patch_state["fired"] = True
                return (hs,) + tuple(output[1:])
        return output

    handle = layer_mod.register_forward_hook(hook)

    @torch.no_grad()
    def generate(batch_recs, patch_vecs=None, bs=24):
        preds = []
        for i in range(0, len(batch_recs), bs):
            chunk = batch_recs[i : i + bs]
            enc = tok([render(r) for r in chunk], return_tensors="pt", padding=True).to(
                model.device
            )
            if patch_vecs is not None:
                patch_state["vec"] = torch.tensor(
                    np.asarray(patch_vecs[i : i + bs], dtype=np.float32)
                )
                patch_state["fired"] = False
            out = model.generate(
                **enc,
                max_new_tokens=8,
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
                pad_token_id=tok.pad_token_id or tok.eos_token_id,
            )
            patch_state["vec"] = None
            for j in range(len(chunk)):
                text = tok.decode(
                    out[j, enc.input_ids.shape[1] :], skip_special_tokens=True
                )
                words = text.strip().split()
                preds.append(words[0].strip(".,!") if words else "")
        return preds

    # ---------- harness checks ----------
    probe_rows = sorted({it["eval_row"] for it in items})[:24]
    pr_recs = [recs[r] for r in probe_rows]
    p_un = generate(pr_recs)
    n_match = sum(p == nat[r] for p, r in zip(p_un, probe_rows))
    log(f"harness unpatched: {n_match}/24 reproduce stored preds")
    assert n_match == 24, "unpatched regeneration mismatch — abort"
    acts = np.load(f"{RES}/phase8a_final/acts/disc_P_transfer.npy", mmap_mode="r")
    self_vecs = np.asarray(acts[probe_rows, LAYER, :])
    p_self = generate(pr_recs, patch_vecs=self_vecs)
    n_self = sum(p == nat[r] for p, r in zip(p_self, probe_rows))
    log(f"harness self-patch @L{LAYER}: {n_self}/24 reproduce stored preds")
    assert n_self >= 22, "self-patch harness failure — abort"

    # ---------- full patch set ----------
    out_items = []
    B = 24
    for i0 in range(0, len(items), 600):
        chunk_items = items[i0 : i0 + 600]
        rws = [it["eval_row"] for it in chunk_items]
        preds = generate([recs[r] for r in rws], patch_vecs=V[i0 : i0 + 600], bs=B)
        for it, p in zip(chunk_items, preds):
            out_items.append(
                dict(
                    **it,
                    pred_patch=p,
                    pred_nat=nat[it["eval_row"]],
                    agree=int(p == nat[it["eval_row"]]),
                )
            )
        agg = {}
        for oi in out_items:
            agg.setdefault(oi["fit"], []).append(oi["agree"])
        log(
            f"{len(out_items)}/{len(items)} done; running agree by fit: "
            + " ".join(f"{k}={np.mean(v):.3f}" for k, v in sorted(agg.items()))
        )
        with open(f"{RES}/phase8c/cond5_results_partial.json", "w") as f:
            json.dump(out_items, f)
        vol.commit()

    wall_h = (time.time() - t_session0) / 3600
    summary = {}
    for oi in out_items:
        summary.setdefault(oi["fit"], []).append(oi["agree"])
    summary = {k: dict(agree=float(np.mean(v)), n=len(v)) for k, v in summary.items()}
    res = dict(
        model=MODEL,
        revision=REVISION,
        gpu=GPU,
        config_sha256=CONFIG_SHA,
        patch_sha256=PATCH_SHA,
        patch_layer=LAYER,
        harness=dict(unpatched=n_match, self_patch=n_self),
        summary=summary,
        items=out_items,
        wall_hours=round(wall_h, 3),
        est_cost_usd=round(wall_h * USD_PER_HOUR, 2),
    )
    with open(f"{RES}/phase8c/cond5_results.json", "w") as f:
        json.dump(res, f, indent=2)
    os.remove(f"{RES}/phase8c/cond5_results_partial.json")
    stop.set()
    vol.commit()
    handle.remove()
    log(
        f"session complete: {wall_h:.2f} h, est ${wall_h*USD_PER_HOUR:.2f}; "
        + " ".join(f"{k}:{v['agree']:.3f}" for k, v in sorted(summary.items()))
    )
    return dict(summary=summary, wall_hours=wall_h)


@APP.local_entrypoint()
def main():
    r = run_session.remote()
    print(json.dumps(r, indent=2))

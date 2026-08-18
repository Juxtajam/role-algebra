"""RA4 — where is binding RESOLVED, and is the timing role-following?

RA1/E27: the answer->person DELIVERY route (layers 57-76) is role-following. RA3: that
route is causally real (vs random control) and conjugates (equal effect frozen/joint) but
REDUNDANT — with all 72 delivery heads ablated, the answer name is still the top-ranked of
the k entities (kway=1.000). So BINDING is resolved UPSTREAM of layer 57.

RA4 localises it with a logit-lens sweep: at the answer position, decode every layer's
residual through the final RMSNorm + lm_head and record, per layer, the margin
  margin(L) = logit(answer name) - max_{other k-1 names} logit
The BINDING DEPTH is the first layer where margin>0 (the answer overtakes its competitors).
Reported for FROZEN and JOINT: if the binding-resolution depth is the same under the joint
permutation, the routing computation's timing is role-following (conjugates), pinning the
binding mechanism upstream of the delivery heads.

output_hidden_states (no attention); batched, left-padded. Verifies the frozen hash.
Launch: modal run --detach phase10/routing/ra4_bindingdepth_modal.py
"""

import hashlib
import json
import os
import threading
import time

import modal

APP = modal.App("dv3-ra4-bindingdepth")
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
GPU = "H100:2"
OUT = "phase10/routing"
N_BASES = 120  # x6 = 720 episodes/set
BS = 12


def log(m):
    line = f"[{time.strftime('%H:%M:%S')}] MILESTONE ra4: {m}"
    print(line, flush=True)
    with open(f"{RES}/progress.log", "a") as f:
        f.write(line + "\n")


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


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
    os.makedirs(f"{RES}/{OUT}", exist_ok=True)
    stop = threading.Event()
    threading.Thread(
        target=lambda: [vol.commit() for _ in iter(lambda: stop.wait(120), True)],
        daemon=True,
    ).start()

    TASK_FILES = sorted(
        [
            f"{k}_{p}_{v}.jsonl"
            for k in ("gate", "disc")
            for p in ("P", "G")
            for v in ("fit", "transfer")
        ]
        + ["manifest.json", "name_pools.json"]
    )
    ent = {f"tasks/{f}": sha(f"{RES}/phase8a/tasks/{f}") for f in TASK_FILES}
    for f in ("phase8a_generate_tasks.py", "phase8a_modal.py"):
        ent[f] = sha(f"{RES}/phase8a/{f}")
    h = hashlib.sha256(
        "".join(f"{k}:{v}\n" for k, v in sorted(ent.items())).encode()
    ).hexdigest()
    assert h == FROZEN_HASH, h
    log(f"frozen hash VERIFIED {h}")

    tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    log(f"loading {MODEL} bf16 on {GPU}")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        revision=REVISION,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map="auto",
        low_cpu_mem_usage=True,
    ).eval()
    hf_vol.commit()
    nL = model.config.num_hidden_layers
    norm = model.model.norm
    lm_head = model.lm_head
    log(f"loaded {time.time()-t0:.0f}s: {nL}L")

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

    def name_id(name):
        ids = tok.encode(" " + name)
        return ids[0] if len(ids) == 1 else None

    @torch.no_grad()
    def sweep(recs, tag):
        # per-episode margin[layer] and kway[layer]; binding depth = first layer margin>0
        ep_margin, ep_kway = [], []  # each: list of (nL+1,) arrays
        cnt = 0
        for b0 in range(0, len(recs), BS):
            batch = recs[b0 : b0 + BS]
            texts, nameids, ansidx = [], [], []
            for rec in batch:
                nids = [name_id(x) for x in rec["base"]["names"]]
                aid = name_id(rec["answer"])
                if None in nids or aid is None:
                    continue
                texts.append(render(rec))
                nameids.append(nids)
                ansidx.append(nids.index(aid))
            if not texts:
                continue
            enc = tok(texts, return_tensors="pt", padding=True).to(model.device)
            hs = model(
                **enc, output_hidden_states=True, use_cache=False
            ).hidden_states  # tuple nL+1
            B = len(texts)
            mrows = np.zeros((B, nL + 1))
            krows = np.zeros((B, nL + 1))
            for li, hlayer in enumerate(hs):
                hl = hlayer[:, -1, :].to(lm_head.weight.device)  # (B,hidden)
                logits = lm_head(norm(hl)).float()  # (B,V)
                for bi in range(B):
                    nid = torch.tensor(nameids[bi], device=logits.device)
                    nl = logits[bi, nid]  # (k,)
                    a = ansidx[bi]
                    other = torch.cat([nl[:a], nl[a + 1 :]])
                    mrows[bi, li] = float(nl[a].item() - other.max().item())
                    krows[bi, li] = int(nl.argmax().item() == a)
            ep_margin.extend(mrows)
            ep_kway.extend(krows)
            cnt += B
            if (b0 // BS) % 10 == 0:
                log(f"  {tag} {b0}/{len(recs)}")
        ep_margin = np.array(ep_margin)
        ep_kway = np.array(ep_kway)  # (cnt, nL+1)
        margins = ep_margin.mean(0)
        kway = ep_kway.mean(0)
        # per-episode binding depth: first layer with margin>0 (else nL)
        pos = ep_margin > 0
        depths = np.where(pos.any(1), pos.argmax(1), nL)
        out = dict(
            n=cnt,
            margin_by_layer=margins.tolist(),
            kway_by_layer=kway.tolist(),
            binding_depth_median=float(np.median(depths)),
            binding_depth_mean=float(depths.mean()),
            binding_depth_p90=float(np.percentile(depths, 90)),
            first_layer_kway_ge_50=int(np.argmax(kway >= 0.5)),
            first_layer_kway_ge_90=int(np.argmax(kway >= 0.9)),
        )
        log(
            f"  {tag}: n={cnt} binding-depth median {out['binding_depth_median']:.0f} "
            f"(kway>=90% at layer {out['first_layer_kway_ge_90']}) of {nL}"
        )
        return out

    def load(path):
        return [json.loads(l) for l in open(path)]

    frozen = load(f"{RES}/phase8a/tasks/disc_P_fit.jsonl")[: N_BASES * 6]
    joint = load(f"{RES}/phase9/tasks_joint/joint_P_fit.jsonl")[: N_BASES * 6]
    res = dict(frozen=sweep(frozen, "frozen"), joint=sweep(joint, "joint"))

    meta = dict(
        model=MODEL,
        revision=REVISION,
        gpu=GPU,
        n_layers=nL,
        method="logit-lens (final RMSNorm + lm_head) on answer-position residual per layer",
        results=res,
        session_seconds=round(time.time() - t0s, 1),
    )
    json.dump(meta, open(f"{RES}/{OUT}/ra4_results.json", "w"), indent=2)
    vol.commit()
    log(f"done {time.time()-t0s:.0f}s")
    stop.set()
    return meta


@APP.local_entrypoint()
def main():
    print("Launching RA4 binding-depth logit-lens sweep (2xH100)...")
    m = run_session.remote()
    r = m["results"]
    for s in ("frozen", "joint"):
        print(
            f"{s}: binding-depth median {r[s]['binding_depth_median']:.0f}, "
            f"kway>=90% at layer {r[s]['first_layer_kway_ge_90']} / {m['n_layers']}"
        )
    print("MILESTONE: RA4 sweep COMPLETE")

"""RA3 — CAUSAL knockout of the role-following pointer route.

RA1/E27 showed correlationally that the answer->person route is role-following. The
novelty bar (vs prior 'attention exists' accounts) needs the CAUSAL group-action claim:
is the SAME route causally load-bearing under the joint permutation (A(gx) ~ P_g A(x)
P_g^-1, causally)? RA3 ablates the RA1 pointer heads (72 heads, layers 57-76) by zeroing
their write to the residual (forward_pre_hook on self_attn.o_proj, zero the head's column
block) and measures the collapse of the answer, on FROZEN and JOINT episodes, against a
layer-matched RANDOM-head control.

Metrics at the answer position (last token), per episode:
  - kway: is the answer name the argmax among the k entity-name logits? (the readout)
  - ans_logit: logit of the answer-name token
  - top1: is the answer name the global argmax?
Report baseline vs pointer-knockout vs random-knockout, frozen vs joint. If the pointer
knockout collapses kway on BOTH sets far more than the random control, the equivariant
route is causal and conjugates causally.

No output_attentions (logits only) -> batched, left-padded. Verifies the frozen hash.
Launch: modal run --detach phase10/routing/ra3_knockout_modal.py
"""

import hashlib
import json
import os
import threading
import time

import modal

APP = modal.App("dv3-ra3-knockout")
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
    line = f"[{time.strftime('%H:%M:%S')}] MILESTONE ra3: {m}"
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

    # RA1 pointer heads (computed locally from frozen_Pfit.npz, sel>0.1)
    ph = json.load(open(f"{RES}/{OUT}/ra1_pointer_heads.json"))
    pointer = [tuple(x) for x in ph["heads"]]  # [(layer,head),...] 72
    log(
        f"pointer heads: {len(pointer)} across layers {ph['layers'][0]}-{ph['layers'][-1]}"
    )

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
    nL, nH = model.config.num_hidden_layers, model.config.num_attention_heads
    hd = model.config.hidden_size // nH
    log(f"loaded {time.time()-t0:.0f}s: {nL}L {nH}H head_dim {hd}")

    # layer-matched random control: for each pointer head pick a random other head in the
    # same layer (deterministic: seeded by layer, no global RNG needed at capture time)
    by_layer = {}
    for l, hh in pointer:
        by_layer.setdefault(l, []).append(hh)
    rand_heads = []
    for l, used in by_layer.items():
        cand = [x for x in range(nH) if x not in used]
        # deterministic pick: stride through candidates by a layer-dependent offset
        off = (l * 7 + 3) % len(cand)
        for i in range(len(used)):
            rand_heads.append((l, cand[(off + i * 5) % len(cand)]))
    log(f"random control heads: {len(rand_heads)}")

    # ---- knockout hooks: zero the head's column block in o_proj INPUT (all positions) ----
    hooks = []

    def install(head_set):
        remove()
        layer_heads = {}
        for l, hh in head_set:
            layer_heads.setdefault(l, []).append(hh)
        for l, hs in layer_heads.items():
            oproj = model.model.layers[l].self_attn.o_proj
            cols = torch.cat([torch.arange(x * hd, (x + 1) * hd) for x in hs])

            def pre_hook(mod, inp, cols=cols):
                x = inp[0].clone()
                x[..., cols.to(x.device)] = 0
                return (x,) + tuple(inp[1:])

            hooks.append(oproj.register_forward_pre_hook(pre_hook))

    def remove():
        while hooks:
            hooks.pop().remove()

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
    def evaluate(recs, tag):
        # returns dict of metrics for baseline / pointer / random
        conds = {"baseline": None, "pointer": pointer, "random": rand_heads}
        agg = {c: dict(kway=0, top1=0, ans_logit=0.0, n=0) for c in conds}
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
            for c, hs in conds.items():
                if hs is None:
                    remove()
                else:
                    install(hs)
                logits = model(**enc, use_cache=False).logits[:, -1, :].float()  # (B,V)
                remove()
                for bi in range(len(texts)):
                    nid = torch.tensor(nameids[bi], device=logits.device)
                    nl = logits[bi, nid]  # (k,)
                    kway = int(nl.argmax().item() == ansidx[bi])
                    top1 = int(logits[bi].argmax().item() == nameids[bi][ansidx[bi]])
                    agg[c]["kway"] += kway
                    agg[c]["top1"] += top1
                    agg[c]["ans_logit"] += float(nl[ansidx[bi]].item())
                    agg[c]["n"] += 1
            if (b0 // BS) % 10 == 0:
                log(f"  {tag} {b0}/{len(recs)}")
        out = {}
        for c in conds:
            n = max(agg[c]["n"], 1)
            out[c] = dict(
                kway_acc=agg[c]["kway"] / n,
                top1_acc=agg[c]["top1"] / n,
                mean_ans_logit=agg[c]["ans_logit"] / n,
                n=agg[c]["n"],
            )
        log(
            f"  {tag}: baseline kway {out['baseline']['kway_acc']:.3f} | "
            f"pointer-KO {out['pointer']['kway_acc']:.3f} | random-KO {out['random']['kway_acc']:.3f}"
        )
        return out

    def load(path):
        return [json.loads(l) for l in open(path)]

    frozen = load(f"{RES}/phase8a/tasks/disc_P_fit.jsonl")[: N_BASES * 6]
    joint = load(f"{RES}/phase9/tasks_joint/joint_P_fit.jsonl")[: N_BASES * 6]
    res = dict(frozen=evaluate(frozen, "frozen"), joint=evaluate(joint, "joint"))

    meta = dict(
        model=MODEL,
        revision=REVISION,
        gpu=GPU,
        n_pointer=len(pointer),
        n_random=len(rand_heads),
        pointer_layers=ph["layers"],
        intervention="zero o_proj input head-block (all positions)",
        results=res,
        session_seconds=round(time.time() - t0s, 1),
    )
    json.dump(meta, open(f"{RES}/{OUT}/ra3_results.json", "w"), indent=2)
    vol.commit()
    log(f"done {time.time()-t0s:.0f}s")
    stop.set()
    return meta


@APP.local_entrypoint()
def main():
    print("Launching RA3 causal knockout (2xH100)...")
    m = run_session.remote()
    print(json.dumps(m["results"], indent=1))
    print("MILESTONE: RA3 knockout COMPLETE")

"""RA1 — capture 72B answer-position attention over entity-name tokens, to test
whether role binding is a ROUTING algebra (paper/routing.md).

For each episode, at the answer position (last prompt token), record the
attention mass on each of the k entity-name tokens, per head, per layer. Both
the FROZEN (in-place) and JOINT (name+order) P episodes are captured: the joint
set decouples role from position, so a role-tracking readout (role-indexed) follows
the entity as it moves, while a position-indexed mechanism does not.

Reduced output only: (n_episodes, n_layers, n_heads, k) attention-mass array +
per-episode answer slot and entity-name token ids. eager attention (weights
needed); bs=1 for exact per-episode name positions.

Verifies the frozen hash (in-place set) and the joint hash before capture.
Launch: modal run --detach phase10/routing/ra1_attention_capture_modal.py
"""

import hashlib
import json
import os
import threading
import time

import modal

APP = modal.App("dv3-ra1-attention")
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
OUT = "phase10/routing/attn"
N_BASES = 150  # bases per set (x 6 perms = 900 episodes/set)


def log(m):
    line = f"[{time.strftime('%H:%M:%S')}] MILESTONE ra1: {m}"
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

    # frozen-set hash (in-place P episodes live in phase8a/tasks)
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

    log(f"loading {MODEL} bf16 on {GPU} (eager attention)")
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        revision=REVISION,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
        device_map="auto",
        low_cpu_mem_usage=True,
    ).eval()
    hf_vol.commit()
    nL, nH = model.config.num_hidden_layers, model.config.num_attention_heads
    log(f"loaded {time.time()-t0:.0f}s: {nL}L {nH}H")

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

    def name_tid(name):
        ids = tok.encode(" " + name)
        return ids[0] if len(ids) == 1 else None

    @torch.no_grad()
    def capture(recs, tag):
        # reduced arrays: attention mass on each of k entity names + answer idx
        mass = np.zeros((len(recs), nL, nH, 3), dtype=np.float32)
        answer_idx = np.zeros(
            len(recs), dtype=np.int64
        )  # canonical index of the ANSWER entity
        valid = np.zeros(len(recs), dtype=bool)
        for i, rec in enumerate(recs):
            names = rec["base"]["names"]
            ids = tok(render(rec), return_tensors="pt").to(model.device)
            seq = ids.input_ids[0]
            # entity name token positions (each single-token name appears once)
            pos = []
            ok = True
            for nm in names:
                tid = name_tid(nm)
                where = (seq == tid).nonzero(as_tuple=True)[0]
                if tid is None or len(where) != 1:
                    ok = False
                    break
                pos.append(int(where[0]))
            # the ANSWER entity's canonical index — found by matching the answer
            # name (robust to frozen/joint g-indexing conventions)
            ai = [j for j in range(len(names)) if names[j] == rec["answer"]]
            if not ok or len(ai) != 1:
                continue
            out = model(**ids, output_attentions=True, use_cache=False)
            ap = seq.shape[0] - 1
            for li, att in enumerate(out.attentions):  # (1,nH,seq,seq)
                row = att[0, :, ap, :]  # (nH, seq)
                for si, p in enumerate(pos):
                    mass[i, li, :, si] = row[:, p].float().cpu().numpy()
            answer_idx[i] = ai[0]
            valid[i] = True
            del out
            if i % 100 == 0:
                log(f"  {tag} {i}/{len(recs)}")
        np.savez_compressed(
            f"{RES}/{OUT}/{tag}.npz", mass=mass, answer_idx=answer_idx, valid=valid
        )
        log(f"  {tag}: {valid.sum()}/{len(recs)} valid; saved")
        vol.commit()
        return int(valid.sum())

    # frozen (in-place) P/fit and joint P/fit — same k=3, disjoint sets
    def load(path):
        return [json.loads(l) for l in open(path)]

    frozen = load(f"{RES}/phase8a/tasks/disc_P_fit.jsonl")[: N_BASES * 6]
    joint = load(f"{RES}/phase9/tasks_joint/joint_P_fit.jsonl")[: N_BASES * 6]
    n1 = capture(frozen, "frozen_Pfit")
    n2 = capture(joint, "joint_Pfit")

    meta = dict(
        model=MODEL,
        revision=REVISION,
        gpu=GPU,
        n_layers=nL,
        n_heads=nH,
        k=3,
        position="answer(last prompt token)",
        sets=dict(frozen_Pfit=n1, joint_Pfit=n2),
        note="mass[ep,layer,head,slot] = answer-position attention on the "
        "entity name at canonical slot 'slot'; answer_idx = the canonical "
        "slot index of the ANSWER entity (matched by name).",
        session_seconds=round(time.time() - t0s, 1),
    )
    json.dump(meta, open(f"{RES}/{OUT}/meta.json", "w"), indent=2)
    vol.commit()
    log(f"done {time.time()-t0s:.0f}s; frozen {n1} joint {n2}")
    stop.set()
    return meta


@APP.local_entrypoint()
def main():
    print("Launching RA1 attention capture (2xH100, eager)...")
    m = run_session.remote()
    print(json.dumps(m, indent=1)[:600])
    print("MILESTONE: RA1 capture COMPLETE")

"""Phase 10 Track A — G-path diagnostic session (Llama-3.3-70B).
20 episodes per G cell (fit + transfer), greedy gate-replication + free
generation logged verbatim alongside candidate scores.
"""

import json, os, threading, time
import modal

APP = modal.App("dv3-phase10-trackA2-diagnostic")
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

MODEL = "meta-llama/Llama-3.3-70B-Instruct"
REVISION = "6f6073b423013f6a7d4d9f39144961bfbfbc386b"
RES = "/results"
GPU = "A100-80GB:2"
USD_PER_HOUR = 5.00
OUT_DIR = "results/cross_family/llama/gate/diagnostic"
N_EPISODES_PER_CELL = 20


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] diag: {msg}"
    print(line, flush=True)


@APP.function(
    image=image,
    gpu=GPU,
    volumes={RES: vol, "/hf": hf_vol},
    timeout=3600,
    memory=131072,
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def run_session():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    os.makedirs(f"{RES}/{OUT_DIR}", exist_ok=True)
    stop = threading.Event()

    def committer():
        while not stop.wait(120):
            vol.commit()

    threading.Thread(target=committer, daemon=True).start()

    log(f"loading {MODEL}@{REVISION[:8]}")
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
    log(f"model loaded in {time.time()-t0:.0f}s")

    def render(rec):
        msgs = [
            {"role": "system", "content": rec["system"]},
            {"role": "user", "content": rec["user"]},
        ]
        s = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        return s + "Answer:"

    def load_file(volpath):
        return [json.loads(l) for l in open(volpath)]

    NORMALIZE_CHARS = " \t\n\r*.!,?"

    def normalize(s):
        return str(s).strip(NORMALIZE_CHARS) if s else ""

    for cell_suffix, cell_label in [("G_fit", "G/fit"), ("G_transfer", "G/transfer")]:
        recs = load_file(f"{RES}/phase8a/tasks/gate_{cell_suffix}.jsonl")[
            :N_EPISODES_PER_CELL
        ]

        results = []
        for i, rec in enumerate(recs):
            prompt = render(rec)
            enc = tok(prompt, return_tensors="pt").to(model.device)

            # 1. Greedy decode (gate replication)
            out_greedy = model.generate(
                **enc,
                max_new_tokens=8,
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
                pad_token_id=tok.pad_token_id or tok.eos_token_id,
            )
            greedy_text = tok.decode(
                out_greedy[0, enc.input_ids.shape[1] :],
                skip_special_tokens=True,
            )
            greedy_word = (
                greedy_text.strip().split()[0].strip(NORMALIZE_CHARS)
                if greedy_text.strip()
                else ""
            )

            # 2. Free generation (max_new_tokens=64)
            out_free = model.generate(
                **enc,
                max_new_tokens=64,
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
                pad_token_id=tok.pad_token_id or tok.eos_token_id,
            )
            free_text = tok.decode(
                out_free[0, enc.input_ids.shape[1] :],
                skip_special_tokens=True,
            )

            answer = rec["answer"]
            greedy_ok = normalize(greedy_word) == normalize(answer)
            free_contains = (
                normalize(answer) in free_text if free_text.strip() else False
            )

            results.append(
                {
                    "base_id": rec["base_id"],
                    "ep_idx": i,
                    "g": "".join(map(str, rec["g"])),
                    "answer": answer,
                    "greedy_raw": greedy_text.replace("\n", "\\n"),
                    "greedy_word": greedy_word,
                    "greedy_correct": greedy_ok,
                    "free_text": free_text.replace("\n", "\\n"),
                    "free_contains_answer": free_contains,
                }
            )

        # Write per-cell results
        out_path = f"{RES}/{OUT_DIR}/diagnostic_{cell_suffix}.json"
        with open(out_path, "w") as f:
            json.dump(
                {
                    "model": MODEL,
                    "revision": REVISION,
                    "cell": cell_label,
                    "n_episodes": N_EPISODES_PER_CELL,
                    "normalization": "strip whitespace + *.!,?",
                    "results": results,
                },
                f,
                indent=2,
            )
        vol.commit()

        n_greedy_ok = sum(1 for r in results if r["greedy_correct"])
        n_free_ok = sum(1 for r in results if r["free_contains_answer"])
        log(
            f"{cell_label}: greedy {n_greedy_ok}/{N_EPISODES_PER_CELL} correct, "
            f"free {n_free_ok}/{N_EPISODES_PER_CELL} contain answer"
        )

    stop.set()
    vol.commit()
    log("session complete")
    return {"done": True}


@APP.local_entrypoint()
def main():
    r = run_session.remote()
    print(json.dumps(r, indent=2))

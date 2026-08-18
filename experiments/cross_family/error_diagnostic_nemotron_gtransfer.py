"""Nemotron G/transfer targeted diagnostic — actual error bases.
Tests: base 31 (4 errors), base 47 (1), base 50 (2), base 58 (5), base 74 (4), base 93 (2).
Base 70 (6 errors) already covered by prior diagnostic.
Total: 6 bases x 6 perms = 36 episodes. Single model, 2xA100-80GB.
Free generation only, greedy, 64 tokens. Weights pre-cached on hf-models.
"""

import json, os, time, threading
import modal

APP = modal.App("dv3-diag-nem-gtr")
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
    .env(
        {"HF_HOME": "/hf", "HF_HUB_ENABLE_HF_TRANSFER": "1", "MODEL_FAMILY": "nemotron"}
    )
)

MODEL = "nvidia/Llama-3.1-Nemotron-70B-Instruct-HF"
REVISION = "031d4042f36adc1a52cca51b331d25cbe3cf1022"
GPU = "A100-80GB:2"
RES = "/results"
OUT_DIR = "results/cross_family/error_diagnostic"
NORMALIZE_CHARS = " \t\n\r*.!,?"

# Nemotron G/transfer actual error bases (from gate CSV):
# 31(4), 47(1), 50(2), 58(5), 70(6), 74(4), 93(2)
# Base 70 already tested. Testing the other 6.
G_TRANSFER_BASES = [31, 47, 50, 58, 74, 93]
PERMS = [(0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)]


@APP.function(
    image=image, gpu=GPU, volumes={RES: vol, "/hf": hf_vol}, timeout=3600, memory=131072
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

    print(f"[{time.strftime('%H:%M:%S')}] diag-nem-gtr: loading", flush=True)
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
    print(
        f"[{time.strftime('%H:%M:%S')}] diag-nem-gtr: loaded {time.time()-t0:.0f}s",
        flush=True,
    )

    def normalize(s):
        return str(s).strip(NORMALIZE_CHARS) if s else ""

    def render(rec):
        msgs = [
            {"role": "system", "content": rec["system"]},
            {"role": "user", "content": rec["user"]},
        ]
        s = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        return s + "Answer:"

    def load_episodes(cell_suffix, bases):
        all_recs = [
            json.loads(l) for l in open(f"{RES}/phase8a/tasks/gate_{cell_suffix}.jsonl")
        ]
        return {
            (b, g): all_recs[b * 6 + pi] for b in bases for pi, g in enumerate(PERMS)
        }

    all_results = []
    eps = load_episodes("G_transfer", G_TRANSFER_BASES)
    for (b, g), rec in sorted(eps.items()):
        prompt = render(rec)
        enc = tok(prompt, return_tensors="pt").to(model.device)
        out_free = model.generate(
            **enc,
            max_new_tokens=64,
            do_sample=False,
            pad_token_id=tok.pad_token_id or tok.eos_token_id,
        )
        free_text = tok.decode(
            out_free[0, enc.input_ids.shape[1] :], skip_special_tokens=True
        )
        answer = rec["answer"]
        free_words = [normalize(w) for w in free_text.strip().split()]
        free_extracted = free_words[0] if free_words else ""
        free_has = normalize(answer) in free_words
        all_results.append(
            {
                "family": "Nemotron-70B",
                "cell": "G/transfer",
                "base": b,
                "g": "".join(map(str, g)),
                "answer": answer,
                "free_text": free_text.replace("\n", "\\n"),
                "free_extracted": free_extracted,
                "free_has_answer": free_has,
            }
        )

    out_file = f"{RES}/{OUT_DIR}/episodes_nemotron_gtr.jsonl"
    with open(out_file, "w") as f:
        for r in all_results:
            f.write(json.dumps(r) + "\n")

    summary = {
        "family": "Nemotron-70B",
        "cell": "G/transfer",
        "n_episodes": len(all_results),
        "by_base": {},
    }
    for r in all_results:
        b = str(r["base"])
        if b not in summary["by_base"]:
            summary["by_base"][b] = {"n": 0, "free_ok": 0}
        summary["by_base"][b]["n"] += 1
        if r["free_has_answer"]:
            summary["by_base"][b]["free_ok"] += 1

    with open(f"{RES}/{OUT_DIR}/summary_nemotron_gtr.json", "w") as f:
        json.dump(summary, f, indent=2)

    stop.set()
    vol.commit()
    for b, s in sorted(summary["by_base"].items(), key=lambda x: int(x[0])):
        print(
            f"[{time.strftime('%H:%M:%S')}] diag-nem-gtr: base {b}: {s['free_ok']}/{s['n']} free-ok",
            flush=True,
        )
    print(
        f"[{time.strftime('%H:%M:%S')}] diag-nem-gtr: done {len(all_results)} episodes",
        flush=True,
    )


@APP.local_entrypoint()
def main():
    r = run_session.remote()
    print(json.dumps(r, indent=2))

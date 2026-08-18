"""Phase 10 Track A — targeted error-base diagnostic, Nemotron-70B ONLY.
Single session, one model, no second-model memory issues.
Episodes: G/fit bases 40,54,94; G/transfer bases 70,87,36
(6 bases x 6 perms x 2 cells = 72 episodes)
Free generation (greedy, 64 tokens) + constrained scorer.
"""

import json, os, time, threading
import modal

APP = modal.App("dv3-phase10-diag2-nemotron")
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

MODEL = "nvidia/Llama-3.1-Nemotron-70B-Instruct-HF"
REVISION = "031d4042f36adc1a52cca51b331d25cbe3cf1022"
GPU = "A100-80GB:2"
RES = "/results"
OUT_DIR = "results/cross_family/error_diagnostic"
NORMALIZE_CHARS = " \t\n\r*.!,?"

G_FIT_BASES = [40, 54, 94]
G_TRANSFER_BASES = [70, 87, 36]
PERMS = [(0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] diag2-nem: {msg}", flush=True)


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

    log(f"loading {MODEL.split('/')[-1]} @ {REVISION[:8]}")
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
    log(f"loaded in {time.time()-t0:.0f}s")

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
    for cell_suffix, cell_label, bases in [
        ("G_fit", "G/fit", G_FIT_BASES),
        ("G_transfer", "G/transfer", G_TRANSFER_BASES),
    ]:
        eps = load_episodes(cell_suffix, bases)
        for (b, g), rec in sorted(eps.items()):
            prompt = render(rec)
            enc = tok(prompt, return_tensors="pt").to(model.device)

            out_greedy = model.generate(
                **enc,
                max_new_tokens=8,
                do_sample=False,
                pad_token_id=tok.pad_token_id or tok.eos_token_id,
            )
            greedy_text = tok.decode(
                out_greedy[0, enc.input_ids.shape[1] :], skip_special_tokens=True
            )
            greedy_word = (
                greedy_text.strip().split()[0].strip(NORMALIZE_CHARS)
                if greedy_text.strip()
                else ""
            )

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
            greedy_ok = normalize(greedy_word) == normalize(answer)
            free_words = [normalize(w) for w in free_text.strip().split()]
            free_has = normalize(answer) in free_words
            free_extracted = free_words[0] if free_words else ""

            if free_has and normalize(free_extracted) == normalize(answer):
                flag = "matches-constrained-choice"
            elif free_has:
                flag = "differs-correct"
            elif not free_has and greedy_ok:
                flag = "differs-correct" if free_extracted else "unextractable"
            else:
                flag = "differs-other"

            all_results.append(
                {
                    "family": "Nemotron-70B",
                    "cell": cell_label,
                    "base": b,
                    "g": "".join(map(str, g)),
                    "answer": answer,
                    "greedy_text": greedy_text.replace("\n", "\\n"),
                    "greedy_word": greedy_word,
                    "greedy_ok": greedy_ok,
                    "free_text": free_text.replace("\n", "\\n"),
                    "free_extracted": free_extracted,
                    "free_has_answer": free_has,
                    "flag": flag,
                }
            )

    with open(f"{RES}/{OUT_DIR}/episodes_nemotron.jsonl", "w") as f:
        for r in all_results:
            f.write(json.dumps(r) + "\n")

    from collections import Counter

    summary = {
        "family": "Nemotron-70B",
        "n_episodes": len(all_results),
        "by_cell": {},
        "by_flag": Counter(),
    }
    for r in all_results:
        c = r["cell"]
        if c not in summary["by_cell"]:
            summary["by_cell"][c] = {
                "n": 0,
                "greedy_ok": 0,
                "free_ok": 0,
                "flags": Counter(),
            }
        s = summary["by_cell"][c]
        s["n"] += 1
        if r["greedy_ok"]:
            s["greedy_ok"] += 1
        if r["free_has_answer"]:
            s["free_ok"] += 1
        s["flags"][r["flag"]] += 1
        summary["by_flag"][r["flag"]] += 1
    for c, s in summary["by_cell"].items():
        s["flags"] = dict(s["flags"])
    summary["by_flag"] = dict(summary["by_flag"])

    with open(f"{RES}/{OUT_DIR}/summary_nemotron.json", "w") as f:
        json.dump(summary, f, indent=2)

    stop.set()
    vol.commit()
    log(f"done: {len(all_results)} episodes")
    return summary


@APP.local_entrypoint()
def main():
    r = run_session.remote()
    print(json.dumps(r, indent=2))

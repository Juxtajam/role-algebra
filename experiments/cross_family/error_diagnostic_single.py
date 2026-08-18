"""Phase 10 Track A — targeted error-base diagnostic, SINGLE model.
Tests free-generation vs constrained scorer on actual G-cell error bases.
One model at a time. Weights pre-cached on hf-models volume if available.
Episodes: G/fit bases 40,54,94; G/transfer bases 70,87,36
(3 bases x 6 perms x 2 cells = 36 episodes per model)
Free generation (greedy, 64 tokens) logged alongside stored constrained preds.
Runs on Modal A100-80GB:2. Raised timeout. Background launch."""

import json, os, time, threading, sys
import modal

MODEL_SPECS = {
    "llama33": {
        "name": "Llama-3.3-70B",
        "repo": "meta-llama/Llama-3.3-70B-Instruct",
        "rev": "6f6073b423013f6a7d4d9f39144961bfbfbc386b",
        "secret_needed": True,
    },
    "nemotron": {
        "name": "Nemotron-70B",
        "repo": "nvidia/Llama-3.1-Nemotron-70B-Instruct-HF",
        "rev": "031d4042f36adc1a52cca51b331d25cbe3cf1022",
        "secret_needed": False,
    },
}

# pick the first argument that matches a known model family
_argv_family = next((a for a in sys.argv[1:] if a in MODEL_SPECS), None)
FAMILY = os.environ.get("MODEL_FAMILY", _argv_family or "llama33")
SPEC = MODEL_SPECS[FAMILY]
APP_NAME = f"dv3-diag-{FAMILY}-v2"
MODEL = SPEC["repo"]
REVISION = SPEC["rev"]

APP = modal.App(APP_NAME)
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
    .env({"HF_HOME": "/hf", "HF_HUB_ENABLE_HF_TRANSFER": "1", "MODEL_FAMILY": FAMILY})
)

GPU = "A100-80GB:2"
RES = "/results"
OUT_DIR = "results/cross_family/error_diagnostic"
NORMALIZE_CHARS = " \t\n\r*.!,?"

G_FIT_BASES = [40, 54, 94]
G_TRANSFER_BASES = [70, 87, 36]
PERMS = [(0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)]


# Build secrets list outside decorator to avoid Modal serializer conditional issues
_secrets = (
    [modal.Secret.from_name("huggingface-secret")] if SPEC["secret_needed"] else []
)


@APP.function(
    image=image,
    gpu=GPU,
    volumes={RES: vol, "/hf": hf_vol},
    timeout=3600,
    memory=131072,
    secrets=_secrets,
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

    print(f"[{time.strftime('%H:%M:%S')}] diag: loading {SPEC['name']}", flush=True)
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    # Check if model is cached on hf_vol
    model_path = f"/hf/models--{MODEL.replace('/','--')}"
    cached = os.path.isdir(model_path)
    print(f"[{time.strftime('%H:%M:%S')}] diag: model cached={cached}", flush=True)

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
    print(f"[{time.strftime('%H:%M:%S')}] diag: loaded in {load_s:.0f}s", flush=True)

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
                    "family": SPEC["name"],
                    "cell": cell_label,
                    "base": b,
                    "g": "".join(map(str, g)),
                    "answer": answer,
                    "free_text": free_text.replace("\n", "\\n"),
                    "free_extracted": free_extracted,
                    "free_has_answer": free_has,
                }
            )

    out_file = f"{RES}/{OUT_DIR}/episodes_{FAMILY}.jsonl"
    with open(out_file, "w") as f:
        for r in all_results:
            f.write(json.dumps(r) + "\n")

    from collections import Counter

    summary = {"family": SPEC["name"], "n_episodes": len(all_results), "by_cell": {}}
    for r in all_results:
        c = r["cell"]
        if c not in summary["by_cell"]:
            summary["by_cell"][c] = {"n": 0, "free_ok": 0, "flags": Counter()}
        s = summary["by_cell"][c]
        s["n"] += 1
        if r["free_has_answer"]:
            s["free_ok"] += 1
        s["flags"]["all"] += 1  # simplified
    for c, s in summary["by_cell"].items():
        s["flags"] = dict(s["flags"])

    with open(f"{RES}/{OUT_DIR}/summary_{FAMILY}.json", "w") as f:
        json.dump(summary, f, indent=2)

    stop.set()
    vol.commit()
    print(
        f"[{time.strftime('%H:%M:%S')}] diag: done {len(all_results)} episodes",
        flush=True,
    )


@APP.local_entrypoint()
def main():
    r = run_session.remote()
    print(json.dumps(r, indent=2))

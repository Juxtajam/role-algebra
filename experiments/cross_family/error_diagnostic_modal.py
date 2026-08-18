# Phase 10 — targeted error-base diagnostic (Track A, final item)
# Single session, both Llama-3.3 and Nemotron
# 6 error bases × 6 perms × 2 G cells × 2 families = 144 episodes
# Episodes: G/fit bases 40,54,94; G/transfer bases 70,87,36
# Free generation (greedy, 64 tokens) + constrained scorer

import hashlib, json, os, threading, time
import modal

APP = modal.App("dv3-phase10-diag2")
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

GPU = "A100-80GB:2"
USD_PER_HOUR = 5.00
RES = "/results"
OUT_DIR = "results/cross_family/error_diagnostic"
FROZEN_HASH = "84f2e54d85d6e8aa4c1474b608bef5ab69babe54353ef0ef2702d9f6ed38baef"
NORMALIZE_CHARS = " \t\n\r*.!,?"

MODELS = [
    {
        "name": "Llama-3.3-70B",
        "repo": "meta-llama/Llama-3.3-70B-Instruct",
        "rev": "6f6073b423013f6a7d4d9f39144961bfbfbc386b",
        "secret_needed": True,
        "track": "trackA2",
    },
    {
        "name": "Nemotron-70B",
        "repo": "nvidia/Llama-3.1-Nemotron-70B-Instruct-HF",
        "rev": "031d4042f36adc1a52cca51b331d25cbe3cf1022",
        "secret_needed": False,
        "track": "trackA",
    },
]

# Error bases: same for both families (identical frozen episodes)
G_FIT_BASES = [40, 54, 94]
G_TRANSFER_BASES = [70, 87, 36]

PERMS = [(0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)]


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] diag2: {msg}"
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

    def sha_file(p):
        return hashlib.sha256(open(p, "rb").read()).hexdigest()

    # Verify frozen hash
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
    entries = {}
    for f in TASK_FILES:
        entries[f"tasks/{f}"] = sha_file(f"{RES}/phase8a/tasks/{f}")
    for f in CODE_FILES:
        entries[f] = sha_file(f"{RES}/phase8a/{f}")
    h = hashlib.sha256(
        "".join(f"{k}:{v}\n" for k, v in sorted(entries.items())).encode()
    ).hexdigest()
    assert h == FROZEN_HASH, f"HASH MISMATCH {h}"
    log(f"frozen hash verified: {h[:16]}...")

    def normalize(s):
        return str(s).strip(NORMALIZE_CHARS) if s else ""

    def load_episodes(cell_suffix, bases):
        """Load all episodes for given bases from gate cell."""
        all_recs = [
            json.loads(l) for l in open(f"{RES}/phase8a/tasks/gate_{cell_suffix}.jsonl")
        ]
        out = {}
        for b in bases:
            for pi, g in enumerate(PERMS):
                idx = b * 6 + pi
                out[(b, g)] = all_recs[idx]
        return out

    all_results = []
    prev_model_name = None

    for mi, mcfg in enumerate(MODELS):
        log(
            f"loading {mcfg['name']} ({mcfg['repo'].split('/')[-1]} @ {mcfg['rev'][:8]})"
        )
        t0 = time.time()

        tok = AutoTokenizer.from_pretrained(mcfg["repo"], revision=mcfg["rev"])
        tok.padding_side = "left"
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            mcfg["repo"],
            revision=mcfg["rev"],
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
            device_map="auto",
            low_cpu_mem_usage=True,
        ).eval()
        hf_vol.commit()
        load_s = time.time() - t0
        log(f"  loaded in {load_s:.0f}s")

        def render(rec):
            msgs = [
                {"role": "system", "content": rec["system"]},
                {"role": "user", "content": rec["user"]},
            ]
            s = tok.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True
            )
            return s + "Answer:"

        for cell_suffix, cell_label, bases in [
            ("G_fit", "G/fit", G_FIT_BASES),
            ("G_transfer", "G/transfer", G_TRANSFER_BASES),
        ]:
            eps = load_episodes(cell_suffix, bases)
            for (b, g), rec in sorted(eps.items()):
                prompt = render(rec)
                enc = tok(prompt, return_tensors="pt").to(model.device)

                # 1. Greedy constrained (gate replication)
                out_greedy = model.generate(
                    **enc,
                    max_new_tokens=8,
                    do_sample=False,
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

                # 2. Free generation (greedy, up to 64 tokens)
                out_free = model.generate(
                    **enc,
                    max_new_tokens=64,
                    do_sample=False,
                    pad_token_id=tok.pad_token_id or tok.eos_token_id,
                )
                free_text = tok.decode(
                    out_free[0, enc.input_ids.shape[1] :],
                    skip_special_tokens=True,
                )

                answer = rec["answer"]
                greedy_ok = normalize(greedy_word) == normalize(answer)
                free_words = [normalize(w) for w in free_text.strip().split()]
                free_has_answer = normalize(answer) in free_words
                free_extracted = free_words[0] if free_words else ""

                # adjudication flag
                if free_has_answer and normalize(free_extracted) == normalize(answer):
                    flag = "matches-constrained-choice"
                elif free_has_answer and normalize(free_extracted) != normalize(answer):
                    flag = (
                        "differs-correct"  # first word wrong but answer appears later
                    )
                elif not free_has_answer and greedy_ok:
                    flag = "differs-correct" if free_extracted else "unextractable"
                elif not free_has_answer and not greedy_ok:
                    flag = "differs-other"
                else:
                    flag = "differs-correct" if free_has_answer else "differs-other"

                all_results.append(
                    {
                        "family": mcfg["name"],
                        "cell": cell_label,
                        "base": b,
                        "g": "".join(map(str, g)),
                        "answer": answer,
                        "greedy_text": greedy_text.replace("\n", "\\n"),
                        "greedy_word": greedy_word,
                        "greedy_ok": greedy_ok,
                        "free_text": free_text.replace("\n", "\\n"),
                        "free_extracted": free_extracted,
                        "free_has_answer": free_has_answer,
                        "flag": flag,
                    }
                )

        # Unload model to free GPU memory for next model
        del model
        torch.cuda.empty_cache()
        prev_model_name = mcfg["name"]

    # Write results
    out_path = f"{RES}/{OUT_DIR}/episodes.jsonl"
    with open(out_path, "w") as f:
        for r in all_results:
            f.write(json.dumps(r) + "\n")

    # Summary table
    from collections import Counter

    summary = {"by_family_cell": {}, "by_flag": Counter()}
    for r in all_results:
        key = f"{r['family']}/{r['cell']}"
        if key not in summary["by_family_cell"]:
            summary["by_family_cell"][key] = {
                "n_episodes": 0,
                "greedy_ok": 0,
                "free_ok": 0,
                "flags": Counter(),
            }
        s = summary["by_family_cell"][key]
        s["n_episodes"] += 1
        if r["greedy_ok"]:
            s["greedy_ok"] += 1
        if r["free_has_answer"]:
            s["free_ok"] += 1
        s["flags"][r["flag"]] += 1
        summary["by_flag"][r["flag"]] += 1

    # Convert Counters for JSON
    for k, v in summary["by_family_cell"].items():
        v["flags"] = dict(v["flags"])
    summary["by_flag"] = dict(summary["by_flag"])

    with open(f"{RES}/{OUT_DIR}/summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    vol.commit()
    stop.set()
    vol.commit()
    log("session complete")
    return summary


@APP.local_entrypoint()
def main():
    r = run_session.remote()
    print(json.dumps(r, indent=2))

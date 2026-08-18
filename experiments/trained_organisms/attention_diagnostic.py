"""Phase 4b pre-Item-3 DIAGNOSTIC — Check 3 attention on the Item 2 finals. (verbatim): "Run the Check 3 attention diagnostic on those final
checkpoints, on training-distribution episodes for property→symbol and
composed path P. Report whether any head at any layer attends from the
query-argument position to its matching occurrence in the fact block, with
attention mass against the uniform baseline, layer by layer."

Protocol (identical to Check 3 / checks23.py, adapted to the L4 static-pool
finals):
  * checkpoints: stage2/static_pool/T1_static/seed{0,1,2}/ckpt.pt (the models
    that reached train accuracy 1.000 on composed and aux);
  * episodes: TRAINING distribution = bases drawn from each run's actual
    fixed pool, reconstructed deterministically (first 8192 draws of
    rng=(seed,77), exactly as item2_static_pool.py sampled it), rendered with
    fresh permutations; 96 episodes per query type per seed;
  * train accuracy re-verified on the sampled episodes (should be 1.000);
  * per-layer/head mean attention mass from the query-argument position to
    the matching HAS-property occurrence in the fact block, and (secondary,
    as in Check 3) from the QMARK position to the match and to the answer
    symbol; uniform-over-attendable-context baseline reported alongside.

Decision rule, stated in advance (Check 3 precedent: max 0.153 at ~0.04
baseline was ruled ABSENT): the edge is PRESENT only if some head's mean
mass from the query-arg position to the matching occurrence is >= 0.25 and
>= 5x the uniform baseline; otherwise ABSENT. Full tables reported either
way so the author can re-cut.

Writes checks/item2_attention.{txt,json} on dv3-results.
"""

import modal
import pathlib

app = modal.App("dv3-item2-attention")
vol = modal.Volume.from_name("dv3-results")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.4.1", "numpy", "scipy", "pandas", "matplotlib")
    .add_local_dir(
        str(pathlib.Path(__file__).resolve().parents[2] / "src"),
        remote_path="/root/dv3",
    )
)


@app.function(image=image, timeout=3600, volumes={"/results": vol})
def diagnose():
    import math, os, sys

    os.environ["DV3_RESULTS"] = "/results"
    sys.path.insert(0, "/root/dv3")
    os.chdir("/root/dv3")

    import numpy as np
    import torch
    from shared import progress
    from shared.progress import log
    from trained import data as D
    from trained.model import TinyTransformer, masked_answer_preds, D_MODEL, N_HEADS

    out, res = [], {}

    def p(s=""):
        out.append(s)
        print(s, flush=True)

    POOL_SIZE, N_EP = 8192, 96
    EDGE_ABS, EDGE_REL = 0.25, 5.0

    def attention_capture(model, toks_batch):
        toks = torch.as_tensor(toks_batch)
        B, T = toks.shape
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool), 1)
        emb = model.emb
        x = emb[toks] + model.pos[:T]
        atts = []
        with torch.no_grad():
            for block in model.blocks:
                h = block.ln1(x)
                q, k, v = block.qkv(h).chunk(3, dim=-1)
                hd = D_MODEL // N_HEADS
                q, k, v = (t.view(B, T, N_HEADS, hd).transpose(1, 2) for t in (q, k, v))
                att = (q @ k.transpose(-2, -1)) / math.sqrt(hd)
                att = att.masked_fill(mask, float("-inf")).softmax(dim=-1)
                atts.append(att)
                x = x + block.proj((att @ v).transpose(1, 2).reshape(B, T, D_MODEL))
                x = x + block.mlp(block.ln2(x))
        return atts

    def head_table(atts, from_pos, to_pos, n_ep):
        n_layers, n_heads = len(atts), atts[0].shape[1]
        tab = []
        for li in range(n_layers):
            tab.append(
                [
                    float(
                        np.mean(
                            [
                                float(atts[li][i, h, from_pos[i], to_pos[i]])
                                for i in range(n_ep)
                            ]
                        )
                    )
                    for h in range(n_heads)
                ]
            )
        return tab

    QTYPES = [("property->symbol", D.A_PS), ("composed_P", D.Q_P)]

    p("=" * 88)
    p("PHASE 4b PRE-ITEM-3 DIAGNOSTIC — ATTENTION ON ITEM 2 STATIC-POOL FINALS (L4)")
    p("training-distribution episodes (each run's own fixed pool); chance = 1/3")
    p(
        f"edge-present rule (stated in advance): mean mass >= {EDGE_ABS} AND >= "
        f"{EDGE_REL}x uniform baseline, query-arg -> matching occurrence"
    )
    p("=" * 88)

    edge_any = {}
    for seed in (0, 1, 2):
        model = TinyTransformer(seed=seed, n_layers=4)
        ck = torch.load(
            f"/results/stage2/static_pool/T1_static/seed{seed}/ckpt.pt",
            map_location="cpu",
        )
        model.load_state_dict(ck["model"])
        model.eval()
        p(f"\n--- seed{seed} (final step {ck['step']}) ---")

        # reconstruct the run's actual fixed pool (first draws of rng (seed,77))
        rng = np.random.default_rng((seed, 77))
        pool = [D.sample_base("T1", rng) for _ in range(POOL_SIZE)]
        ep_rng = np.random.default_rng((seed, 4242))
        res[f"seed{seed}"] = {}

        for label, qt in QTYPES:
            bs_all = [b for b in pool if b.qtok == qt]
            take = [bs_all[i] for i in ep_rng.integers(0, len(bs_all), size=N_EP)]
            gs = [D.PERMS[i] for i in ep_rng.integers(0, len(D.PERMS), size=N_EP)]

            toks_l, argpos_l, matchpos_l, qmarkpos_l = [], [], [], []
            apos_l, ans_l, cands_l = [], [], []
            for b, g in zip(take, gs):
                toks, apos, ans, cands = D.render(b, g)
                seq = toks.tolist()
                qpos = seq.index(qt)
                arg_id = seq[qpos + 1]
                # matching occurrence: the same property token inside its HAS clause
                mpos, j = None, 1
                while seq[j] in (D.HAS, D.CARRY, D.GUARD):
                    if seq[j] == D.HAS and seq[j + 1] == arg_id:
                        mpos = j + 1
                    j += 4
                assert mpos is not None
                toks_l.append(toks)
                argpos_l.append(qpos + 1)
                matchpos_l.append(mpos)
                qmarkpos_l.append(apos)
                apos_l.append(apos)
                ans_l.append(ans)
                cands_l.append(cands)

            toks_b = np.stack(toks_l)
            # re-verify train accuracy on these exact episodes
            preds = masked_answer_preds(
                model, toks_b, np.array(apos_l), np.array([list(c) for c in cands_l])
            )
            tacc = float((preds == np.array(ans_l)).mean())

            atts = attention_capture(model, toks_b)
            baseline = 1.0 / float(np.mean(argpos_l))  # uniform over attendable context
            tab = head_table(atts, argpos_l, matchpos_l, N_EP)
            arr = np.array(tab)
            li, h = np.unravel_index(arr.argmax(), arr.shape)
            edge = bool(arr.max() >= EDGE_ABS and arr.max() >= EDGE_REL * baseline)
            edge_any[(seed, label)] = (edge, float(arr.max()), baseline)

            p(
                f"\n  [{label}] train acc on these episodes: {tacc:.3f}  "
                f"(uniform baseline ~{baseline:.3f})"
            )
            p("  attention FROM query-arg position TO matching HAS-property token:")
            for lii, row in enumerate(tab):
                p(
                    f"    layer {lii}: "
                    + "  ".join(f"h{hh}={row[hh]:.3f}" for hh in range(len(row)))
                )
            p(
                f"    MAX mean mass: layer {li} head {h} = {arr.max():.3f} "
                f"({arr.max() / baseline:.1f}x baseline) -> edge "
                + ("PRESENT" if edge else "ABSENT")
            )

            tab_q = head_table(atts, qmarkpos_l, matchpos_l, N_EP)
            arr_q = np.array(tab_q)
            p(
                "  from QMARK -> match, per-layer max: "
                + "  ".join(f"L{lii}={max(r):.3f}" for lii, r in enumerate(tab_q))
                + f"  (overall max {arr_q.max():.3f})"
            )
            tab_s = head_table(atts, qmarkpos_l, [m + 1 for m in matchpos_l], N_EP)
            arr_s = np.array(tab_s)
            lis, hs = np.unravel_index(arr_s.argmax(), arr_s.shape)
            p(
                f"  from QMARK -> answer-symbol token (HAS target): max layer {lis} "
                f"head {hs} = {arr_s.max():.3f}"
            )

            res[f"seed{seed}"][label] = dict(
                train_acc=tacc,
                baseline=baseline,
                arg_to_match=tab,
                arg_to_match_max=float(arr.max()),
                arg_to_match_argmax=[int(li), int(h)],
                edge_present=edge,
                qmark_to_match=tab_q,
                qmark_to_match_max=float(arr_q.max()),
                qmark_to_symbol=tab_s,
                qmark_to_symbol_max=float(arr_s.max()),
            )

    p("\n" + "=" * 88)
    n_present = sum(1 for e, _, _ in edge_any.values() if e)
    p("SUMMARY (query-arg -> matching occurrence, max over layers/heads):")
    for (seed, label), (e, mx, bl) in sorted(edge_any.items()):
        p(
            f"  seed{seed} {label:<18} max {mx:.3f} ({mx / bl:.1f}x baseline) "
            + ("PRESENT" if e else "ABSENT")
        )
    verdict = (
        "EDGE PRESENT in at least one run"
        if n_present
        else "EDGE ABSENT in all runs at train accuracy ~1.000 — the models "
        "solve the training pool by a non-attentional (non-matching) "
        "route; per the author, the single most mechanistically informative "
        "result in Stage 2"
    )
    p(f"VERDICT: {verdict}")
    res["verdict"] = verdict
    res["edge_rule"] = dict(abs=EDGE_ABS, rel=EDGE_REL)

    os.makedirs("/results/checks", exist_ok=True)
    with open("/results/checks/item2_attention.txt", "w") as f:
        f.write("\n".join(out) + "\n")
    progress.save_json("checks/item2_attention.json", res)
    log("MILESTONE: ITEM2 ATTENTION DIAGNOSTIC done — " + verdict)
    vol.commit()
    return "\n".join(out)


@app.local_entrypoint()
def main():
    print(diagnose.remote())

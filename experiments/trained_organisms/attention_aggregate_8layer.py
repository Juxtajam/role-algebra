import pathlib

"""Phase 7 Item 1 — the Item A AGGREGATE attention diagnostic applied to
the 8-LAYER fine-tune checkpoints of EVERY COMPOSING SEED, per
Item 1: "for every seed that composes, run the aggregate
attention diagnostic at the second hop (final checkpoint and one
mid-formation checkpoint) — the question is whether the hop-2 mechanism is
consistent across composing seeds or seed-dependent like A_SN was."

This is itemB_aggregate_attention8.py with exactly two mechanical
adaptations, declared here: (1) diagnose() takes (seed, ckpt_file) pairs so
the same battery runs on ckpt.pt (final) and ckpt_midformation.pt (first
eval past composed 0.5; for seed 2 a declared sanity-gated reconstruction);
(2) the mid-formation pass drops the step==50k assertion and keys results
by "seed{s}@{step}". Measurements and adjudication cuts UNCHANGED.

Original Item B docstring follows.

One eval pass, no training, no checkpoint modified. This is the authorised
Item A diagnostic (itemA_aggregate_attention.py), unchanged in measurements
and adjudication cuts, with exactly two mechanical adaptations:

  * checkpoints: stage2/induction8/finetune/seed{s}/ckpt.pt (8 layers);
    the seed list is restricted to seeds whose composed held-out crossed
    0.5 during fine-tuning (asserted from finetune_result.json at load) —
    per the fine-tune log, seed 2 only (crossed at step 1,500; binding
    branch >= 0.95 fired at step 18,000).
  * n_layers = 8 everywhere (aggregate over 8x4 heads; composed route over
    7 adjacent layer pairs; uniform-aggregate baseline scales accordingly).

Mandated measurements per (seed, qtype in {A_PS, A_SN}) — verbatim Item A:
  1. aggregate query-arg -> match mass summed across ALL heads and layers
     vs the uniform-baseline aggregate;
  2. per-head decomposition: heads >= 2x uniform, layer distribution;
  3. two-step composed route over adjacent layer pairs (head-mean and
     head-pair max vs composed uniform baseline; top intermediate
     classified by token role).
Supplementary blocks carried over from Item A (declared there): the same
measurements from the QMARK/answer position; arg -> match+1 (the
match-successor / canonical induction edge); A_SN secondary HAS-target.

SUPPLEMENTARY, NEW, DECLARED HERE (measurement-only; the reason this seed
is being probed at all is that its COMPOSED query works): the identical
edge battery on composed Q_P episodes, for both hops of the composition —
  hop 1: query-arg (property) -> its HAS-clause occurrence (match1) and
         -> match1+1 (the intermediate symbol = hop-1 answer);
  hop 2: query-arg -> the intermediate symbol's CARRY-key occurrence
         (match2) and -> match2+1 (the final answer name);
plus the QMARK-position equivalents of all four edges. Retrieval accuracy
(candidate-masked) re-measured on the exact episodes probed. No new
adjudication branches: the Item A HIGH/AT-BASELINE cuts are reported for
the mandated A_PS/A_SN block; the composed block is descriptive.

Writes checks/itemB_aggregate_attention8.{txt,json} on dv3-results.
"""
import modal

app = modal.App("dv3-item1-aggregate-attention8")
vol = modal.Volume.from_name("dv3-results")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.4.1", "numpy", "scipy", "pandas", "matplotlib")
    .add_local_dir(
        str(pathlib.Path(__file__).resolve().parents[2] / "src"),
        remote_path="/root/dv3",
    )
)

N_EP = 256
N_LAYERS = 8
CONTRIB_REL = 2.0  # per-head "contributing" threshold (>= 2x uniform)
LOCAL_ABS, LOCAL_REL = 0.25, 5.0  # standing localised-edge rule (Check 3)
AGG_HIGH, AGG_BASE = 2.0, 1.5  # Item A adjudication cuts, unchanged


@app.function(image=image, timeout=3600, volumes={"/results": vol})
def diagnose(jobs: list):
    """jobs: list of [seed, ckpt_file] pairs."""
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

    def attention_capture(model, toks_batch):
        toks = torch.as_tensor(toks_batch)
        B, T = toks.shape
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool), 1)
        x = model.emb[toks] + model.pos[:T]
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

    def classify_pos(seq, j, argp, matchp, qmarkp):
        if j == matchp:
            return "match itself"
        if j == matchp + 1:
            return "answer token (match+1)"
        if j == argp:
            return "query-arg"
        if j == qmarkp:
            return "QMARK"
        t = seq[j]
        if t == D.BOS:
            return "BOS"
        if t == D.SEP:
            return "SEP"
        if t in (D.HAS, D.CARRY, D.GUARD):
            return "relation token"
        if D.PROP0 <= t < D.SYM0:
            return "other property"
        if D.SYM0 <= t < D.NAME0:
            return "other symbol"
        if t >= D.NAME0:
            return "name"
        return "other"

    def edge_stats(atts, from_pos, to_pos, toks_np, argpos, matchpos, qmarkpos):
        """All Item A measurements from from_pos -> to_pos (n_layers=8)."""
        B = len(from_pos)
        n_layers = len(atts)
        rows = torch.arange(B)
        fp = torch.as_tensor(from_pos)
        tp = torch.as_tensor(to_pos)

        tab = np.zeros((n_layers, N_HEADS))
        for li in range(n_layers):
            tab[li] = atts[li][rows, :, fp, tp].mean(0).numpy()
        u = float(np.mean(1.0 / (np.asarray(from_pos) + 1.0)))
        agg = float(tab.sum())
        agg_base = n_layers * N_HEADS * u

        contrib = [
            (int(li), int(h), float(tab[li, h]))
            for li in range(n_layers)
            for h in range(N_HEADS)
            if tab[li, h] >= CONTRIB_REL * u
        ]
        layer_hist = {
            li: sum(1 for c in contrib if c[0] == li) for li in range(n_layers)
        }
        mx = float(tab.max())
        mli, mh = np.unravel_index(tab.argmax(), tab.shape)
        localised = bool(mx >= LOCAL_ABS and mx >= LOCAL_REL * u)

        comp_base = float(
            np.mean(
                [
                    sum(1.0 / ((f + 1.0) * (j + 1.0)) for j in range(t_, f + 1))
                    for f, t_ in zip(from_pos, to_pos)
                ]
            )
        )

        pairs = []
        for l in range(n_layers - 1):
            M_hi = atts[l + 1].mean(1)
            M_lo = atts[l].mean(1)
            row = M_hi[rows, fp, :]
            col = M_lo[rows, :, tp]
            comp_ep = (row * col).sum(1)
            self_term = row[rows, tp] * col[rows, tp]
            comp = float(comp_ep.mean())
            comp_noself = float((comp_ep - self_term).mean())
            R = atts[l + 1][rows, :, fp, :]
            C = atts[l][rows, :, :, tp]
            pairmat = torch.einsum("bht,bgt->bhg", R, C).mean(0).numpy()
            pm = float(pairmat.max())
            ph, pg = np.unravel_index(pairmat.argmax(), pairmat.shape)
            contrib_ep = (row * col).numpy()
            top_j = contrib_ep.argmax(1)
            cats = {}
            for i in range(B):
                c = classify_pos(
                    toks_np[i].tolist(),
                    int(top_j[i]),
                    argpos[i],
                    matchpos[i],
                    qmarkpos[i],
                )
                cats[c] = cats.get(c, 0) + 1
            pairs.append(
                dict(
                    layers=[l, l + 1],
                    composed_headmean=comp,
                    composed_headmean_no_self=comp_noself,
                    composed_x_baseline=comp / comp_base if comp_base else None,
                    headpair_max=pm,
                    headpair_argmax=[int(ph), int(pg)],
                    headpair_x_baseline=pm / comp_base if comp_base else None,
                    top_intermediate_hist={
                        k: v for k, v in sorted(cats.items(), key=lambda kv: -kv[1])
                    },
                )
            )

        return dict(
            table=[[float(v) for v in r] for r in tab],
            per_head_uniform=u,
            aggregate=agg,
            aggregate_baseline=agg_base,
            aggregate_x_baseline=agg / agg_base,
            max_single=mx,
            max_single_head=[int(mli), int(mh)],
            max_single_x_baseline=mx / u,
            localised_edge_present=localised,
            contributing_heads=contrib,
            n_contributing=len(contrib),
            contributing_layer_hist=layer_hist,
            composed_baseline=comp_base,
            composed_pairs=pairs,
        )

    def brief(st):
        return dict(
            aggregate=st["aggregate"],
            aggregate_x_baseline=st["aggregate_x_baseline"],
            max_single=st["max_single"],
            max_single_head=st["max_single_head"],
            max_single_x_baseline=st["max_single_x_baseline"],
            localised_edge_present=st["localised_edge_present"],
            n_contributing=st["n_contributing"],
            contributing_layer_hist=st["contributing_layer_hist"],
        )

    def print_edge(tag, st):
        p(
            f"    {tag}: aggregate {st['aggregate']:.3f} "
            f"({st['aggregate_x_baseline']:.2f}x) | max single "
            f"{st['max_single']:.3f} at L{st['max_single_head'][0]}"
            f"h{st['max_single_head'][1]} ({st['max_single_x_baseline']:.1f}x) "
            f"localised={st['localised_edge_present']}, contributing "
            f"{st['n_contributing']}/{N_LAYERS * N_HEADS} "
            f"{st['contributing_layer_hist']}"
        )

    QTYPES = [("property->symbol", D.A_PS), ("symbol->person", D.A_SN)]

    p("=" * 88)
    p(
        "PHASE 7 ITEM 1 — AGGREGATE DIAGNOSTIC, ALL COMPOSING 8L SEEDS, FINAL + MID-FORMATION"
    )
    p(f"(seed, checkpoint) jobs probed: {jobs}")
    p(
        f"{N_EP} held-out episodes per (seed, query type); {N_LAYERS} layers x "
        f"{N_HEADS} heads; declared cuts unchanged from Item A: contributing "
        f">= {CONTRIB_REL}x uniform; localised >= {LOCAL_ABS} & >= {LOCAL_REL}x; "
        f"aggregate HIGH >= {AGG_HIGH}x, AT-BASELINE < {AGG_BASE}x"
    )
    p("=" * 88)

    for seed, ckpt_file in jobs:
        key = f"induction8/finetune/seed{seed}"
        fr = progress.load_json(f"stage2/{key}/finetune_result.json")
        assert fr["max_composed_held"] >= 0.5, (
            f"seed{seed} composed never crossed 0.5 "
            f"(max {fr['max_composed_held']:.3f}) — not in the mandate"
        )
        ck = torch.load(
            progress.results_dir() / f"stage2/{key}/{ckpt_file}", map_location="cpu"
        )
        if ckpt_file == "ckpt.pt":
            assert ck["step"] == 50_000, f"ckpt at step {ck['step']}"
        model = TinyTransformer(seed=seed, n_layers=N_LAYERS)
        model.load_state_dict(ck["model"])
        model.eval()
        rkey = (
            f"seed{seed}"
            if ckpt_file == "ckpt.pt"
            else f"seed{seed}@midformation_step{ck['step']}"
        )
        p(
            f"\n--- 8L finetune seed{seed} [{ckpt_file}, step {ck['step']}"
            + (
                f", reconstructed composed {ck.get('composed_held'):.3f}"
                if ck.get("reconstructed")
                else ""
            )
            + f"; run composed final {fr['final']['held_episode']:.3f}, max "
            f"{fr['max_composed_held']:.3f}] ---"
        )
        res[rkey] = {}

        # ---- mandated block: A_PS / A_SN, verbatim Item A ------------------
        for label, qt in QTYPES:
            rng = np.random.default_rng((seed, 62_000 + int(qt)))
            toks_l, argpos, matchpos, match2pos, qmarkpos = [], [], [], [], []
            apos_l, ans_l, cands_l = [], [], []
            for _ in range(N_EP):
                b = D.sample_base("T1", rng, force_qtok=qt)
                g = D.PERMS[int(rng.integers(len(D.PERMS)))]
                toks, apos, ans, cands = D.render(b, g)
                seq = toks.tolist()
                qpos = seq.index(qt)
                arg_id = seq[qpos + 1]
                mpos, m2, j = None, None, 1
                while seq[j] in (D.HAS, D.CARRY, D.GUARD):
                    if qt == D.A_PS and seq[j] == D.HAS and seq[j + 1] == arg_id:
                        mpos = j + 1
                    if qt == D.A_SN:
                        if seq[j] == D.CARRY and seq[j + 1] == arg_id:
                            mpos = j + 1
                        if seq[j] == D.HAS and seq[j + 2] == arg_id:
                            m2 = j + 2
                    j += 4
                assert mpos is not None
                toks_l.append(toks)
                argpos.append(qpos + 1)
                matchpos.append(mpos)
                match2pos.append(m2)
                qmarkpos.append(apos)
                apos_l.append(apos)
                ans_l.append(ans)
                cands_l.append(cands)

            toks_b = np.stack(toks_l)
            preds = masked_answer_preds(
                model, toks_b, np.array(apos_l), np.array([list(c) for c in cands_l])
            )
            acc = float((preds == np.array(ans_l)).mean())
            atts = attention_capture(model, toks_b)

            st = edge_stats(atts, argpos, matchpos, toks_b, argpos, matchpos, qmarkpos)
            st["retrieval_acc"] = acc
            st_succ = edge_stats(
                atts,
                argpos,
                [m + 1 for m in matchpos],
                toks_b,
                argpos,
                matchpos,
                qmarkpos,
            )
            st["arg_to_match_succ"] = brief(st_succ)
            st["arg_to_match_succ"]["table"] = st_succ["table"]
            st_q = edge_stats(
                atts, qmarkpos, matchpos, toks_b, argpos, matchpos, qmarkpos
            )
            st["qmark"] = st_q
            st_q_succ = edge_stats(
                atts,
                qmarkpos,
                [m + 1 for m in matchpos],
                toks_b,
                argpos,
                matchpos,
                qmarkpos,
            )
            st["qmark_to_match_succ"] = brief(st_q_succ)
            if qt == D.A_SN and all(m is not None for m in match2pos):
                st2 = edge_stats(
                    atts, argpos, match2pos, toks_b, argpos, matchpos, qmarkpos
                )
                st["secondary_has_target"] = brief(st2)
            res[rkey][label] = st

            p(f"\n  [{label}] retrieval acc (masked, these episodes): {acc:.3f}")
            print_edge("DIRECT arg->match", st)
            for pr in st["composed_pairs"]:
                p(
                    f"    two-step L{pr['layers'][0]}->L{pr['layers'][1]}: "
                    f"head-mean {pr['composed_headmean']:.4f} "
                    f"({pr['composed_x_baseline']:.2f}x; no-self "
                    f"{pr['composed_headmean_no_self']:.4f}) | head-pair max "
                    f"{pr['headpair_max']:.4f} ({pr['headpair_x_baseline']:.2f}x) "
                    f"at hi-h{pr['headpair_argmax'][0]}/lo-h{pr['headpair_argmax'][1]} | "
                    f"top intermediate: {pr['top_intermediate_hist']}"
                )
            print_edge(
                "[suppl] arg->match+1 (induction edge)",
                {k: v for k, v in st["arg_to_match_succ"].items() if k != "table"},
            )
            print_edge("[suppl] QMARK->match", brief(st_q))
            print_edge("[suppl] QMARK->match+1", st["qmark_to_match_succ"])
            if "secondary_has_target" in st:
                print_edge(
                    "[A_SN secondary] arg->HAS-target", st["secondary_has_target"]
                )

        # ---- supplementary block (declared): composed Q_P, both hops --------
        rng = np.random.default_rng((seed, 71_000))
        toks_l, argpos, m1pos, m2pos, qmarkpos = [], [], [], [], []
        apos_l, ans_l, cands_l = [], [], []
        for _ in range(N_EP):
            b = D.sample_base("T1", rng, force_qtok=D.Q_P)
            g = D.PERMS[int(rng.integers(len(D.PERMS)))]
            toks, apos, ans, cands = D.render(b, g)
            seq = toks.tolist()
            qpos = seq.index(D.Q_P)
            arg_id = seq[qpos + 1]
            m1, sym, j = None, None, 1
            while seq[j] in (D.HAS, D.CARRY, D.GUARD):
                if seq[j] == D.HAS and seq[j + 1] == arg_id:
                    m1 = j + 1
                    sym = seq[j + 2]
                j += 4
            assert m1 is not None
            m2, j = None, 1
            while seq[j] in (D.HAS, D.CARRY, D.GUARD):
                if seq[j] == D.CARRY and seq[j + 1] == sym:
                    m2 = j + 1
                j += 4
            assert m2 is not None
            assert seq[m2 + 1] == ans, "hop-2 successor is not the label"
            toks_l.append(toks)
            argpos.append(qpos + 1)
            m1pos.append(m1)
            m2pos.append(m2)
            qmarkpos.append(apos)
            apos_l.append(apos)
            ans_l.append(ans)
            cands_l.append(cands)

        toks_b = np.stack(toks_l)
        preds = masked_answer_preds(
            model, toks_b, np.array(apos_l), np.array([list(c) for c in cands_l])
        )
        acc = float((preds == np.array(ans_l)).mean())
        atts = attention_capture(model, toks_b)

        comp = dict(retrieval_acc=acc)
        edges = [
            (
                "hop1 arg->match (HAS prop occurrence)",
                "hop1_match",
                argpos,
                m1pos,
                m1pos,
            ),
            (
                "hop1 arg->match+1 (intermediate symbol)",
                "hop1_succ",
                argpos,
                [m + 1 for m in m1pos],
                m1pos,
            ),
            (
                "hop2 arg->match (CARRY key occurrence)",
                "hop2_match",
                argpos,
                m2pos,
                m2pos,
            ),
            (
                "hop2 arg->match+1 (ANSWER name)",
                "hop2_succ",
                argpos,
                [m + 1 for m in m2pos],
                m2pos,
            ),
            ("QMARK->hop1 match", "qmark_hop1_match", qmarkpos, m1pos, m1pos),
            (
                "QMARK->hop1 match+1 (intermediate symbol)",
                "qmark_hop1_succ",
                qmarkpos,
                [m + 1 for m in m1pos],
                m1pos,
            ),
            ("QMARK->hop2 match", "qmark_hop2_match", qmarkpos, m2pos, m2pos),
            (
                "QMARK->hop2 match+1 (ANSWER name)",
                "qmark_hop2_succ",
                qmarkpos,
                [m + 1 for m in m2pos],
                m2pos,
            ),
        ]
        p(
            f"\n  [SUPPLEMENTARY composed Q_P] retrieval acc (masked, these "
            f"episodes): {acc:.3f}"
        )
        for tag, kk, fp_, tp_, ref in edges:
            st = edge_stats(atts, fp_, tp_, toks_b, argpos, ref, qmarkpos)
            comp[kk] = st
            print_edge(tag, st)
            for pr in st["composed_pairs"]:
                if (pr["composed_x_baseline"] or 0) >= 2.0 or (
                    pr["headpair_x_baseline"] or 0
                ) >= 4.0:
                    p(
                        f"      two-step L{pr['layers'][0]}->L{pr['layers'][1]}: "
                        f"head-mean {pr['composed_headmean']:.4f} "
                        f"({pr['composed_x_baseline']:.2f}x) | head-pair max "
                        f"{pr['headpair_max']:.4f} ({pr['headpair_x_baseline']:.2f}x) "
                        f"at hi-h{pr['headpair_argmax'][0]}/lo-h"
                        f"{pr['headpair_argmax'][1]} | top intermediate: "
                        f"{pr['top_intermediate_hist']}"
                    )
        res[rkey]["composed_Q_P"] = comp

    # ---- adjudication (Item A rule, mandated block only) --------------------
    p("\n" + "=" * 88)
    p("ADJUDICATION (Item A rule, unchanged; mandated A_PS/A_SN block):")
    flags = {}
    for seed, ckpt_file in jobs:
        rkey = (
            f"seed{seed}"
            if ckpt_file == "ckpt.pt"
            else [k for k in res if k.startswith(f"seed{seed}@")][0]
        )
        for label, _ in QTYPES:
            st = res[rkey][label]
            direct_x = st["aggregate_x_baseline"]
            comp_x = max(pr["composed_x_baseline"] for pr in st["composed_pairs"])
            pair_x = max(pr["headpair_x_baseline"] for pr in st["composed_pairs"])
            q = st["qmark"]
            q_x = q["aggregate_x_baseline"]
            q_comp_x = max(pr["composed_x_baseline"] for pr in q["composed_pairs"])
            q_pair_x = max(pr["headpair_x_baseline"] for pr in q["composed_pairs"])
            works = st["retrieval_acc"] >= 0.9
            high = direct_x >= AGG_HIGH or comp_x >= AGG_HIGH or pair_x >= AGG_HIGH
            at_base = (
                works
                and direct_x < AGG_BASE
                and comp_x < AGG_BASE
                and pair_x < AGG_HIGH
                and q_x < AGG_BASE
                and q_comp_x < AGG_BASE
                and q_pair_x < AGG_HIGH
            )
            verdict = (
                "HIGH (distributed/multi-step route present)"
                if high
                else (
                    "AT BASELINE while retrieval works — STOP FLAG"
                    if at_base
                    else "INTERMEDIATE"
                )
            )
            flags[(rkey, label)] = verdict
            res.setdefault("adjudication", {})[f"{rkey}|{label}"] = dict(
                verdict=verdict,
                direct_x=direct_x,
                comp_x=comp_x,
                pair_x=pair_x,
                qmark_x=q_x,
                qmark_comp_x=q_comp_x,
                qmark_pair_x=q_pair_x,
                retrieval_acc=st["retrieval_acc"],
            )
            p(
                f"  {rkey} {label:<22} acc {st['retrieval_acc']:.3f} | direct "
                f"{direct_x:.2f}x comp {comp_x:.2f}x pair {pair_x:.2f}x | QMARK "
                f"{q_x:.2f}x comp {q_comp_x:.2f}x pair {q_pair_x:.2f}x -> {verdict}"
            )

    stop_flag = any(v.startswith("AT BASELINE") for v in flags.values())
    overall = (
        "STOP — at-baseline while retrieval works"
        if stop_flag
        else "no STOP flag; see per-condition verdicts"
    )
    p(f"\nOVERALL: {overall}")
    res["overall"] = overall
    res["rule"] = dict(
        contrib_rel=CONTRIB_REL,
        local_abs=LOCAL_ABS,
        local_rel=LOCAL_REL,
        agg_high=AGG_HIGH,
        agg_base=AGG_BASE,
        n_ep=N_EP,
        n_layers=N_LAYERS,
    )

    os.makedirs("/results/checks", exist_ok=True)
    with open("/results/checks/item1_aggregate_attention8.txt", "w") as f:
        f.write("\n".join(out) + "\n")
    progress.save_json("checks/item1_aggregate_attention8.json", res)
    log(
        "MILESTONE: ITEM 1 AGGREGATE ATTENTION DIAGNOSTIC (8L, final+midformation) done — "
        + overall
    )
    vol.commit()
    return "\n".join(out)


@app.local_entrypoint()
def main(jobs: str = "2:ckpt.pt,2:ckpt_midformation.pt"):
    parsed = [[int(j.split(":")[0]), j.split(":")[1]] for j in jobs.split(",")]
    print(diagnose.remote(parsed))

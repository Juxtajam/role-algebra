import pathlib

"""Phase 6 Item A — AGGREGATE attention diagnostic.

One eval pass, no training. Blocking the writeup; gates Item B's
interpretation. rationale (verbatim summary): the Phase 5 claim of a non-attentional
mechanism is unsupported — Check 3 only detects a single localised head
above 0.25 mass; a distributed or multi-step route is invisible to it by
construction. Max observed 0.135 at 3.7x baseline is well above noise;
several such heads could jointly implement the match.

Checkpoints (both arms, 3 seeds each):
  * phase5:  stage2/induction/finetune/seed{0,1,2}/ckpt.pt (Phase 5 finals
    — generalising: held-out A_PS 0.89-0.97, A_SN 0.99-1.00)
  * scratch: stage2/T1/seed{0,1,2}/ckpt.pt (from-scratch T1 finals — the
    comparison arm; held-out at chance)

Episodes: held-out task episodes (fresh bases, standing convention), 256
per (seed, query type), query types property->symbol (A_PS) and
symbol->person (A_SN). Retrieval accuracy (candidate-masked) re-measured
on the exact episodes probed.

Match positions (key-before-target clause order):
  * A_PS: arg = property; match = the property occurrence in its HAS
    clause (unique). Answer symbol sits at match+1.
  * A_SN: arg = symbol; PRIMARY match = the symbol's CARRY-key occurrence
    (the fact needed for the answer; answer name at match+1); SECONDARY =
    the same symbol's HAS-target occurrence, reported alongside (a
    match-anywhere circuit may attend there).

Measurements per (arm, seed, qtype):
  1. AGGREGATE mass query-arg -> match, summed across ALL heads and ALL
     layers, vs the uniform-baseline aggregate (n_layers*n_heads * mean
     1/(argpos+1)).
  2. Per-head decomposition: count of heads with mean mass >= 2x the
     per-head uniform baseline at that edge, and their layer distribution
     (full LxH table saved).
  3. Two-step route via composed attention over adjacent layer pairs
     (l, l+1): info flows match -> intermediate j at layer l, then
     j -> query-arg at layer l+1, i.e.
       composed = sum_j  A_{l+1}[arg, j] * A_l[j, match]
     computed (a) on head-MEAN attention per layer (distributed route),
     (b) as the max over head PAIRS (localised two-step route), both vs
     the uniform composed baseline sum_{j=match..arg} 1/((arg+1)(j+1)).
     The j == match term (self-attention at l then direct edge) is also
     reported excluded, and the top intermediate position per episode is
     classified by token role (answer position match+1, SEP, BOS, ...).

Supplementary (reported, clearly labelled): the same aggregate + composed
measurements from the QMARK position (= the answer position, where the
prediction is read out). Justification: the author's at-baseline branch is
defined as "information reaching the answer position without attending to
its source". If the answer position itself attends to the match, that
branch's premise is false even when the arg-position edge is at baseline,
so the flag must not fire on the mandated measurement alone.

ADJUDICATION, recorded in advance (decision thresholds declared here;
the two-recorded outcomes are quoted):
  * HIGH-BUT-DISTRIBUTED (=> "head repurposed and delocalised" replaces
    "consumed scaffold" in all reports): aggregate >= 2x uniform aggregate
    OR any composed pair >= 2x its baseline (head-mean or head-pair max),
    with no single localised head >= 0.25 mass and >= 5x.
  * GENUINELY AT BASELINE while retrieval works (=> FLAG IMMEDIATELY,
    STOP for a decision, do NOT proceed to Item B): on the generalising
    (phase5) arm, retrieval >= 0.9 on the probed episodes AND direct
    aggregate < 1.5x baseline AND every composed pair < 1.5x baseline AND
    max head-pair composed < 2x baseline AND the supplementary
    QMARK-position aggregate + composed also < 1.5x/2x (per the branch's
    own premise, above).
  * Intermediate results between the two: report exact numbers; the STOP
    branch does not fire (it is defined by the at-baseline condition);
    framing follows the numbers with the ambiguity stated.

Writes checks/itemA_aggregate_attention.{txt,json} on dv3-results.
"""
import modal

app = modal.App("dv3-itemA-aggregate-attention")
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
CONTRIB_REL = 2.0  # per-head "contributing" threshold (spec: >2x uniform)
LOCAL_ABS, LOCAL_REL = 0.25, 5.0  # the standing localised-edge rule (Check 3)
AGG_HIGH, AGG_BASE = 2.0, 1.5  # declared adjudication cuts (docstring)


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
        """All mandated measurements from from_pos -> to_pos."""
        B = len(from_pos)
        n_layers = len(atts)
        rows = torch.arange(B)
        fp = torch.as_tensor(from_pos)
        tp = torch.as_tensor(to_pos)

        # 1. direct per-head table + aggregate
        tab = np.zeros((n_layers, N_HEADS))
        for li in range(n_layers):
            tab[li] = atts[li][rows, :, fp, tp].mean(0).numpy()
        u = float(np.mean(1.0 / (np.asarray(from_pos) + 1.0)))  # per-head uniform
        agg = float(tab.sum())
        agg_base = n_layers * N_HEADS * u

        # 2. contributing heads (>= 2x uniform)
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

        # composed baseline per episode: sum_{j=to..from} 1/((from+1)(j+1))
        comp_base = float(
            np.mean(
                [
                    sum(1.0 / ((f + 1.0) * (j + 1.0)) for j in range(t_, f + 1))
                    for f, t_ in zip(from_pos, to_pos)
                ]
            )
        )

        # 3. two-step composed route over adjacent layer pairs
        pairs = []
        for l in range(n_layers - 1):
            M_hi = atts[l + 1].mean(1)  # (B,T,T) head-mean, later layer
            M_lo = atts[l].mean(1)
            row = M_hi[rows, fp, :]  # (B,T)  from_pos -> j
            col = M_lo[rows, :, tp]  # (B,T)  j -> to_pos
            comp_ep = (row * col).sum(1)  # (B,)
            self_term = row[rows, tp] * col[rows, tp]
            comp = float(comp_ep.mean())
            comp_noself = float((comp_ep - self_term).mean())
            # head-pair max
            R = atts[l + 1][rows, :, fp, :]  # (B,H,T)
            C = atts[l][rows, :, :, tp]  # (B,H,T)
            pairmat = torch.einsum("bht,bgt->bhg", R, C).mean(0).numpy()  # (H_hi,H_lo)
            pm = float(pairmat.max())
            ph, pg = np.unravel_index(pairmat.argmax(), pairmat.shape)
            # top intermediate classification (head-mean contributions)
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

    QTYPES = [("property->symbol", D.A_PS), ("symbol->person", D.A_SN)]
    ARMS = [
        ("phase5", "stage2/induction/finetune/seed{s}/ckpt.pt"),
        ("scratch", "stage2/T1/seed{s}/ckpt.pt"),
        # supplementary (declared): same from-scratch protocol at hard 50k
        # (the T1 mains stopped at 25-29k under the never-left-chance
        # rule); matched-horizon control, not part of the adjudication
        ("scratch_L4_50k", "stage2/depth_sweep/T1_L4/seed{s}/ckpt.pt"),
    ]

    p("=" * 88)
    p("PHASE 6 ITEM A — AGGREGATE ATTENTION DIAGNOSTIC")
    p(
        f"{N_EP} held-out episodes per (seed, query type); arms: Phase 5 finals "
        "(generalising) + from-scratch T1 finals (comparison)"
    )
    p(
        f"declared cuts: contributing head >= {CONTRIB_REL}x uniform; localised "
        f"edge >= {LOCAL_ABS} & >= {LOCAL_REL}x; aggregate HIGH >= {AGG_HIGH}x, "
        f"AT-BASELINE < {AGG_BASE}x (see script docstring for the full rule)"
    )
    p("=" * 88)

    for arm, path_tpl in ARMS:
        res[arm] = {}
        for seed in (0, 1, 2):
            ck = torch.load(f"/results/{path_tpl.format(s=seed)}", map_location="cpu")
            sd = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
            step = ck.get("step", "?") if isinstance(ck, dict) else "?"
            model = TinyTransformer(seed=seed, n_layers=4)
            model.load_state_dict(sd)
            model.eval()
            p(f"\n--- arm {arm} seed{seed} (ckpt step {step}) ---")
            res[arm][f"seed{seed}"] = {}

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
                                mpos = j + 1  # primary: CARRY key
                            if seq[j] == D.HAS and seq[j + 2] == arg_id:
                                m2 = j + 2  # secondary: HAS target
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
                    model,
                    toks_b,
                    np.array(apos_l),
                    np.array([list(c) for c in cands_l]),
                )
                acc = float((preds == np.array(ans_l)).mean())
                atts = attention_capture(model, toks_b)

                st = edge_stats(
                    atts, argpos, matchpos, toks_b, argpos, matchpos, qmarkpos
                )
                st["retrieval_acc"] = acc
                # supplementary (decomposition of the dominant two-step term):
                # DIRECT mass from query-arg to the SUCCESSOR of the match
                # (match+1 = the answer token in the key-before-target clause)
                # — the canonical induction edge on task inputs
                st_succ = edge_stats(
                    atts,
                    argpos,
                    [m + 1 for m in matchpos],
                    toks_b,
                    argpos,
                    matchpos,
                    qmarkpos,
                )
                st["arg_to_match_succ"] = dict(
                    table=st_succ["table"],
                    aggregate=st_succ["aggregate"],
                    aggregate_x_baseline=st_succ["aggregate_x_baseline"],
                    max_single=st_succ["max_single"],
                    max_single_head=st_succ["max_single_head"],
                    max_single_x_baseline=st_succ["max_single_x_baseline"],
                    localised_edge_present=st_succ["localised_edge_present"],
                    contributing_heads=st_succ["contributing_heads"],
                    n_contributing=st_succ["n_contributing"],
                    contributing_layer_hist=st_succ["contributing_layer_hist"],
                )
                # supplementary: from the QMARK/answer position
                st_q = edge_stats(
                    atts, qmarkpos, matchpos, toks_b, argpos, matchpos, qmarkpos
                )
                st["qmark"] = st_q
                # A_SN secondary occurrence (HAS target)
                if qt == D.A_SN and all(m is not None for m in match2pos):
                    st2 = edge_stats(
                        atts, argpos, match2pos, toks_b, argpos, matchpos, qmarkpos
                    )
                    st["secondary_has_target"] = dict(
                        aggregate=st2["aggregate"],
                        aggregate_x_baseline=st2["aggregate_x_baseline"],
                        max_single=st2["max_single"],
                        max_single_head=st2["max_single_head"],
                        n_contributing=st2["n_contributing"],
                        contributing_layer_hist=st2["contributing_layer_hist"],
                    )
                res[arm][f"seed{seed}"][label] = st

                p(f"\n  [{label}] retrieval acc (masked, these episodes): {acc:.3f}")
                p(
                    f"    DIRECT arg->match: aggregate {st['aggregate']:.3f} vs "
                    f"uniform-aggregate {st['aggregate_baseline']:.3f} "
                    f"({st['aggregate_x_baseline']:.2f}x) | max single head "
                    f"{st['max_single']:.3f} at L{st['max_single_head'][0]}"
                    f"h{st['max_single_head'][1]} ({st['max_single_x_baseline']:.1f}x) "
                    f"localised={st['localised_edge_present']}"
                )
                p(
                    f"    contributing heads (>= {CONTRIB_REL}x uniform "
                    f"{st['per_head_uniform']:.4f}): {st['n_contributing']}/16, "
                    f"layer hist {st['contributing_layer_hist']}"
                    + (
                        f", heads {st['contributing_heads']}"
                        if st["contributing_heads"]
                        else ""
                    )
                )
                for pr in st["composed_pairs"]:
                    p(
                        f"    two-step L{pr['layers'][0]}->L{pr['layers'][1]}: "
                        f"head-mean {pr['composed_headmean']:.4f} "
                        f"({pr['composed_x_baseline']:.2f}x baseline "
                        f"{st['composed_baseline']:.4f}; no-self "
                        f"{pr['composed_headmean_no_self']:.4f}) | head-pair max "
                        f"{pr['headpair_max']:.4f} "
                        f"({pr['headpair_x_baseline']:.2f}x) at hi-h"
                        f"{pr['headpair_argmax'][0]}/lo-h{pr['headpair_argmax'][1]} | "
                        f"top intermediate: {pr['top_intermediate_hist']}"
                    )
                p(
                    f"    [suppl] arg->match+1 (answer token; canonical induction "
                    f"edge): aggregate {st['arg_to_match_succ']['aggregate']:.3f} "
                    f"({st['arg_to_match_succ']['aggregate_x_baseline']:.2f}x), "
                    f"max single {st['arg_to_match_succ']['max_single']:.3f} at "
                    f"L{st['arg_to_match_succ']['max_single_head'][0]}"
                    f"h{st['arg_to_match_succ']['max_single_head'][1]} "
                    f"({st['arg_to_match_succ']['max_single_x_baseline']:.1f}x) "
                    f"localised={st['arg_to_match_succ']['localised_edge_present']}, "
                    f"contributing {st['arg_to_match_succ']['n_contributing']}/16 "
                    f"{st['arg_to_match_succ']['contributing_layer_hist']}"
                )
                p(
                    f"    [suppl] QMARK->match: aggregate {st_q['aggregate']:.3f} "
                    f"({st_q['aggregate_x_baseline']:.2f}x), max single "
                    f"{st_q['max_single']:.3f} at L{st_q['max_single_head'][0]}"
                    f"h{st_q['max_single_head'][1]}, contributing "
                    f"{st_q['n_contributing']}/16 {st_q['contributing_layer_hist']}; "
                    "two-step head-mean x-baseline: "
                    + ", ".join(
                        f"L{pr['layers'][0]}L{pr['layers'][1]}="
                        f"{pr['composed_x_baseline']:.2f}x"
                        for pr in st_q["composed_pairs"]
                    )
                )
                if "secondary_has_target" in st:
                    s2 = st["secondary_has_target"]
                    p(
                        f"    [A_SN secondary] arg->HAS-target occurrence: aggregate "
                        f"{s2['aggregate']:.3f} ({s2['aggregate_x_baseline']:.2f}x), "
                        f"max {s2['max_single']:.3f}, contributing "
                        f"{s2['n_contributing']}/16"
                    )

    # ---- adjudication on the generalising (phase5) arm ----------------------
    p("\n" + "=" * 88)
    p("ADJUDICATION (phase5 arm; rule declared in the script docstring):")
    flags = {}
    for seed in (0, 1, 2):
        for label, _ in QTYPES:
            st = res["phase5"][f"seed{seed}"][label]
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
            flags[(seed, label)] = (
                verdict,
                direct_x,
                comp_x,
                pair_x,
                q_x,
                q_comp_x,
                q_pair_x,
                st["retrieval_acc"],
            )
            p(
                f"  seed{seed} {label:<18} acc {st['retrieval_acc']:.3f} | direct "
                f"{direct_x:.2f}x comp {comp_x:.2f}x pair {pair_x:.2f}x | QMARK "
                f"{q_x:.2f}x comp {q_comp_x:.2f}x pair {q_pair_x:.2f}x -> {verdict}"
            )

    stop_flag = any(v[0].startswith("AT BASELINE") for v in flags.values())
    any_high = any(v[0].startswith("HIGH") for v in flags.values())
    all_high = all(v[0].startswith("HIGH") for v in flags.values())
    overall = (
        "STOP — at least one generalising condition shows aggregate "
        "genuinely at baseline while retrieval works"
        if stop_flag
        else (
            "HIGH-BUT-DISTRIBUTED in all conditions"
            if all_high
            else (
                "HIGH in some conditions, INTERMEDIATE in others"
                if any_high
                else "INTERMEDIATE everywhere (no STOP; no clean HIGH)"
            )
        )
    )
    p(f"\nOVERALL: {overall}")
    res["adjudication"] = {
        f"seed{s}|{l}": dict(
            verdict=v[0],
            direct_x=v[1],
            comp_x=v[2],
            pair_x=v[3],
            qmark_x=v[4],
            qmark_comp_x=v[5],
            qmark_pair_x=v[6],
            retrieval_acc=v[7],
        )
        for (s, l), v in flags.items()
    }
    res["overall"] = overall
    res["rule"] = dict(
        contrib_rel=CONTRIB_REL,
        local_abs=LOCAL_ABS,
        local_rel=LOCAL_REL,
        agg_high=AGG_HIGH,
        agg_base=AGG_BASE,
        n_ep=N_EP,
    )

    os.makedirs("/results/checks", exist_ok=True)
    with open("/results/checks/itemA_aggregate_attention.txt", "w") as f:
        f.write("\n".join(out) + "\n")
    progress.save_json("checks/itemA_aggregate_attention.json", res)
    log("MILESTONE: ITEM A AGGREGATE ATTENTION DIAGNOSTIC done — " + overall)
    vol.commit()
    return "\n".join(out)


@app.local_entrypoint()
def main():
    print(diagnose.remote())

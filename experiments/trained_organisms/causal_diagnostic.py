import pathlib

"""Phase 7 Item 2 — causal test of the hop-2 edge on
seed 2's 8-LAYER fine-tune FINAL checkpoint. Scope: EXACTLY one head, L4h3
(the 9.6x arg->match2+1 head from the Item B aggregate diagnostic), exactly
the listed interventions. No broader circuit tracing.

Interventions (one eval pass each; no training; no checkpoint modified):
  (a) ZERO-ABLATION: L4h3's output (the head's slice of att@v, before proj)
      is zeroed at ALL positions. Measured on composed held-out episodes
      (the fine-tune eval set, 192 bases x 6 perms) and, as the selectivity
      control, on the single-hop aux sets (A_PS, A_SN, A_NS — the fine-tune
      eval sets, 64 bases each). Item (c) — property->symbol and
      symbol->person under the same ablation — is the A_PS / A_SN rows of
      this same battery.
  (b) RESTORATION: mean-ablation (head output replaced by its across-episode
      mean at each position; episode layout is fixed for T1 so positions
      align) as the corrupted baseline, then the head's NATURAL per-episode
      output (captured from the intact pass on the identical episodes — the
      matched run) patched back. Patching the full natural output must
      reconstruct the intact computation exactly if the mean-ablation
      deficit is attributable to this head's output alone; it is the
      positive control for the patching machinery.

Adjudication, PRE-RECORDED; numeric cuts declared HERE,
before launch: composed "collapses" if zero-ablated composed held-out
<= 0.5 (the standing diagnostic trigger; chance = 1/3); single hops "hold"
if zero-ablated A_PS and A_SN both >= 0.9; restoration "recovers" if
restored composed >= 0.95. Collapse + hold -> language upgrades to
"implements hop 2 (single-seed)". Composed survives -> edge epiphenomenal,
hedged phrasing kept.

TRANSFER-GAP DECOMPOSITION (same Item; intact model only): per-hop transfer
accuracy via aux queries on transfer-vocabulary episodes — A_PS (hop 1;
frozen-embedding induction, should be name-agnostic: no name in query or
answer) and A_SN (hop 2 on a literal symbol key; answer IS a never-trained
name, so it also exercises the frozen-name readout), against the composed
transfer set (the fine-tune eval set, 0.634 final) and the fit-pool
equivalents. New eval sets declared here (measurement-only): transfer aux
orbits at seed 9900 + 31*qtok + seed. If A_SN transfer is high while
composed transfer is not, the gap localises to the COMPUTED-key hop-2 /
readout path and the computed-key operation is recorded as partially
vocabulary-bound.

Per-seed reporting with bootstrap over base problems (standing rule):
10,000 resamples of bases with replacement, percentile 95% CI.

Writes checks/item2_causal.{txt,json} on dv3-results.
"""
import modal

app = modal.App("dv3-item2-causal")
vol = modal.Volume.from_name("dv3-results")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.4.1", "numpy", "scipy", "pandas", "matplotlib")
    .add_local_dir(
        str(pathlib.Path(__file__).resolve().parents[2] / "src"),
        remote_path="/root/dv3",
    )
)

N_LAYERS = 8
LAYER, HEAD = 4, 3  # exactly one head: L4h3
SEED = 2
N_BOOT = 10_000
COLLAPSE_CUT = 0.5  # composed "collapses" if ablated <= this
HOLD_CUT = 0.9  # single hops "hold" if ablated >= this
RECOVER_CUT = 0.95  # restoration "recovers" if restored >= this


@app.function(image=image, timeout=3600, volumes={"/results": vol})
def causal():
    import math, os, sys

    os.environ["DV3_RESULTS"] = "/results"
    sys.path.insert(0, "/root/dv3")
    os.chdir("/root/dv3")

    import numpy as np
    import torch
    from shared import progress
    from shared.progress import log
    from trained import data as D
    from trained.model import TinyTransformer, D_MODEL, N_HEADS

    out, res = [], {}

    def p(s=""):
        out.append(s)
        print(s, flush=True)

    key = f"induction8/finetune/seed{SEED}"
    ck = torch.load(
        progress.results_dir() / f"stage2/{key}/ckpt.pt", map_location="cpu"
    )
    assert ck["step"] == 50_000, f"ckpt at step {ck['step']}"
    model = TinyTransformer(seed=SEED, n_layers=N_LAYERS)
    model.load_state_dict(ck["model"])
    model.eval()

    HD = D_MODEL // N_HEADS

    @torch.no_grad()
    def run(toks_np, mode="intact", values=None, capture=False, batch=256):
        """Forward with the single-head intervention at (LAYER, HEAD).
        mode: intact | zero | replace (values (N,T,hd) required for replace).
        capture=True -> also return the head's natural output (N,T,hd)."""
        N, T = toks_np.shape
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool), 1)
        logits_l, cap_l = [], []
        for i in range(0, N, batch):
            toks = torch.as_tensor(toks_np[i : i + batch])
            B = len(toks)
            x = model.emb[toks] + model.pos[:T]
            for li, block in enumerate(model.blocks):
                h = block.ln1(x)
                q, k, v = block.qkv(h).chunk(3, dim=-1)
                q, k, v = (t.view(B, T, N_HEADS, HD).transpose(1, 2) for t in (q, k, v))
                att = (q @ k.transpose(-2, -1)) / math.sqrt(HD)
                att = att.masked_fill(mask, float("-inf")).softmax(dim=-1)
                ho = (att @ v).transpose(1, 2)  # (B, T, H, hd)
                if li == LAYER:
                    if capture:
                        cap_l.append(ho[:, :, HEAD, :].clone())
                    if mode == "zero":
                        ho = ho.clone()
                        ho[:, :, HEAD, :] = 0.0
                    elif mode == "replace":
                        ho = ho.clone()
                        ho[:, :, HEAD, :] = values[i : i + batch]
                x = x + block.proj(ho.reshape(B, T, D_MODEL))
                x = x + block.mlp(block.ln2(x))
            logits_l.append(model.ln_f(x) @ model.emb.T)
        logits = torch.cat(logits_l)
        return (logits, torch.cat(cap_l)) if capture else logits

    def masked_acc(logits, ev):
        rows = torch.arange(len(logits))
        sel = logits[rows, torch.as_tensor(ev["answer_pos"])]
        cd = torch.as_tensor(ev["candidates"])
        preds = (
            cd.gather(1, sel.gather(1, cd).argmax(1, keepdim=True)).squeeze(1).numpy()
        )
        return preds == ev["answers"]

    def boot_ci(correct, n_bases, rng):
        per_base = correct.reshape(n_bases, -1).mean(1)
        idx = rng.integers(0, n_bases, size=(N_BOOT, n_bases))
        means = per_base[idx].mean(1)
        return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))

    brng = np.random.default_rng(20260807)

    def measure(ev, logits, tag):
        corr = masked_acc(logits, ev)
        lo, hi = boot_ci(corr.astype(float), ev["n_bases"], brng)
        acc = float(corr.mean())
        p(
            f"    {tag:<44} acc {acc:.3f}  [95% CI {lo:.3f}, {hi:.3f}]  "
            f"(n_bases={ev['n_bases']})"
        )
        return dict(
            acc=acc, ci95=[lo, hi], n_bases=ev["n_bases"], n_episodes=int(len(corr))
        )

    p("=" * 88)
    p(
        f"PHASE 7 ITEM 2 — CAUSAL TEST OF L{LAYER}h{HEAD} ON 8L SEED {SEED} "
        f"FINAL (step {ck['step']})"
    )
    p(
        f"declared cuts: collapse <= {COLLAPSE_CUT}, hold >= {HOLD_CUT}, "
        f"recover >= {RECOVER_CUT}; bootstrap {N_BOOT} resamples over bases"
    )
    p("=" * 88)

    # ---- eval sets (fine-tune held-out sets, identical seeds) ---------------
    ev_comp = D.build_eval_orbits("T1", "fit", 192, seed=9000 + SEED)
    ev_aux = {
        qt: D.build_eval_orbits("T1", "fit", 64, seed=9600 + 31 * qt + SEED, qtok=qt)
        for qt in (D.A_PS, D.A_SN, D.A_NS)
    }
    ev_tcomp = D.build_eval_orbits("T1", "transfer", 96, seed=9500 + SEED)
    ev_taux = {
        qt: D.build_eval_orbits(
            "T1", "transfer", 64, seed=9900 + 31 * qt + SEED, qtok=qt
        )
        for qt in (D.A_PS, D.A_SN)
    }

    # ---- (a) zero-ablation battery ------------------------------------------
    p("\n(a) ZERO-ABLATION of L4h3 (all positions) — composed + selectivity:")
    res["zero"] = {}
    p("  composed Q_P (held-out, fit):")
    logits_i, nat_comp = run(ev_comp["tokens"], "intact", capture=True)
    res["zero"]["composed_intact"] = measure(ev_comp, logits_i, "intact")
    res["zero"]["composed_ablated"] = measure(
        ev_comp, run(ev_comp["tokens"], "zero"), "L4h3 zero-ablated"
    )
    for qt, label in (
        (D.A_PS, "A_PS property->symbol (hop 1)"),
        (D.A_SN, "A_SN symbol->person (hop 2, literal key)"),
        (D.A_NS, "A_NS person->symbol (backward read)"),
    ):
        p(f"  {label}:")
        ev = ev_aux[qt]
        res["zero"][f"{D.token_name(qt)}_intact"] = measure(
            ev, run(ev["tokens"], "intact"), "intact"
        )
        res["zero"][f"{D.token_name(qt)}_ablated"] = measure(
            ev, run(ev["tokens"], "zero"), "L4h3 zero-ablated"
        )

    # ---- (b) mean-ablation + restoration ------------------------------------
    p("\n(b) MEAN-ABLATION + RESTORATION on composed (matched natural output):")
    mean_out = nat_comp.mean(0, keepdim=True).expand_as(nat_comp).contiguous()
    res["restore"] = {}
    res["restore"]["composed_mean_ablated"] = measure(
        ev_comp,
        run(ev_comp["tokens"], "replace", values=mean_out),
        "L4h3 mean-ablated (corrupted baseline)",
    )
    res["restore"]["composed_restored"] = measure(
        ev_comp,
        run(ev_comp["tokens"], "replace", values=nat_comp),
        "natural output patched back (restoration)",
    )

    # ---- transfer-gap decomposition (intact model) ---------------------------
    p("\nTRANSFER-GAP DECOMPOSITION (intact model, per-hop aux):")
    res["transfer"] = {}
    p("  fit vocabulary:")
    res["transfer"]["fit_A_PS"] = measure(
        ev_aux[D.A_PS], run(ev_aux[D.A_PS]["tokens"]), "A_PS (hop 1)"
    )
    res["transfer"]["fit_A_SN"] = measure(
        ev_aux[D.A_SN], run(ev_aux[D.A_SN]["tokens"]), "A_SN (hop 2)"
    )
    res["transfer"]["fit_composed"] = res["zero"]["composed_intact"]
    p("    (composed fit = intact row above)")
    p("  transfer vocabulary (never-trained name embeddings):")
    res["transfer"]["transfer_A_PS"] = measure(
        ev_taux[D.A_PS], run(ev_taux[D.A_PS]["tokens"]), "A_PS (hop 1)"
    )
    res["transfer"]["transfer_A_SN"] = measure(
        ev_taux[D.A_SN],
        run(ev_taux[D.A_SN]["tokens"]),
        "A_SN (hop 2 + frozen-name readout)",
    )
    res["transfer"]["transfer_composed"] = measure(
        ev_tcomp, run(ev_tcomp["tokens"]), "composed Q_P"
    )

    # ---- adjudication (pre-recorded) -----------------------------------------
    p("\n" + "=" * 88)
    comp_abl = res["zero"]["composed_ablated"]["acc"]
    ps_abl = res["zero"]["A_PS_ablated"]["acc"]
    sn_abl = res["zero"]["A_SN_ablated"]["acc"]
    restored = res["restore"]["composed_restored"]["acc"]
    collapses = comp_abl <= COLLAPSE_CUT
    holds = ps_abl >= HOLD_CUT and sn_abl >= HOLD_CUT
    recovers = restored >= RECOVER_CUT
    if collapses and holds:
        verdict = (
            "CAUSALLY IMPLICATED — composed collapses "
            f"({comp_abl:.3f} <= {COLLAPSE_CUT}) while single hops hold "
            f"(A_PS {ps_abl:.3f}, A_SN {sn_abl:.3f} >= {HOLD_CUT}); "
            "language upgrades to 'implements hop 2 (single-seed)'"
        )
    elif not collapses:
        verdict = (
            f"EPIPHENOMENAL — composed survives ablation "
            f"({comp_abl:.3f} > {COLLAPSE_CUT}); hop-2 mechanism is "
            "elsewhere or distributed; hedged phrasing kept"
        )
    else:
        verdict = (
            f"NOT SELECTIVE — composed collapses ({comp_abl:.3f}) but a "
            f"single hop also fails (A_PS {ps_abl:.3f}, A_SN {sn_abl:.3f} "
            f"< {HOLD_CUT}); the edge is not hop-2-specific; hedged "
            "phrasing kept"
        )
    p(f"ADJUDICATION: {verdict}")
    p(
        f"restoration control: restored {restored:.3f} "
        f"({'RECOVERS' if recovers else 'FAILS TO RECOVER — machinery suspect'})"
    )
    res["adjudication"] = dict(
        verdict=verdict,
        collapses=collapses,
        holds=holds,
        recovers=recovers,
        cuts=dict(collapse=COLLAPSE_CUT, hold=HOLD_CUT, recover=RECOVER_CUT),
    )

    tps, tsn = (
        res["transfer"]["transfer_A_PS"]["acc"],
        res["transfer"]["transfer_A_SN"]["acc"],
    )
    tcomp = res["transfer"]["transfer_composed"]["acc"]
    p(
        f"\nTRANSFER GAP: A_PS {tps:.3f}, A_SN {tsn:.3f}, composed {tcomp:.3f} "
        f"(fit composed {res['transfer']['fit_composed']['acc']:.3f})"
    )
    if tps >= 0.9 and tsn >= 0.9 and tcomp < 0.9:
        p(
            "  -> both single hops transfer while composition does not: the gap "
            "localises to the COMPUTED-key hop-2/readout path; the computed-key "
            "operation is recorded as partially vocabulary-bound."
        )
    res["scope"] = dict(layer=LAYER, head=HEAD, seed=SEED, step=int(ck["step"]))

    os.makedirs("/results/checks", exist_ok=True)
    with open("/results/checks/item2_causal.txt", "w") as f:
        f.write("\n".join(out) + "\n")
    progress.save_json("checks/item2_causal.json", res)
    log("MILESTONE: ITEM 2 CAUSAL TEST done — " + verdict)
    vol.commit()
    return "\n".join(out)


@app.local_entrypoint()
def main():
    print(causal.remote())

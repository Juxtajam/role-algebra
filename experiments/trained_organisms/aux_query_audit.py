"""7.1 — Auxiliary query scoring audit (blocking).

Runs on CPU against the existing T0/T1 checkpoints on the dv3-results volume.
Nothing here modifies frozen configs, checkpoints, or Stage 1 artifacts —
output goes to /results/aux_audit/.

Steps:
  1. Dump 20 raw property->symbol (A_PS) and 20 person->symbol (A_NS)
     episodes per model: full token sequence, the candidate set actually
     scored, stored label, model's masked prediction, and top-5 UNMASKED
     predictions over the whole vocabulary.
  2. State candidate-set size and chance per aux query type.
  3. Independent symbolic-solver audit on aux episodes specifically
     (2000 per aux type per organism).
  4. Realised aux-type proportions from an actual training pool sampled
     exactly as train.py samples it.
"""

import modal
import pathlib

app = modal.App("dv3-aux-audit")
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
def audit():
    import os, sys

    os.environ["DV3_RESULTS"] = "/results"
    os.environ["MPLBACKEND"] = "Agg"
    sys.path.insert(0, "/root/dv3")
    os.chdir("/root/dv3")

    import numpy as np
    import torch
    from trained import data as D
    from trained.model import TinyTransformer, masked_answer_preds

    out = []

    def p(s=""):
        out.append(s)
        print(s, flush=True)

    def load_model(org, seed):
        m = TinyTransformer(seed=seed)
        ck = torch.load(f"/results/stage2/{org}/seed{seed}/ckpt.pt", map_location="cpu")
        m.load_state_dict(ck["model"])
        m.eval()
        return m

    AUX_TYPES = [
        ("property->symbol", D.A_PS),
        ("symbol->person", D.A_SN),
        ("person->symbol", D.A_NS),
        ("symbol->guarded", D.A_SG),
    ]

    # ---- 7.1.2 first: candidate-set structure and chance, from code+data --
    p("=" * 78)
    p("7.1.2 — CANDIDATE SETS AND CHANCE PER AUX TYPE")
    p("=" * 78)
    rng = np.random.default_rng(1)
    for label, qt in AUX_TYPES:
        b = D.sample_base("T3", rng, force_qtok=qt)
        _, ans, cands = b.query_and_answer(D.PERMS[0])
        kind = (
            "symbols"
            if all(D.SYM0 <= c < D.NAME0 for c in cands)
            else "names" if all(c >= D.NAME0 for c in cands) else "MIXED(BUG)"
        )
        p(
            f"  {label:<18} candidates={len(cands)} ({kind}), answer in candidates: "
            f"{ans in cands}, chance under masking = 1/{len(cands)} = {1/len(cands):.3f}"
        )
    p("  masked_answer_preds restricts logits to exactly these candidate tokens;")
    p("  unmasked emission space is the full vocab (V=%d)." % D.VOCAB)

    # ---- 7.1.1: episode dumps with model predictions ----------------------
    def dump(org, seed, qt, label, n=20):
        p("")
        p("-" * 78)
        p(f"7.1.1 — {org}/seed{seed} — {label} — {n} episodes")
        p("-" * 78)
        model = load_model(org, seed)
        rng = np.random.default_rng(20260806)
        n_correct_masked = 0
        top1_type_counts = {}
        for i in range(n):
            b = D.sample_base(org, rng, force_qtok=qt)
            g = D.PERMS[int(rng.integers(len(D.PERMS)))]
            toks, apos, ans, cands = D.render(b, g)
            with torch.no_grad():
                logits = model(torch.as_tensor(toks)[None])[0, apos]
            top5 = torch.topk(logits, 5)
            top5_toks = [
                (D.token_name(int(t)), round(float(v), 2))
                for t, v in zip(top5.indices, top5.values)
            ]
            cand_logits = logits[torch.as_tensor(cands)]
            pred_masked = cands[int(cand_logits.argmax())]
            ok = pred_masked == ans
            n_correct_masked += ok
            t1 = int(top5.indices[0])
            t1_type = (
                "symbol"
                if D.SYM0 <= t1 < D.NAME0
                else (
                    "name"
                    if t1 >= D.NAME0
                    else "prop" if D.PROP0 <= t1 < D.SYM0 else "special"
                )
            )
            top1_type_counts[t1_type] = top1_type_counts.get(t1_type, 0) + 1
            seq = " ".join(D.token_name(t) for t in toks.tolist() if t != D.PAD)
            p(f"[{i:02d}] {'OK ' if ok else 'ERR'} seq: {seq}")
            p(
                f"     candidates={[D.token_name(c) for c in cands]}  "
                f"label={D.token_name(ans)}  masked_pred={D.token_name(pred_masked)}"
            )
            p(f"     top5 unmasked: {top5_toks}")
        p(
            f"  SUMMARY {org}/seed{seed} {label}: masked acc {n_correct_masked}/{n}, "
            f"unmasked top-1 token types: {top1_type_counts}"
        )

    for org, seed in (("T1", 0), ("T0", 0)):
        for label, qt in (("property->symbol", D.A_PS), ("person->symbol", D.A_NS)):
            dump(org, seed, qt, label)

    # ---- larger-n masked + unmasked accuracy per aux type, all 6 models ---
    p("")
    p("=" * 78)
    p("LARGE-N PER-TYPE ACCURACY (masked vs unmasked-argmax), 96 bases x 6 perms")
    p("=" * 78)
    for org, seeds in (("T0", (0, 1, 2)), ("T1", (0, 1, 2))):
        for seed in seeds:
            model = load_model(org, seed)
            row = []
            for label, qt in AUX_TYPES:
                if qt in (D.A_SG,) and "G" not in D.ORG_SPECS[org]["paths"]:
                    continue
                ev = D.build_eval_orbits(org, "fit", 96, seed=1234, qtok=qt)
                preds = masked_answer_preds(
                    model, ev["tokens"], ev["answer_pos"], ev["candidates"]
                )
                acc_m = float((preds == ev["answers"]).mean())
                # unmasked argmax over full vocab
                accs_u = []
                with torch.no_grad():
                    for j in range(0, len(ev["tokens"]), 512):
                        t = torch.as_tensor(ev["tokens"][j : j + 512])
                        ap = torch.as_tensor(ev["answer_pos"][j : j + 512])
                        lg = model(t)
                        rows_i = torch.arange(len(t))
                        accs_u.append(
                            (
                                lg[rows_i, ap].argmax(1).numpy()
                                == ev["answers"][j : j + 512]
                            )
                        )
                acc_u = float(np.concatenate(accs_u).mean())
                row.append(f"{label}: masked={acc_m:.3f} unmasked={acc_u:.3f}")
            p(f"  {org}/seed{seed}: " + " | ".join(row))

    # ---- 7.1.3: symbolic solver audit on aux episodes ---------------------
    p("")
    p("=" * 78)
    p("7.1.3 — INDEPENDENT SOLVER AUDIT, AUX EPISODES (2000 per type per org)")
    p("=" * 78)

    def solver_audit_aux(org, qt, n=2000, seed=321):
        rng = np.random.default_rng(seed)
        failures = 0
        first_failure = None
        for i in range(n):
            b = D.sample_base(org, rng, force_qtok=qt)
            g = D.PERMS[int(rng.integers(len(D.PERMS)))]
            toks, apos, ans, cands = D.render(b, g)
            seq = toks.tolist()
            has, carry, guard = {}, {}, {}
            j = 1
            while seq[j] not in (D.Q_P, D.Q_G) + D.AUX:
                rel, a, t = seq[j], seq[j + 1], seq[j + 2]
                {D.HAS: has, D.CARRY: carry, D.GUARD: guard}[rel][a] = t
                j += 4
            qtok, arg, qm = seq[j], seq[j + 1], seq[j + 2]
            if qtok == D.A_PS:
                solved = has[arg]
            elif qtok == D.A_SN:
                solved = carry[arg]
            elif qtok == D.A_NS:
                solved = {v: k for k, v in carry.items()}[arg]
            else:
                solved = guard[arg]
            checks = (
                solved == ans
                and qm == D.QMARK
                and apos == j + 2
                and seq[apos + 1] == ans
                and ans in list(cands)
                and len(set(cands)) == D.K
            )
            if not checks:
                failures += 1
                if first_failure is None:
                    first_failure = (
                        i,
                        D.token_name(qtok),
                        f"solved={D.token_name(solved)} label={D.token_name(ans)}",
                    )
        return failures, first_failure

    for org in ("T0", "T1"):
        for label, qt in AUX_TYPES:
            if qt == D.A_SG and "G" not in D.ORG_SPECS[org]["paths"]:
                continue
            f, ff = solver_audit_aux(org, qt)
            p(
                f"  {org} {label:<18}: {f}/2000 solver/label/mask failures"
                + (f"  FIRST: {ff}" if ff else "")
            )

    # ---- 7.1.4: realised aux proportions from an actual training pool -----
    p("")
    p("=" * 78)
    p(
        "7.1.4 — REALISED QUERY-TYPE PROPORTIONS (training pool, sampled as train.py does)"
    )
    p("=" * 78)
    for org in ("T0", "T1"):
        for seed in (0,):
            rng = np.random.default_rng((seed, 77))
            pool = [D.sample_base(org, rng) for _ in range(8192)]
            counts = {}
            for b in pool:
                counts[D.token_name(b.qtok)] = counts.get(D.token_name(b.qtok), 0) + 1
            total = sum(counts.values())
            p(
                f"  {org}/seed{seed} pool (n={total}): "
                + "  ".join(
                    f"{k}={v} ({v/total:.1%})" for k, v in sorted(counts.items())
                )
            )

    text = "\n".join(out)
    os.makedirs("/results/aux_audit", exist_ok=True)
    with open("/results/aux_audit/report.txt", "w") as f:
        f.write(text + "\n")
    vol.commit()
    return text


@app.local_entrypoint()
def main():
    print(audit.remote()[-2000:])

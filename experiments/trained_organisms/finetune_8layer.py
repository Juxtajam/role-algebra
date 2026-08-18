"""Phase 6 Item B steps 2+3 — task fine-tuning of the
verified 8-LAYER induction-pretrained checkpoints, with mandatory per-eval
induction re-verification and train-accuracy logging, exactly as in
Phase 5 (phase5_finetune.py); the ONLY variable vs Phase 5 is depth (8
layers, carried in from pretraining).

Protocol verbatim from Phase 5: standard T1, batch 256, AdamW lr 1e-3
wd 0.01 (name rows in the explicit wd=0.0 group, frozen, bit-exact audited
every eval), cosine over hard 50k with 500 warmup, full-sequence LM loss,
candidate-masked eval every 500, pool 8192, POOL REFRESH every 2500 (ON),
train-accuracy logging on (separate eval rng stream). Optimizer starts
FRESH. Control arm: the existing Phase 5 4-layer runs — NOT rerun.

Gate: a seed advances only if stage2/induction8/pretrain/seed{s}/
pretrain_verify.json has pass=true (behavioural AND mechanistic AND
off-period probe); step-0 induction re-verification re-asserted on the
loaded checkpoint (standard battery; the off-period result is carried in
the pretrain verification).

Binding branch conditions logged as MILESTONE lines when observed
(decisions are the author's, pre-recorded):
  * composed held-out >= 0.95 on any seed -> stop and report for a
    decision (dose-response reopens then, not automatically);
  * composed at chance at 50k with hops at ceiling -> stop the line and
    write up (hops-without-chaining with both hops verified is complete);
  * hops fail to form at 8 layers despite verified pretraining -> report
    and stop (contradicts Phase 5; needs a decision).
Additionally logged: composed crossing 0.5 (triggers the Item A aggregate
diagnostic on that seed per the reporting requirements), and
aux_min >= 0.95 / induction-loss milestones as in Phase 5.

Runs keyed stage2/induction8/finetune/seed{s} on dv3-results.
"""

import modal
import pathlib

app = modal.App("dv3-phase6-finetune8")
vol = modal.Volume.from_name("dv3-results")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.4.1", "numpy", "scipy", "pandas", "matplotlib")
    .add_local_dir(
        str(pathlib.Path(__file__).resolve().parents[2] / "src"),
        remote_path="/root/dv3",
    )
)

MAX_STEPS = 50_000
N_LAYERS = 8


@app.function(image=image, gpu="A10G", timeout=60 * 60 * 12, volumes={"/results": vol})
def finetune(seed: int):
    import math, os, sys, threading, time, traceback

    os.environ["DV3_RESULTS"] = "/results"
    sys.path.insert(0, "/root/dv3")
    os.chdir("/root/dv3")

    stop_evt = threading.Event()

    def committer():
        while not stop_evt.is_set():
            time.sleep(60)
            try:
                vol.commit()
            except Exception:
                pass

    threading.Thread(target=committer, daemon=True).start()

    import numpy as np
    import torch
    from shared import progress
    from shared.progress import log
    from trained import data as D
    from trained import induction as I
    from trained.model import (
        TinyTransformer,
        lm_loss,
        make_optimizer,
        masked_answer_preds,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    key = f"induction8/finetune/seed{seed}"
    ckpt_path = progress.results_dir() / f"stage2/{key}/ckpt.pt"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    BATCH, EVAL_EVERY, POOL_SIZE, POOL_REFRESH, WARMUP = 256, 500, 8192, 2500, 500
    AUX_T1 = (D.A_PS, D.A_SN, D.A_NS)

    def lr_at(step):
        if step < WARMUP:
            return 1e-3 * (step + 1) / WARMUP
        t = (step - WARMUP) / max(1, MAX_STEPS - WARMUP)
        return 1e-3 * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * min(t, 1.0))))

    # ---- verification gate: only verified seeds advance --------------------
    ver_rel = f"stage2/induction8/pretrain/seed{seed}/pretrain_verify.json"
    assert progress.exists(ver_rel), f"no pretrain verification for seed{seed}"
    pre_ver = progress.load_json(ver_rel)
    assert pre_ver["pass"], (
        f"seed{seed} FAILED Item B verification battery "
        f"(behavioural={pre_ver['behavioural_pass']}, "
        f"mechanistic={pre_ver['mechanistic_pass']}, "
        f"off_period={pre_ver.get('off_period_pass')}) — does not advance"
    )
    pre_head = tuple(pre_ver["max_head"])
    log(
        f"phase6 finetune8 {key}: gate OK — 8L pretrained head "
        f"L{pre_head[0]} h{pre_head[1]} (copy {pre_ver['copy_acc']:.3f}, "
        f"mass {pre_ver['max_mass']:.3f} = {pre_ver['x_baseline']:.1f}x, "
        f"off-period pass={pre_ver.get('off_period_pass')})"
    )

    torch.manual_seed(seed)
    model = TinyTransformer(seed=seed, n_layers=N_LAYERS).to(device)
    pre_ck = torch.load(
        progress.results_dir() / f"stage2/induction8/pretrain/seed{seed}/ckpt.pt",
        map_location=device,
    )
    assert pre_ck["step"] == 50_000, f"pretrain ckpt at step {pre_ck['step']}"
    model.load_state_dict(pre_ck["model"])
    log(
        f"phase6 finetune8 {key}: loaded 8L pretrained checkpoint; optimizer "
        f"FRESH (initialisation is the variable; depth carried from pretraining)"
    )

    opt = make_optimizer(model, lr=1e-3, weight_decay=0.01)
    assert [pg["weight_decay"] for pg in opt.param_groups] == [0.01, 0.0]

    frozen_init = model.emb_names.detach().clone()
    ev_ind = I.build_eval_set(seed)

    ver0 = I.verify_induction(model, ev_ind, device)
    log(
        f"phase6 finetune8 {key}: step-0 induction re-verify — copy "
        f"{ver0['copy_acc']:.3f}, max mass {ver0['max_mass']:.3f} at "
        f"L{ver0['max_head'][0]} h{ver0['max_head'][1]} (pass={ver0['pass']})"
    )
    assert ver0["pass"], "loaded checkpoint fails induction verification"

    log(
        f"phase6 finetune8 {key}: START — device={device}, standard T1, "
        f"batch {BATCH}, AdamW lr 1e-3 wd 0.01 (name rows wd=0.0, frozen), "
        f"cosine/{MAX_STEPS} warmup {WARMUP}, hard {MAX_STEPS}, pool "
        f"{POOL_SIZE} refresh {POOL_REFRESH} (ON), train-acc logging ON, "
        f"per-eval induction re-verification ON"
    )

    rng = np.random.default_rng((seed, 77))
    eval_rng = np.random.default_rng((seed, 88))
    pool = [D.sample_base("T1", rng) for _ in range(POOL_SIZE)]

    ev_aux = {
        qt: D.build_eval_orbits("T1", "fit", 64, seed=9600 + 31 * qt + seed, qtok=qt)
        for qt in AUX_T1
    }
    ev_comp = D.build_eval_orbits("T1", "fit", 192, seed=9000 + seed)
    ev_transfer = D.build_eval_orbits("T1", "transfer", 96, seed=9500 + seed)

    def acc(ev):
        preds = masked_answer_preds(
            model, ev["tokens"], ev["answer_pos"], ev["candidates"]
        )
        return float((preds == ev["answers"]).mean())

    def train_acc(qtoks, n=96):
        bs = [b for b in pool if b.qtok in qtoks][:n]
        if not bs:
            return None
        gs = [D.PERMS[i] for i in eval_rng.integers(0, len(D.PERMS), size=len(bs))]
        toks, apos, ans, cands = D.render_batch(bs, gs)
        preds = masked_answer_preds(model, toks, apos, cands)
        return float((preds == ans).mean())

    step, history = 0, []
    if ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        step, history = ck["step"], ck["history"]
        log(f"phase6 finetune8 {key}: RESUMED at step {step}")

    head_lost_logged = False
    comp_05_logged = False
    comp_95_logged = False
    t0 = time.time()
    model.train()
    try:
        while step < MAX_STEPS:
            if step and step % POOL_REFRESH == 0:
                pool = [D.sample_base("T1", rng) for _ in range(POOL_SIZE)]
            idx = rng.integers(0, len(pool), size=BATCH)
            gs = [D.PERMS[i] for i in rng.integers(0, len(D.PERMS), size=BATCH)]
            toks, _, _, _ = D.render_batch([pool[i] for i in idx], gs)
            toks = torch.as_tensor(toks, device=device)
            for pg in opt.param_groups:
                pg["lr"] = lr_at(step)
            loss = lm_loss(model(toks), toks)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            step += 1

            if step % EVAL_EVERY == 0 or step == MAX_STEPS:
                model.eval()
                aux_held = {D.token_name(qt): acc(ev_aux[qt]) for qt in AUX_T1}
                aux_min = min(aux_held.values())
                comp_held = acc(ev_comp)
                tr_aux = train_acc(AUX_T1)
                tr_comp = train_acc((D.Q_P,))
                ind_acc = I.copy_accuracy(model, ev_ind, device)
                ind_tab = I.induction_attention(model, ev_ind, device, limit=256)
                li, h = np.unravel_index(ind_tab.argmax(), ind_tab.shape)
                ind_max = float(ind_tab.max())
                pre_mass = float(ind_tab[pre_head])
                ind_alive = bool(
                    ind_max >= I.EDGE_ABS and ind_max >= I.EDGE_REL * I.UNIFORM_BASELINE
                )
                ind_behav = bool(ind_acc > I.BEHAV_THRESH)
                model.train()
                with torch.no_grad():
                    frozen_ok = bool(torch.equal(model.emb_names.detach(), frozen_init))
                rec = dict(
                    step=step,
                    loss=float(loss.item()),
                    aux_held=aux_held,
                    aux_min=aux_min,
                    train_aux=tr_aux,
                    train_composed=tr_comp,
                    held_episode=comp_held,
                    transfer_episode=acc(ev_transfer),
                    ind_copy_acc=ind_acc,
                    ind_max_mass=ind_max,
                    ind_max_head=[int(li), int(h)],
                    ind_pretrained_head_mass=pre_mass,
                    ind_behavioural=ind_behav,
                    ind_mechanistic=ind_alive,
                    frozen_rows_bit_exact=frozen_ok,
                )
                history.append(rec)
                rate = step / max(time.time() - t0, 1e-9)
                log(
                    f"phase6 finetune8 {key}: step {step} loss {rec['loss']:.3f} "
                    f"aux_min {aux_min:.3f} ({aux_held}) "
                    f"train_aux {tr_aux if tr_aux is None else round(tr_aux, 3)} "
                    f"composed_held {comp_held:.3f} transfer {rec['transfer_episode']:.3f} "
                    f"| IND copy {ind_acc:.3f} mass {ind_max:.3f}@L{li}h{h} "
                    f"preHead {pre_mass:.3f} alive={ind_alive} behav={ind_behav} "
                    f"| frozen={frozen_ok} ({rate:.1f} steps/s)"
                )
                if not frozen_ok:
                    log(
                        f"MILESTONE: phase6 finetune8 {key} FREEZE VIOLATION at "
                        f"step {step} — run invalid"
                    )
                    raise RuntimeError("freeze violation")
                if aux_min >= 0.95:
                    log(
                        f"MILESTONE: phase6 finetune8 {key} aux_held_min "
                        f"{aux_min:.3f} >= 0.95 at step {step}"
                    )
                if comp_held >= 0.5 and not comp_05_logged:
                    comp_05_logged = True
                    log(
                        f"MILESTONE: phase6 finetune8 {key} COMPOSED CROSSED 0.5 "
                        f"({comp_held:.3f}) at step {step} — Item A aggregate "
                        f"diagnostic due on this seed at final"
                    )
                if comp_held >= 0.95 and not comp_95_logged:
                    comp_95_logged = True
                    log(
                        f"MILESTONE: phase6 finetune8 {key} BINDING BRANCH — "
                        f"COMPOSED {comp_held:.3f} >= 0.95 held-out at step "
                        f"{step}; stop and report for a decision"
                    )
                if not ind_alive and not head_lost_logged:
                    head_lost_logged = True
                    log(
                        f"MILESTONE: phase6 finetune8 {key} INDUCTION lost "
                        f"mechanistic criterion (pretraining distribution) at "
                        f"step {step} — max {ind_max:.3f} "
                        f"({ind_max / I.UNIFORM_BASELINE:.1f}x), preHead "
                        f"{pre_mass:.3f}, copy {ind_acc:.3f}"
                    )
                torch.save(
                    dict(
                        model=model.state_dict(),
                        opt=opt.state_dict(),
                        step=step,
                        history=history,
                    ),
                    ckpt_path,
                )
                progress.save_json(f"stage2/{key}/trajectory.json", history)

        final = history[-1]
        result = dict(
            seed=seed,
            n_layers=N_LAYERS,
            final=final,
            pretrain_verify=pre_ver,
            max_aux_held_min=max(h_["aux_min"] for h_ in history),
            max_aux_held_any=max(max(h_["aux_held"].values()) for h_ in history),
            max_composed_held=max(h_["held_episode"] for h_ in history),
            max_train_aux=max(
                h_["train_aux"] for h_ in history if h_["train_aux"] is not None
            ),
            max_train_composed=max(
                h_["train_composed"]
                for h_ in history
                if h_["train_composed"] is not None
            ),
            final_ind_alive=final["ind_mechanistic"],
            final_ind_behav=final["ind_behavioural"],
            min_ind_max_mass=min(h_["ind_max_mass"] for h_ in history),
            min_ind_copy_acc=min(h_["ind_copy_acc"] for h_ in history),
        )
        progress.save_json(f"stage2/{key}/finetune_result.json", result)
        log(
            f"MILESTONE: phase6 finetune8 seed{seed} DONE at {MAX_STEPS} — "
            f"final aux_min {final['aux_min']:.3f} (max-min ever "
            f"{result['max_aux_held_min']:.3f}), composed {final['held_episode']:.3f} "
            f"(max ever {result['max_composed_held']:.3f}), transfer "
            f"{final['transfer_episode']:.3f} | induction final: copy "
            f"{final['ind_copy_acc']:.3f} mass {final['ind_max_mass']:.3f} "
            f"alive={final['ind_mechanistic']}"
        )
        return result
    except Exception:
        log(
            f"MILESTONE: PHASE6 FINETUNE8 seed{seed} CRASHED:\n"
            + traceback.format_exc()
        )
        raise
    finally:
        stop_evt.set()
        vol.commit()


@app.function(image=image, timeout=3600, volumes={"/results": vol})
def finalize(seeds: list):
    """Per-seed final query-type breakdown + Check-3-style attention
    diagnostic (property->symbol) on the 8L finals, plus summary json."""
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

    EDGE_ABS, EDGE_REL = 0.25, 5.0
    N_EP = 96
    QUERY_TYPES = [
        ("property->symbol", D.A_PS),
        ("symbol->person", D.A_SN),
        ("person->symbol", D.A_NS),
        ("composed_P", D.Q_P),
    ]

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

    summary = {}
    for seed in seeds:
        key = f"induction8/finetune/seed{seed}"
        ck = torch.load(
            progress.results_dir() / f"stage2/{key}/ckpt.pt", map_location="cpu"
        )
        model = TinyTransformer(seed=seed, n_layers=N_LAYERS)
        model.load_state_dict(ck["model"])
        model.eval()

        breakdown = {}
        for label, qt in QUERY_TYPES:
            ev = D.build_eval_orbits(
                "T1", "fit", 96, seed=9800 + 31 * qt + seed, qtok=qt
            )
            preds = masked_answer_preds(
                model, ev["tokens"], ev["answer_pos"], ev["candidates"]
            )
            from trained.data import orbit_metrics

            m = orbit_metrics(preds, ev)
            breakdown[label] = dict(
                accuracy=m["episode_acc"], orbit_consistency=m["orbit_consistency"]
            )
            log(
                f"phase6 breakdown8 seed{seed}: {label:<18} "
                f"acc={m['episode_acc']:.3f} cons={m['orbit_consistency']:.3f}"
            )
        progress.save_json(f"stage2/{key}/query_breakdown.json", breakdown)

        rng = np.random.default_rng((seed, 424242))
        toks_l, argpos_l, matchpos_l = [], [], []
        for _ in range(N_EP):
            b = D.sample_base("T1", rng, force_qtok=D.A_PS)
            g = D.PERMS[int(rng.integers(len(D.PERMS)))]
            toks, apos, ans, cands = D.render(b, g)
            seq = toks.tolist()
            qpos = seq.index(D.A_PS)
            arg_id = seq[qpos + 1]
            mpos, j = None, 1
            while seq[j] in (D.HAS, D.CARRY, D.GUARD):
                if seq[j] == D.HAS and seq[j + 1] == arg_id:
                    mpos = j + 1
                j += 4
            assert mpos is not None
            toks_l.append(toks)
            argpos_l.append(qpos + 1)
            matchpos_l.append(mpos)
        atts = attention_capture(model, np.stack(toks_l))
        baseline = 1.0 / float(np.mean(np.array(argpos_l) + 1))
        tab = [
            [
                float(
                    np.mean(
                        [
                            float(atts[li][i, h, argpos_l[i], matchpos_l[i]])
                            for i in range(N_EP)
                        ]
                    )
                )
                for h in range(N_HEADS)
            ]
            for li in range(N_LAYERS)
        ]
        arr = np.array(tab)
        li, h = np.unravel_index(arr.argmax(), arr.shape)
        edge = bool(arr.max() >= EDGE_ABS and arr.max() >= EDGE_REL * baseline)
        diag = dict(
            baseline=baseline,
            table=tab,
            max_mass=float(arr.max()),
            max_head=[int(li), int(h)],
            x_baseline=float(arr.max() / baseline),
            edge_present=edge,
        )
        progress.save_json(f"stage2/{key}/final_attention_ps.json", diag)
        log(
            f"phase6 attention8 seed{seed}: A_PS arg->match max {arr.max():.3f} "
            f"at L{li}h{h} ({arr.max()/baseline:.1f}x) -> edge "
            + ("PRESENT" if edge else "ABSENT")
        )

        summary[f"seed{seed}"] = dict(
            finetune=progress.load_json(f"stage2/{key}/finetune_result.json"),
            breakdown=breakdown,
            attention_ps=diag,
        )

    progress.save_json("stage2/induction8/finetune_summary.json", summary)
    log("MILESTONE: PHASE6 FINETUNE8 SUMMARY written")
    vol.commit()
    return summary


@app.local_entrypoint()
def main(seeds: str = "0,1,2"):
    import json

    seed_list = [int(s) for s in seeds.split(",")]
    results = list(finetune.map(seed_list))
    print(json.dumps(results, indent=2, default=str))
    print(json.dumps(finalize.remote(seed_list), indent=2, default=str))

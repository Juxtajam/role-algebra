"""Track B discriminator — d=128 implementation per discriminator_design.md.

Frozen-logic reuse policy:
  - `ridge_fit`            imported UNCHANGED from src/shared/discriminator.py
  - `METRICS`, `LENIENT_QUANTILE`, `CONDITIONS`, `evaluate_conditions`,
    `verdict_and_score`    imported UNCHANGED (the frozen verdict logic)
  - `_quantile_thresholds` imported UNCHANGED from src/shared/calibrate.py
  - `pair_error_mat`, `group_law_metrics_mat`, `support_mass_lex_mat`
                           line-for-line ports of the frozen formulas onto
                           explicit (X, Y, H) matrices (the frozen versions
                           are coupled to the Stage-1 organism interface; the
                           formulas are identical — see the originals at
                           src/shared/discriminator.py lines 129–186)
  - pairing, caching, probes driver, C5 patching: new plumbing, specified in
    discriminator_design.md §§3–6 (primal ridge B1; MOVED-columns carry
    pairing §4.2; splits §5; battery §6).

k = 3 only (D19: k4 UNAVAILABLE). Two position classes: answer,
entity_mention_carry. CPU only.
"""

import sys
import pathlib

import numpy as np
import torch

_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))
from shared.discriminator import (  # noqa: E402  (frozen, unchanged)
    ridge_fit,
    compose,
    _fit_probe,
    _probe_pred,
)

import trained_organism_data as od  # noqa: E402
from trained.model import TinyTransformer, masked_answer_preds  # noqa: E402

LAM_GRID = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]
GEN3 = od.GEN3
MOVED3 = od.MOVED3
PERMS3 = od.PERMS3
N_LAYERS = 8
D = 128


# ---------------------------------------------------------------------------
# Activation caching
# ---------------------------------------------------------------------------


def load_model(seed, arm, ckpt_path):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = TinyTransformer(seed=seed, n_layers=N_LAYERS)
    model.load_state_dict(ck["model"])
    model.eval()
    return model


@torch.no_grad()
def capture_states(model, ev, batch=256):
    """Returns answer-position states (n_ep, L, d) and carry states
    (n_ep, k, L, d), fp32, resid_post per block (L = 8)."""
    toks = ev["tokens"]
    apos = ev["answer_pos"]
    cpos = ev["carry_pos"]
    n = len(toks)
    k = cpos.shape[1]
    ans = np.zeros((n, N_LAYERS, D), dtype=np.float32)
    car = np.zeros((n, k, N_LAYERS, D), dtype=np.float32)
    for i in range(0, n, batch):
        t = torch.as_tensor(toks[i : i + batch])
        _, resids = model(t, capture=True)
        rows = torch.arange(len(t))
        ap = torch.as_tensor(apos[i : i + batch])
        for li, r in enumerate(resids):
            ans[i : i + batch, li] = r[rows, ap].numpy()
            for s in range(k):
                cp = torch.as_tensor(cpos[i : i + batch, s])
                car[i : i + batch, s, li] = r[rows, cp].numpy()
    return ans, car


@torch.no_grad()
def predictions(model, ev):
    return masked_answer_preds(model, ev["tokens"], ev["answer_pos"], ev["candidates"])


def strict_ok_bases(preds, ev):
    """Base indices where all 6 orbit episodes are answered correctly."""
    n, n_perm = ev["n_bases"], len(PERMS3)
    correct = (preds == ev["answers"]).reshape(n, n_perm)
    return np.where(correct.all(axis=1))[0]


# ---------------------------------------------------------------------------
# Pairing (episode index arithmetic: eval sets are base-major, PERMS3 order)
# ---------------------------------------------------------------------------

PERM_IDX = {p: i for i, p in enumerate(PERMS3)}


def ep_index(base_idx, g):
    return base_idx * len(PERMS3) + PERM_IDX[g]


def pairs_answer(bases, a):
    """(x_idx, y_idx) episode pairs for generator a at the answer position:
    x = episode g, y = episode a∘g, all g, all bases."""
    xs, ys = [], []
    for b in bases:
        for g in PERMS3:
            xs.append(ep_index(b, g))
            ys.append(ep_index(b, compose(a, g)))
    return np.array(xs), np.array(ys)


def pairs_carry(bases, a):
    """((x_idx, slot), (y_idx, slot)) for generator a at carry positions:
    per design §4.2 only the MOVED columns of a are sampled; pairing is
    same-column across (g, a∘g)."""
    xs, ys, ss = [], [], []
    for b in bases:
        for g in PERMS3:
            for s in MOVED3[a]:
                xs.append(ep_index(b, g))
                ys.append(ep_index(b, compose(a, g)))
                ss.append(s)
    return np.array(xs), np.array(ys), np.array(ss)


def XY_answer(acts, bases, a, layer):
    xi, yi = pairs_answer(bases, a)
    return acts[xi, layer], acts[yi, layer]


def XY_carry(carr, bases, a, layer):
    xi, yi, ss = pairs_carry(bases, a)
    return carr[xi, ss, layer], carr[yi, ss, layer]


def fit_generators(get_XY, bases, lam, null_mode=None, null_seed=0):
    """fit_maps port: per-generator ridge on (X, Y) with frozen null modes
    (shuffled: Y rows re-paired, frozen seed convention; identity: Y := X)."""
    rng = np.random.default_rng(null_seed)
    Rs = {}
    for a in GEN3:
        X, Y = get_XY(bases, a)
        if null_mode == "identity":
            Y = X.copy()
        elif null_mode == "shuffled":
            Y = Y[rng.permutation(len(Y))]
        Rs[a] = ridge_fit(X.astype(np.float64), Y.astype(np.float64), lam)
    return Rs


# ---------------------------------------------------------------------------
# Metric formulas — line-for-line ports onto explicit matrices
# ---------------------------------------------------------------------------


def pair_error_mat(Rs, get_XY, bases):
    """Port of frozen pair_error: relative error on genuine (x, g·x) pairs;
    identity map scores 1."""
    errs = []
    for a, R in Rs.items():
        X, Y = get_XY(bases, a)
        X, Y = X.astype(np.float64), Y.astype(np.float64)
        errs.append(((X @ R.T - Y) ** 2).sum() / max(((Y - X) ** 2).sum(), 1e-12))
    return float(np.mean(errs))


def group_law_metrics_mat(Rs, H):
    """Port of frozen group_law_metrics (identical formulas)."""
    H = H.astype(np.float64)
    hn = (H**2).sum()
    Rlist = [Rs[a] for a in GEN3]
    inv = np.mean([((H @ (R.T @ R.T - np.eye(len(R)))) ** 2).sum() / hn for R in Rlist])
    R12, R23 = Rlist[0], Rlist[1]
    A, B = R12 @ R23 @ R12, R23 @ R12 @ R23
    braid_den = 0.5 * (((H @ A.T) ** 2).sum() + ((H @ B.T) ** 2).sum()) + 1e-12 * hn
    braid = ((H @ (A - B).T) ** 2).sum() / braid_den
    nontriv = np.mean([((H @ (R - np.eye(len(R))).T) ** 2).sum() / hn for R in Rlist])
    noncomm = ((H @ (R12 @ R23 - R23 @ R12).T) ** 2).sum() / hn
    return dict(
        law_inv_defect=float(inv),
        law_braid_defect=float(braid),
        nontriv=float(nontriv),
        noncommute=float(noncomm),
    )


def support_mass_lex_mat(Rs, model, ev, bases, max_eps=48):
    """Port of frozen support_and_rank, Stage-2 lexical variant: fraction of
    (R − I) row-space mass inside the episode-local difference span of the
    episode's k name READOUT rows (tied model: readout row = emb row)."""
    with torch.no_grad():
        emb = model.emb.numpy().astype(np.float64)
    ep_idx = [ep_index(b, PERMS3[0]) for b in bases][:max_eps]
    masses, ranks = [], []
    for R in Rs.values():
        _, S, Vt = np.linalg.svd(R - np.eye(len(R)))
        w = S**2 / max((S**2).sum(), 1e-12)
        p = w / w.sum()
        ranks.append(float(np.exp(-(p * np.log(p + 1e-30)).sum())))
        per_ep = []
        for i in ep_idx:
            U = emb[ev["candidates"][i]]  # (k, d) readout rows
            diffs = U[1:] - U[:1]
            q, _ = np.linalg.qr(
                diffs.T
                + 1e-12 * np.random.default_rng(0).standard_normal(diffs.T.shape)
            )
            per_ep.append((w * ((Vt @ q) ** 2).sum(axis=1)).sum())
        masses.append(float(np.mean(per_ep)))
    return dict(mass=float(np.mean(masses)), eff_rank=float(np.mean(ranks)))


# ---------------------------------------------------------------------------
# C5 — causal transport (patch-and-continue), native model support
# ---------------------------------------------------------------------------


@torch.no_grad()
def transport_agreement(
    model, Rs, ev, acts, carr, bases, layer, position, per_base=2, batch=256
):
    """Patch R_g h_l(x) into the y-episode's forward pass at `layer` and the
    position, decode, compare to the natural decode of y (never the
    unembedding directly). position: 'answer' or ('carry', per-generator
    MOVED columns)."""
    agrees = []
    for a, R in Rs.items():
        if position == "answer":
            xi, yi = pairs_answer(bases, a)
            sel = np.concatenate(
                [np.arange(i * 6, i * 6 + per_base) for i in range(len(bases))]
            )
            xi, yi = xi[sel], yi[sel]
            Xs = acts[xi, layer]
            pos_y = ev["answer_pos"][yi]
        else:
            xi, yi, ss = pairs_carry(bases, a)
            sel = np.concatenate(
                [np.arange(i * 12, i * 12 + per_base) for i in range(len(bases))]
            )
            xi, yi, ss = xi[sel], yi[sel], ss[sel]
            Xs = carr[xi, ss, layer]
            pos_y = ev["carry_pos"][yi, ss]
        patched = (Xs.astype(np.float64) @ R.T).astype(np.float32)
        toks_y = ev["tokens"][yi]
        cands_y = ev["candidates"][yi]
        apos_y = ev["answer_pos"][yi]
        pred_patch = np.zeros(len(yi), dtype=np.int64)
        for i in range(0, len(yi), batch):
            t = torch.as_tensor(toks_y[i : i + batch])
            logits = model(
                t,
                patch=(
                    layer,
                    torch.as_tensor(pos_y[i : i + batch]),
                    torch.as_tensor(patched[i : i + batch]),
                ),
            )
            rows = torch.arange(len(t))
            sel_l = logits[rows, torch.as_tensor(apos_y[i : i + batch])]
            cd = torch.as_tensor(cands_y[i : i + batch])
            cl = sel_l.gather(1, cd)
            pred_patch[i : i + batch] = (
                cd.gather(1, cl.argmax(1, keepdim=True)).squeeze(1).numpy()
            )
        pred_nat = predictions(
            model,
            dict(
                tokens=ev["tokens"][yi],
                answer_pos=ev["answer_pos"][yi],
                candidates=ev["candidates"][yi],
            ),
        )
        agrees.append(float(np.mean(pred_patch == pred_nat)))
    return dict(transport_agree=float(np.mean(agrees)))


# ---------------------------------------------------------------------------
# C6 — probes (frozen _fit_probe/_probe_pred imported)
# ---------------------------------------------------------------------------


def probe_metrics_mat(
    Rs, states_tr, states_ev, content_tr, content_ev, role_tr, role_ev, k=3
):
    """Port of frozen probe_metrics onto explicit matrices."""
    nc = int(max(content_tr.max(), content_ev.max())) + 1
    pc = _fit_probe(states_tr.astype(np.float64), content_tr, nc)
    pr = _fit_probe(states_tr.astype(np.float64), role_tr, k)
    Xev = states_ev.astype(np.float64)
    c_acc = float(np.mean(_probe_pred(pc, Xev) == content_ev))
    r_acc = float(np.mean(_probe_pred(pr, Xev) == role_ev))
    available = c_acc >= 0.9 and r_acc >= 0.9
    keep, perm = [], []
    for a, R in Rs.items():
        base_c = _probe_pred(pc, Xev)
        base_r = _probe_pred(pr, Xev)
        keep.append(float(np.mean(_probe_pred(pc, Xev @ R.T) == base_c)))
        a_arr = np.array(a)
        perm.append(float(np.mean(_probe_pred(pr, Xev @ R.T) == a_arr[base_r])))
    return dict(
        probe_content_acc=c_acc,
        probe_role_acc=r_acc,
        probes_available=bool(available),
        probe_content_keep=float(np.mean(keep)) if available else None,
        probe_role_perm=float(np.mean(perm)) if available else None,
    )


def first_decodable_layer_mat(states_by_layer_tr, y_tr, states_by_layer_ev, y_ev, k=3):
    """Port of frozen first_decodable_layer onto cached states."""
    for layer in range(N_LAYERS):
        p = _fit_probe(states_by_layer_tr[:, layer].astype(np.float64), y_tr, k)
        acc = float(
            np.mean(
                _probe_pred(p, states_by_layer_ev[:, layer].astype(np.float64)) == y_ev
            )
        )
        if acc >= 0.9:
            return layer
    return N_LAYERS - 1

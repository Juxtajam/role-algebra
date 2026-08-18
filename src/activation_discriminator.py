"""Phase 8C — core library: cached-activation system + FACTORED (dual-form)
implementation of the frozen discriminator formulations (src/shared/
discriminator.py, spec v3).

Mathematical fidelity: the ridge fit
    R = argmin ||X R^T - Y||_F^2 + lam_eff ||R||_F^2,
    lam_eff = lam * tr(X^T X)/d + 1e-12          (frozen scale-free form)
has the exact dual representation (push-through identity)
    R = Y^T M X,   M = (X X^T + lam_eff I_n)^{-1},
so every activation-weighted metric H @ f(R)^T is computed in factored form
without materialising the d x d matrix. Identical formulation, not an
approximation; verified against src.shared.discriminator.ridge_fit in
phase8c_fidelity.py before any real fitting.

Everything here is CPU/numpy float64. Never parameterises R into any known
representation (standing rule): the fit is the unconstrained ridge above.
"""

import hashlib
import itertools
import json
import pathlib

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
TASKS = ROOT / "results/verdict/gate/tasks"
ACTS = ROOT / "results/verdict/answer_position/acts"
OUT = ROOT / "results/verdict/discriminator"

K = 3
PERMS = [tuple(p) for p in itertools.permutations(range(K))]  # frozen order
PERM_IDX = {p: i for i, p in enumerate(PERMS)}
GENERATORS = [(1, 0, 2), (0, 2, 1)]  # adjacent transpositions (12), (23)
GEN_NAMES = ["R12", "R23"]
N_LAYERS = 80
D_MODEL = 8192
LAM_GRID = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]  # frozen Stage-1 grid
MARKS = [
    "crimson",
    "amber",
    "violet",
    "copper",
    "ivory",
    "scarlet",
    "golden",
    "silver",
    "maroon",
    "indigo",
    "teal",
    "coral",
]
BOOT_SEED = 20260807
N_BOOT = 10_000


def compose(a, b):
    """(a o b)(j) = a(b(j)) — frozen convention (src)."""
    return tuple(a[b[j]] for j in range(len(a)))


def sha_file(p):
    return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()


# ---------------------------------------------------------------- data layer
class Cell:
    """One disc cell: episode records + mmap'd activations (n_eps, 80, 8192).
    Row index = base_id * 6 + PERM_IDX[g]."""

    def __init__(self, path, vocab):
        self.path, self.vocab = path, vocab
        fp = TASKS / f"disc_{path}_{vocab}.jsonl"
        self.recs = [json.loads(l) for l in open(fp)]
        self.acts = np.load(ACTS / f"disc_{path}_{vocab}.npy", mmap_mode="r")
        assert self.acts.shape == (len(self.recs), N_LAYERS, D_MODEL)
        preds = json.load(
            open(ROOT / "results/verdict/answer_position" / f"preds_disc_{path}_{vocab}.json")
        )
        assert len(preds) == len(self.recs)
        self.preds = preds
        # verify row layout: rec i has base_id i//6 and g == PERMS[i%6]
        for i in (0, 1, 7, 1793):
            r = self.recs[i]
            assert r["base_id"] == i // 6 and tuple(r["g"]) == PERMS[i % 6]
        self.n_bases = len(self.recs) // 6
        # strict-orbit-correct bases (all 6 episodes answered correctly);
        # behavioural labels from the 8A-final session (preds_disc_*)
        self.strict_ok = []
        for b in range(self.n_bases):
            ok = all(
                self.preds[b * 6 + j] == self.recs[b * 6 + j]["answer"]
                for j in range(6)
            )
            if ok:
                self.strict_ok.append(b)

    def row(self, base, g):
        return base * 6 + PERM_IDX[tuple(g)]

    def states(self, rows, layer):
        return np.asarray(self.acts[rows, layer, :], dtype=np.float64)

    def rec(self, base, g):
        return self.recs[self.row(base, g)]


def load_cells():
    return {(p, v): Cell(p, v) for p in ("P", "G") for v in ("fit", "transfer")}


def pair_rows(cell, bases, a):
    """Frozen _pair_eps semantics: for each base, for each g in all 6 perms,
    pair (episode g, episode a o g). Returns (rows_x, rows_y)."""
    rx, ry = [], []
    for b in bases:
        for g in PERMS:
            rx.append(cell.row(b, g))
            ry.append(cell.row(b, compose(a, g)))
    return np.array(rx), np.array(ry)


def role_labels(cell, rows):
    """k-way answer label: which of the base's canonical persons is the
    answer = g[answer_slot]. Under pairing x -> a.x the label permutes by a
    (label(a o g) = a[label(g)]), matching the frozen probe_role_perm test."""
    return np.array(
        [cell.recs[i]["g"][cell.recs[i]["answer_slot"]] for i in rows], dtype=int
    )


def vocab_names(cell):
    pools = json.load(open(TASKS / "name_pools.json"))
    return pools[cell.vocab]


def name_labels(cell, rows, names):
    """12-way answer-name label (index into the cell's vocab pool)."""
    ni = {n: i for i, n in enumerate(names)}
    return np.array([ni[cell.recs[i]["answer"]] for i in rows], dtype=int)


def candidate_matrix(cell, rows, names):
    """(n, k) vocab-pool indices of the episode's canonical names[0..2]."""
    ni = {n: i for i, n in enumerate(names)}
    return np.array(
        [[ni[m] for m in cell.recs[i]["base"]["names"]] for i in rows], dtype=int
    )


def episode_local_pred(probe, X, cand):
    """Episode-local k-way readout of a 12-way name probe: restrict logits
    to the episode's k candidate names, return the SLOT index (0..k-1).
    Chance = 1/k."""
    Z = ((X - probe["mu"]) / probe["sd"]) @ probe["W"] + probe["b"]
    return np.take_along_axis(Z, cand, axis=1).argmax(1)


def content_labels(cell, rows):
    """g-invariant, vocabulary-independent episode content: the queried mark
    (path P), 12 classes shared across fit/transfer vocabularies."""
    out = []
    for i in rows:
        r = cell.recs[i]
        assert r["path"] == "P"
        out.append(MARKS.index(r["base"]["marks"][r["base"]["qi"]]))
    return np.array(out, dtype=int)


# ------------------------------------------------------------- dual ridge fit
class DualRidge:
    """Factored representation of the frozen ridge fit R = Y^T M X.
    apply_RT(V) = V @ R.T; apply_R_cols(Q) = R @ Q; traces for ||R - I||_F^2."""

    def __init__(self, X, Y, lam):
        n, d = X.shape
        self.X, self.Y, self.d, self.n = X, Y, d, n
        self.lam = lam
        self.lam_eff = lam * float((X * X).sum()) / d + 1e-12
        XXt = X @ X.T
        s, U = np.linalg.eigh(XXt)
        self.eig_s = s
        self.M = (U / (s + self.lam_eff)) @ U.T  # (XX^T + lam I)^-1
        self.MX = self.M @ X  # n x d (cached)
        self.cond_eff = float((s[-1] + self.lam_eff) / (s[0] + self.lam_eff))

    def apply_RT(self, V, VXt=None):
        """V @ R.T = V @ X^T M Y (m x d)."""
        t = V @ self.X.T if VXt is None else VXt
        return (t @ self.M) @ self.Y

    def apply_R_cols(self, Q):
        """R @ Q = Y^T M (X Q) (d x q)."""
        return self.Y.T @ (self.M @ (self.X @ Q))

    def trace_R(self):
        return float((self.MX * self.Y).sum())  # tr(Y^T M X)

    def frob2_R(self):
        # ||R||_F^2 = tr(M XX^T M Y Y^T): with A = X^T M (d x n... use
        # B = (MX) (MX)^T? tr(Y^T M X X^T M Y) = || (M X) ... simpler:
        # T = X @ (M Y ... ) — do: C = self.MX @ self.MX.T? that's
        # tr(Y^T M X X^T M Y) = sum over (M X X^T M) * (Y Y^T).
        G = self.MX @ self.X.T  # M X X^T  (n x n)
        A = G @ self.M  # M X X^T M
        return float((A * (self.Y @ self.Y.T)).sum())

    def frob2_R_minus_I(self):
        return self.frob2_R() - 2.0 * self.trace_R() + self.d


def fit_generators(cell_fit, bases, layer, lam, null_mode=None, null_seed=0):
    """Frozen fit_maps semantics on cached activations. Returns
    {gen_tuple: DualRidge}. Null modes corrupt only the fit:
      'shuffled' — Y rows re-paired at random (frozen rng convention)
      'identity' — Y := states(xs) (g = identity)."""
    rng = np.random.default_rng(null_seed)
    Rs = {}
    for a in GENERATORS:
        rx, ry = pair_rows(cell_fit, bases, a)
        X = cell_fit.states(rx, layer)
        if null_mode == "identity":
            Y = cell_fit.states(rx, layer)
        else:
            Y = cell_fit.states(ry, layer)
            if null_mode == "shuffled":
                Y = Y[rng.permutation(len(Y))]
        Rs[a] = DualRidge(X, Y, lam)
    return Rs


# ------------------------------------------------------------------- metrics
def pair_error(Rs, cell_eval, bases, layer, per_base=False):
    """Frozen pair_error: relative prediction error on genuine (x, g.x)
    pairs; identity map scores 1. Mean over generators. If per_base, also
    returns per-base numerators/denominators for bootstrap/reproduction."""
    errs, pb = [], []
    for a, R in Rs.items():
        rx, ry = pair_rows(cell_eval, bases, a)
        X, Y = cell_eval.states(rx, layer), cell_eval.states(ry, layer)
        P = R.apply_RT(X)
        num = ((P - Y) ** 2).sum(axis=1)
        den = ((Y - X) ** 2).sum(axis=1)
        errs.append(num.sum() / max(den.sum(), 1e-12))
        if per_base:
            nb = len(bases)
            pb.append(
                dict(
                    num=num.reshape(nb, 6).sum(1).tolist(),
                    den=den.reshape(nb, 6).sum(1).tolist(),
                )
            )
    m = float(np.mean(errs))
    return (m, pb) if per_base else m


def group_law_metrics(Rs, cell_eval, bases, layer, per_base=False):
    """Frozen activation-weighted defects on held-out states."""
    rows = [cell_eval.row(b, g) for b in bases for g in PERMS]
    H = cell_eval.states(np.array(rows), layer)
    hn_ep = (H**2).sum(axis=1)
    hn = hn_ep.sum()
    Rl = [Rs[a] for a in GENERATORS]
    HXt = {id(R): H @ R.X.T for R in Rl}

    def ap(V, R):  # V @ R.T
        return R.apply_RT(V, VXt=(HXt[id(R)] if V is H else None))

    # involution: || H (R^T R^T - I) ||^2 / hn  per generator, mean
    inv_ep = []
    for R in Rl:
        D = ap(ap(H, R), R) - H
        inv_ep.append((D**2).sum(axis=1))
    R12, R23 = Rl
    # braid words A = R12 R23 R12, B = R23 R12 R23 (activation-weighted)
    HA = ap(ap(ap(H, R12), R23), R12)
    HB = ap(ap(ap(H, R23), R12), R23)
    braid_num_ep = ((HA - HB) ** 2).sum(axis=1)
    braid_den_ep = 0.5 * ((HA**2).sum(axis=1) + (HB**2).sum(axis=1))
    braid = braid_num_ep.sum() / (braid_den_ep.sum() + 1e-12 * hn)
    nontriv_ep = [((ap(H, R) - H) ** 2).sum(axis=1) for R in Rl]
    HC = ap(ap(H, R23), R12) - ap(ap(H, R12), R23)  # H (R12R23 - R23R12)^T
    noncomm_ep = (HC**2).sum(axis=1)
    m = dict(
        law_inv_defect=float(np.mean([e.sum() / hn for e in inv_ep])),
        law_braid_defect=float(braid),
        nontriv=float(np.mean([e.sum() / hn for e in nontriv_ep])),
        noncommute=float(noncomm_ep.sum() / hn),
    )
    if per_base:
        nb = len(bases)
        pb = dict(
            hn=hn_ep.reshape(nb, 6).sum(1).tolist(),
            inv=[e.reshape(nb, 6).sum(1).tolist() for e in inv_ep],
            braid_num=braid_num_ep.reshape(nb, 6).sum(1).tolist(),
            braid_den=braid_den_ep.reshape(nb, 6).sum(1).tolist(),
            nontriv=[e.reshape(nb, 6).sum(1).tolist() for e in nontriv_ep],
            noncomm=noncomm_ep.reshape(nb, 6).sum(1).tolist(),
        )
        return m, pb
    return m


def support_mass_lexical(Rs, cell_eval, bases, name_vectors, max_eps=48, per_ep=False):
    """Frozen support_and_rank, Stage-2 lexical variant: fraction of (R - I)
    row-space mass inside the EPISODE-LOCAL difference subspace of the
    episode's k name readout vectors (never the span of all name vectors).
    mass = ||(R - I) q||_F^2 / ||R - I||_F^2 with q an orthonormal basis of
    span{u_i - u_1} (exact identity with the frozen SVD-weighted form).
    Direction lt: high mass = readout-level lexical-swap artifact."""
    sel = bases[:max_eps]
    Qs = []
    jit = np.random.default_rng(0)
    for b in sel:
        r = cell_eval.rec(b, PERMS[0])
        U = np.stack([name_vectors[n] for n in r["base"]["names"]])  # (k, d)
        diffs = (U[1:] - U[:1]).T  # (d, k-1)
        q, _ = np.linalg.qr(diffs + 1e-12 * jit.standard_normal(diffs.shape))
        Qs.append(q)
    masses, per = [], []
    for a, R in Rs.items():
        f2 = R.frob2_R_minus_I()
        RQ = R.apply_R_cols(np.concatenate(Qs, axis=1))  # d x (2*len(sel))
        vals = []
        for i, b in enumerate(sel):
            D = RQ[:, 2 * i : 2 * i + 2] - Qs[i]
            vals.append(float((D**2).sum() / max(f2, 1e-12)))
        masses.append(float(np.mean(vals)))
        per.append(vals)
    m = float(np.mean(masses))
    return (m, per) if per_ep else m


def eff_rank_R_minus_I(R):
    """Diagnostic ONLY (frozen note: nothing may depend on it). Exact
    spectrum of R - I via the low-rank structure: col(R) in col(Y^T),
    col(R^T) in col(X^T), so (R-I)(R-I)^T = I + W with rank(W) <= 2n."""
    B = np.concatenate([R.X.T, R.Y.T], axis=1)  # d x 2n
    Qb, _ = np.linalg.qr(B)  # d x r
    RQ = R.apply_R_cols(Qb)  # R Qb   (d x r)
    RtQ = R.apply_RT(Qb.T).T  # R^T Qb (d x r)
    # (R-I)(R-I)^T = I + W, W = R R^T - R - R^T, and col(W) in span(Qb):
    # Wp = Qb^T W Qb = (R^T Qb)^T (R^T Qb) - Qb^T R Qb - (Qb^T R Qb)^T
    A1 = RtQ.T @ RtQ  # Qb^T R R^T Qb
    A2 = Qb.T @ RQ  # Qb^T R Qb
    Wp = A1 - A2 - A2.T
    mu = np.linalg.eigvalsh(Wp)
    s2 = np.concatenate([1.0 + mu, np.ones(R.d - len(mu))])
    s2 = np.clip(s2, 0, None)
    w = s2 / max(s2.sum(), 1e-12)
    p = w / w.sum()
    return float(np.exp(-(p * np.log(p + 1e-30)).sum()))


# -------------------------------------------------------------------- probes
def fit_probe(X, y, n_classes, iters=300, lr=1.0, l2=1e-4):
    """Frozen _fit_probe verbatim (multinomial logistic, standardized)."""
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Xn = (X - mu) / sd
    n, d = Xn.shape
    W = np.zeros((d, n_classes))
    b = np.zeros(n_classes)
    Yoh = np.eye(n_classes)[y]
    for _ in range(iters):
        Z = Xn @ W + b
        Z -= Z.max(1, keepdims=True)
        P = np.exp(Z)
        P /= P.sum(1, keepdims=True)
        G = (P - Yoh) / n
        W -= lr * (Xn.T @ G + l2 * W)
        b -= lr * G.sum(0)
    return dict(W=W, b=b, mu=mu, sd=sd)


def probe_pred(probe, X):
    return (((X - probe["mu"]) / probe["sd"]) @ probe["W"] + probe["b"]).argmax(1)


def probe_metrics(Rs, cells, cal_bases_Pfit, eval_bases_Ptr, layer, per_ep=False):
    """Frozen probe_metrics: probes trained on the calibration split (P/fit
    cal), evaluated on the eval split (P/transfer); availability >= 0.9."""
    cf, ct = cells[("P", "fit")], cells[("P", "transfer")]
    tr_rows = np.array([cf.row(b, g) for b in cal_bases_Pfit for g in PERMS])
    ev_rows = np.array([ct.row(b, g) for b in eval_bases_Ptr for g in PERMS])
    Xtr, Xev = cf.states(tr_rows, layer), ct.states(ev_rows, layer)
    c_tr, c_ev = content_labels(cf, tr_rows), content_labels(ct, ev_rows)
    r_tr, r_ev = role_labels(cf, tr_rows), role_labels(ct, ev_rows)
    nc = int(max(c_tr.max(), c_ev.max())) + 1
    pc = fit_probe(Xtr, c_tr, nc)
    pr = fit_probe(Xtr, r_tr, K)
    c_acc = float(np.mean(probe_pred(pc, Xev) == c_ev))
    r_acc = float(np.mean(probe_pred(pr, Xev) == r_ev))
    available = c_acc >= 0.9 and r_acc >= 0.9
    keep, perm, keep_ep, perm_ep = [], [], [], []
    base_c = probe_pred(pc, Xev)
    base_r = probe_pred(pr, Xev)
    for a, R in Rs.items():
        XevR = R.apply_RT(Xev)
        kbits = probe_pred(pc, XevR) == base_c
        pbits = probe_pred(pr, XevR) == np.array(a)[base_r]
        keep.append(float(kbits.mean()))
        perm.append(float(pbits.mean()))
        keep_ep.append(kbits.astype(int).tolist())
        perm_ep.append(pbits.astype(int).tolist())
    m = dict(
        probe_content_acc=c_acc,
        probe_role_acc=r_acc,
        probes_available=bool(available),
        probe_content_keep=float(np.mean(keep)) if available else None,
        probe_role_perm=float(np.mean(perm)) if available else None,
    )
    if per_ep:
        return m, dict(keep=keep_ep, perm=perm_ep, probes=(pc, pr))
    return m


def decodability_curve(cell, cal_bases, layers=range(N_LAYERS)):
    """instruction B: ANSWER-decodability per layer, CAL split only.
    Frozen first_decodable_layer semantics (cal bases halved, linear probe,
    held-out acc >= 0.9), adapted to the pretrained LM (declared): the
    Stage-1 k-way slot probe is inexpressible here because slot j names a
    DIFFERENT token in every base, so answer decodability is measured with
    a 12-way answer-NAME probe read episode-locally (logits restricted to
    the episode's k candidate names -> slot; chance 1/k, threshold 0.9
    unchanged). The raw global slot-probe curve is logged as a diagnostic.
    """
    names = vocab_names(cell)
    half = len(cal_bases) // 2
    tr = np.array([cell.row(b, g) for b in cal_bases[:half] for g in PERMS])
    ev = np.array([cell.row(b, g) for b in cal_bases[half:] for g in PERMS])
    y_tr = name_labels(cell, tr, names)
    cand_ev = candidate_matrix(cell, ev, names)
    slot_ev = np.array(
        [list(cell.recs[i]["base"]["names"]).index(cell.recs[i]["answer"]) for i in ev]
    )
    y_slot_tr, y_slot_ev = role_labels(cell, tr), role_labels(cell, ev)
    curve, slot_curve = [], []
    for layer in layers:
        Xtr, Xev = cell.states(tr, layer), cell.states(ev, layer)
        p = fit_probe(Xtr, y_tr, len(names))
        acc = float(np.mean(episode_local_pred(p, Xev, cand_ev) == slot_ev))
        curve.append(acc)
        ps = fit_probe(Xtr, y_slot_tr, K)
        slot_curve.append(float(np.mean(probe_pred(ps, Xev) == y_slot_ev)))
    first = next((l for l, a in zip(layers, curve) if a >= 0.9), N_LAYERS - 1)
    return curve, first, slot_curve


# --------------------------------------------------------- spectrum (8B) ----
def spectrum_diagnostic(Rs, cell_cal, cal_bases, layer, rank_cut=0.10, centered=False):
    """8B carry-forward: u_j = slot-conditional activation means on the CAL
    split at the frozen layer; SVD rank cut; restricted generators
    A_g = B^T R_g B; spectra + min singular value of stacked [A_g - I]."""
    rows = np.array([cell_cal.row(b, g) for b in cal_bases for g in PERMS])
    X = cell_cal.states(rows, layer)
    lab = role_labels(cell_cal, rows)
    U = np.stack([X[lab == j].mean(0) for j in range(K)])  # (k, d)
    if centered:
        U = U - U.mean(0, keepdims=True)
    _, S, Vt = np.linalg.svd(U, full_matrices=False)
    r = int((S >= rank_cut * S[0]).sum())
    B = Vt[:r].T  # d x r
    out = dict(sing_vals_u=S.tolist(), span_dim=r, sv_ratios=(S / S[0]).tolist())
    As = {}
    for a, R in Rs.items():
        A = B.T @ R.apply_R_cols(B)  # r x r
        As[a] = A
        ev = np.linalg.eigvals(A)
        out[f"spectrum_{GEN_NAMES[GENERATORS.index(a)]}"] = dict(
            real=np.real(ev).tolist(), imag=np.imag(ev).tolist()
        )
    stacked = np.concatenate([As[a] - np.eye(r) for a in GENERATORS], axis=0)
    sv = np.linalg.svd(stacked, compute_uv=False)
    out["stacked_min_sv"] = float(sv[-1])
    out["stacked_svs"] = sv.tolist()
    out["top_sv_A"] = float(
        np.median([np.linalg.svd(As[a], compute_uv=False)[0] for a in GENERATORS])
    )
    return out

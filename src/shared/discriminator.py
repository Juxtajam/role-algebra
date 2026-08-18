"""The discriminator (spec v3, "Discriminator (both stages)").

Fitting: unconstrained d x d ridge least squares between matched pairs
(h_l(x), h_l(g.x)) for each adjacent transposition g, per layer, never
parameterised into a permutation or known representation. lambda comes from
a fixed grid selected on the calibration split only and is then frozen.

Every check is a frozen numeric threshold from calibration (no "~=", no
"exceeds baseline"). Null modes used by calibration:
  'shuffled'  — fit pairs with Y rows re-paired at random
  'identity'  — fit pairs (h(x), h'(x)) with g = identity
plus the full pipeline run on S-retrieval. Evaluation pairs are always
genuine (x, g.x) pairs; null modes corrupt only the fit.
"""

import numpy as np

from shared.progress import log

LAM_GRID = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]

# metric -> (pass direction, null sources used for its threshold)
METRICS = {
    "content_transfer_err": ("lt", ("shuffled", "retrieval")),
    "crosspath_err": ("lt", ("shuffled", "retrieval")),
    "law_inv_defect": ("lt", ("shuffled", "retrieval")),
    "law_braid_defect": ("lt", ("shuffled", "retrieval")),
    "nontriv": ("gt", ("identity",)),
    "noncommute": ("gt", ("identity",)),
    # Two separately configured support tests over the same quantity — the
    # fraction of (R - I) row-space mass inside span{u_i - u_j}:
    #  * Stage 1 ("support_mass", gt): u_i are the role vectors, the subspace
    #    is the role-difference subspace, and R SHOULD act there.
    #  * Stage 2 ("support_mass_lex", lt): u_i are the episode's name
    #    embeddings, and high mass there is the readout-level lexical-swap
    #    artifact this test exists to detect. Its threshold sits at the far
    #    (1 - FPR) end of the Stage 2 null distribution (nulls model
    #    non-artifact maps, whose mass is uniformly small; a lexical swap
    #    concentrates far beyond that range). C4's false-positive rate stays
    #    controlled by nontriv/noncommute.
    "support_mass": ("gt", ("identity", "retrieval")),
    "support_mass_lex": ("lt", ("shuffled", "identity")),
    "transport_agree": ("gt", ("shuffled", "retrieval", "identity")),
    "probe_content_keep": ("gt", ("shuffled", "retrieval")),
    "probe_role_perm": ("gt", ("shuffled", "retrieval")),
}

# Degenerate maps satisfy the braid relation trivially (any R12 ~ R23 gives a
# zero braid defect), so a low-quantile braid threshold collapses to noise
# level. Condition 3's false-positive rate is controlled at the target FPR by
# the involution defect; the braid is a conjunctive outlier check whose
# threshold sits at the far (1 - FPR) end of the null braid distribution.
LENIENT_QUANTILE = {"law_braid_defect", "support_mass_lex"}

CONDITIONS = {
    "C1_content_transfer": ["content_transfer_err"],
    "C2_crosspath": ["crosspath_err"],
    "C3_group_laws": ["law_inv_defect", "law_braid_defect"],
    "C4_nontriviality": ["nontriv", "noncommute", "support_mass"],
    "C5_causal_transport": ["transport_agree"],
    "C6_content_preserve": ["probe_content_keep", "probe_role_perm"],
}
MANDATORY = ("C1_content_transfer", "C2_crosspath", "C5_causal_transport")


def compose(a, b):
    return tuple(a[b[j]] for j in range(len(a)))


def ridge_fit(X, Y, lam):
    """R minimising ||X R^T - Y||_F^2 + lam_eff ||R||_F^2, scale-free lambda."""
    d = X.shape[1]
    G = X.T @ X
    lam_eff = lam * np.trace(G) / d + 1e-12
    return np.linalg.solve(G + lam_eff * np.eye(d), X.T @ Y).T


def _pair_eps(system, bases, a):
    xs, ys = [], []
    for b in bases:
        orb = system.orbit(b)
        for g, ep in orb.items():
            xs.append(ep)
            ys.append(orb[compose(a, g)])
    return xs, ys


def fit_maps(system, layer, bases, lam, null_mode=None, null_seed=0):
    rng = np.random.default_rng(null_seed)
    Rs = {}
    for a in system.generators:
        xs, ys = _pair_eps(system, bases, a)
        X = system.states(xs, layer)
        if null_mode == "identity":
            Y = system.states(xs, layer)
        else:
            Y = system.states(ys, layer)
            if null_mode == "shuffled":
                Y = Y[rng.permutation(len(Y))]
        Rs[a] = ridge_fit(X, Y, lam)
    return Rs


def pair_error(system, Rs, layer, bases):
    """Relative prediction error on genuine (x, g.x) pairs; identity map scores 1."""
    errs = []
    for a, R in Rs.items():
        xs, ys = _pair_eps(system, bases, a)
        X, Y = system.states(xs, layer), system.states(ys, layer)
        errs.append(((X @ R.T - Y) ** 2).sum() / max(((Y - X) ** 2).sum(), 1e-12))
    return float(np.mean(errs))


def group_law_metrics(system, Rs, layer, bases):
    """Activation-weighted defects on held-out states (Frobenius defects are
    secondary diagnostics; least squares leaves R underdetermined off the
    activation span)."""
    eps = [ep for b in bases for ep in system.orbit(b).values()]
    H = system.states(eps, layer)
    hn = (H**2).sum()
    Rlist = [Rs[a] for a in system.generators]
    inv = np.mean([((H @ (R.T @ R.T - np.eye(len(R)))) ** 2).sum() / hn for R in Rlist])
    R12, R23 = Rlist[0], Rlist[1]
    A, B = R12 @ R23 @ R12, R23 @ R12 @ R23
    # normalise by the size of the braid words so the defect is scale-free
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


def support_and_rank(system, Rs, layer, bases, max_eps=48):
    """Fraction of (R - I) row-space mass inside span{u_i - u_j}, the
    episode-local difference subspace (never the span of all name vectors),
    plus effective rank of (R - I)."""
    eps = [system.orbit(b)[system.all_perms[0]] for b in bases][:max_eps]
    U = system.u_vectors(eps, layer)  # (n, k, d)
    masses, ranks = [], []
    for R in Rs.values():
        _, S, Vt = np.linalg.svd(R - np.eye(len(R)))
        w = S**2 / max((S**2).sum(), 1e-12)
        p = w / w.sum()
        ranks.append(float(np.exp(-(p * np.log(p + 1e-30)).sum())))
        per_ep = []
        for i in range(len(eps)):
            diffs = U[i, 1:] - U[i, :1]  # (k-1, d)
            q, _ = np.linalg.qr(
                diffs.T
                + 1e-12 * np.random.default_rng(0).standard_normal(diffs.T.shape)
            )
            per_ep.append((w * ((Vt @ q) ** 2).sum(axis=1)).sum())
        masses.append(float(np.mean(per_ep)))
    # eff_rank is a logged diagnostic ONLY: R is fit unconstrained, so off the
    # role subspace it reflects ridge shrinkage and noise, not structure.
    # Nothing may depend on it.
    return dict(mass=float(np.mean(masses)), eff_rank=float(np.mean(ranks)))


def transport_agreement(system, Rs, layer, bases, per_base=2):
    """Condition 5: patch R_g h_l(x), run the remaining layers, decode, and
    compare to the natural run on g.x (never the unembedding directly)."""
    agrees = []
    for a, R in Rs.items():
        xs, ys = [], []
        for b in bases:
            orb = system.orbit(b)
            for g in system.all_perms[:per_base]:
                xs.append(orb[g])
                ys.append(orb[compose(a, g)])
        X = system.states(xs, layer)
        pred_patch = system.decode_from(X @ R.T, layer, ys)
        pred_nat = system.decode(ys)
        agrees.append(float(np.mean(pred_patch == pred_nat)))
    return dict(transport_agree=float(np.mean(agrees)))


def _fit_probe(X, y, n_classes, iters=300, lr=1.0, l2=1e-4):
    """Linear multinomial-logistic probe (full-batch GD on standardized features)."""
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Xn = (X - mu) / sd
    n, d = Xn.shape
    W = np.zeros((d, n_classes))
    b = np.zeros(n_classes)
    Y = np.eye(n_classes)[y]
    for _ in range(iters):
        Z = Xn @ W + b
        Z -= Z.max(1, keepdims=True)
        P = np.exp(Z)
        P /= P.sum(1, keepdims=True)
        G = (P - Y) / n
        W -= lr * (Xn.T @ G + l2 * W)
        b -= lr * G.sum(0)
    return dict(W=W, b=b, mu=mu, sd=sd)


def _probe_pred(probe, X):
    return (((X - probe["mu"]) / probe["sd"]) @ probe["W"] + probe["b"]).argmax(1)


def probe_metrics(system, Rs, layer, train_bases, eval_bases):
    """Condition 6. Content probe unchanged under R_g; role probe is an
    episode-local k-way decoder whose outputs are permuted by g. Probes are
    trained on the calibration split; held-out accuracy >= 0.9 required,
    else the condition is unavailable."""
    tr = [ep for b in train_bases for ep in system.orbit(b).values()]
    ev = [ep for b in eval_bases for ep in system.orbit(b).values()]
    Xtr, Xev = system.states(tr, layer), system.states(ev, layer)
    c_tr, c_ev = system.content_labels(tr), system.content_labels(ev)
    r_tr, r_ev = system.answers(tr), system.answers(ev)
    nc = int(max(c_tr.max(), c_ev.max())) + 1
    pc = _fit_probe(Xtr, c_tr, nc)
    pr = _fit_probe(Xtr, r_tr, system.k)
    c_acc = float(np.mean(_probe_pred(pc, Xev) == c_ev))
    r_acc = float(np.mean(_probe_pred(pr, Xev) == r_ev))
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


def first_decodable_layer(system, bases):
    """First layer at which the answer is linearly decodable (probe acc >= 0.9),
    estimated on the validation/calibration split only. Bounds the patch layer."""
    half = len(bases) // 2
    tr = [ep for b in bases[:half] for ep in system.orbit(b).values()]
    ev = [ep for b in bases[half:] for ep in system.orbit(b).values()]
    y_tr, y_ev = system.answers(tr), system.answers(ev)
    for layer in range(system.n_layers):
        p = _fit_probe(system.states(tr, layer), y_tr, system.k)
        if float(np.mean(_probe_pred(p, system.states(ev, layer)) == y_ev)) >= 0.9:
            return layer
    return system.n_layers - 1


def build_splits(system, phase):
    """phase 'cal' -> calibration halves only; 'test' -> untouched test halves.
    Probes always train on the calibration split (spec, condition 6)."""
    return dict(
        fit=system.bases("P", "fit", phase),
        transfer=system.bases("P", "transfer", phase),
        cross=system.bases("G", "fit", phase) if system.crosspath_available else None,
        probe_train=system.bases("P", "fit", "cal"),
    )


def all_metrics(system, layer, phase, lam, null_mode=None, null_seed=0):
    s = build_splits(system, phase)
    Rs = fit_maps(system, layer, s["fit"], lam, null_mode, null_seed)
    m = dict(content_transfer_err=pair_error(system, Rs, layer, s["transfer"]))
    m["crosspath_err"] = (
        pair_error(system, Rs, layer, s["cross"]) if s["cross"] else None
    )
    m.update(group_law_metrics(system, Rs, layer, s["transfer"]))
    sr = support_and_rank(system, Rs, layer, s["transfer"])
    support_key = (
        "support_mass_lex"
        if getattr(system, "support_test", "role") == "lexical"
        else "support_mass"
    )
    m[support_key] = sr["mass"]
    m["eff_rank"] = sr["eff_rank"]
    m.update(transport_agreement(system, Rs, layer, s["transfer"]))
    m.update(probe_metrics(system, Rs, layer, s["probe_train"], s["transfer"]))
    return m


def evaluate_conditions(m, thresholds):
    """Apply frozen thresholds; returns {condition: 'pass'|'fail'|'unavailable'}.
    C4 uses whichever support test the system computed (role / lexical)."""
    conds = {k: list(v) for k, v in CONDITIONS.items()}
    if "support_mass_lex" in m:
        conds["C4_nontriviality"] = ["nontriv", "noncommute", "support_mass_lex"]
    out = {}
    for cond, keys in conds.items():
        vals = [(k, m.get(k)) for k in keys]
        if any(v is None or k not in thresholds for k, v in vals):
            out[cond] = "unavailable"
            continue
        ok = all(
            (
                v < thresholds[k]["tau"]
                if METRICS[k][0] == "lt"
                else v > thresholds[k]["tau"]
            )
            for k, v in vals
        )
        out[cond] = "pass" if ok else "fail"
    return out


def verdict_and_score(conds, m):
    """H_role needs every available condition to pass with C1/C2/C5 available
    (any of those unavailable -> inconclusive). Continuous score = number of
    conditions passed, tie-broken by the activation-weighted group-law defect
    on the held-out split."""
    statuses = list(conds.values())
    if "fail" in statuses:
        verdict = "H_retrieval"
    elif all(conds[c] == "pass" for c in MANDATORY):
        verdict = "H_role"
    else:
        verdict = "inconclusive"
    n_pass = statuses.count("pass")
    score = n_pass + 0.999 * max(0.0, 1.0 - min(1.0, m.get("law_inv_defect", 1.0)))
    return verdict, float(score)


def run_frozen(system, cfg, run_id=""):
    """Single evaluation on the untouched test split with a frozen config."""
    layer, lam = cfg["layer"], cfg["lambda"]
    m = all_metrics(system, layer, "test", lam)
    conds = evaluate_conditions(m, cfg["thresholds"])
    verdict, score = verdict_and_score(conds, m)
    log(
        f"{run_id}: layer={layer} verdict={verdict} score={score:.3f} "
        + " ".join(f"{c.split('_')[0]}={s[:4]}" for c, s in conds.items())
    )
    for k in (
        "content_transfer_err",
        "law_inv_defect",
        "transport_agree",
        "support_mass",
        "support_mass_lex",
        "eff_rank",
    ):
        if k in m:
            log(
                f"{run_id}:   {k}={m[k] if m[k] is not None else 'n/a'}"
                + (" (diagnostic only)" if k == "eff_rank" else "")
            )
    return dict(
        metrics=m,
        conditions=conds,
        verdict=verdict,
        score=score,
        layer=layer,
        patch_layer=layer,
    )

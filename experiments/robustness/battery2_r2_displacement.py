"""Phase 8C supplementary robustness — R2-displacement.

Per-layer relative displacement norms ||h(g.x) - h(x)|| / ||h(x)|| across
ALL orbit pairs (all 5 non-identity group elements g applied to all 6
episodes of every base orbit), both paths (P, G), both vocabularies
(fit, transfer), at the FROZEN 8C reporting layer set as stored in
results/phase8c/committed_config.json .reporting_layer_set. NOTE ON COUNT:
the issuing instruction says "10 frozen layers"; the stored frozen set has
12 layers [0,8,16,24,32,40,48,56,61,64,72,79]; per instruction the stored
set is used and the count discrepancy is noted.

Then PCA of the displacement vectors across episodes, per cell per layer;
variance explained by the top-5 PCs.

Descriptive only: nothing is fitted, no thresholds, no selection.
Group convention: y = g.x means perm_y = compose(g, perm_x) (frozen
pair_rows semantics, phase8c_lib.py lines 96-104).

Scopes:
  all_bases      — all 300 base orbits per cell (literal "all orbit pairs")
  split_universe — cal+test bases from splits.json (strict-orbit-correct
                   universe; P cells 300, G/fit 290, G/transfer 294)
Norm statistics are reported for BOTH scopes; PCA on all_bases (declared).

Exact factored PCA (no d x N displacement matrix): with C the (N_pairs x
N_eps) signed incidence matrix (+1 at row(g.x), -1 at row(x)), D = C H, so
the nonzero eigenvalues of D D^T equal those of K^{1/2} S K^{1/2} with
K = C^T C and S = H H^T. Column sums of C are exactly zero (every episode
appears +1 and -1 five times each), so the displacement mean is exactly 0
and no centering is applied (verified numerically). The factored spectrum
is verified against an explicit small-subset SVD before any full-layer
computation (assert), and trace(M) is cross-checked against the sum of
squared displacement norms at every (cell, layer).

CPU/numpy float64 on the checksummed cached fp16 activations. Writes
results/phase8c/robustness/r2_displacement.json AS COMPUTED.
"""

import hashlib
import json
import time

import numpy as np

import activation_discriminator as lib
from activation_discriminator import PERMS, compose

t0 = time.time()

cfg_txt = (lib.OUT / "committed_config.json").read_text()
h = hashlib.sha256(cfg_txt.encode()).hexdigest()
assert h == (lib.OUT / "committed_config.sha256").read_text().strip()
cfg = json.loads(cfg_txt)
LAYER_SET = cfg["reporting_layer_set"]
print(
    f"config verified sha256={h[:16]}...; frozen reporting set "
    f"({len(LAYER_SET)} layers): {LAYER_SET}"
)

NONID = [p for p in PERMS if p != (0, 1, 2)]  # 5 non-identity elems
GLAB = {p: f"g{''.join(map(str, p))}" for p in NONID}  # e.g. (1,0,2)->g102

splits = json.load(open(lib.OUT / "splits.json"))["splits"]
cells = lib.load_cells()


def pair_index(cell, bases):
    """Ordered pairs (x, g.x) for all g in NONID, all 6 episodes x of each
    base. Returns (rows_x, rows_y, g_ids)."""
    rx, ry, gi = [], [], []
    for b in bases:
        for k, a in enumerate(NONID):
            for g in PERMS:
                rx.append(cell.row(b, g))
                ry.append(cell.row(b, compose(a, g)))
                gi.append(k)
    return np.array(rx), np.array(ry), np.array(gi)


def build_K(cell, bases):
    """K = C^T C for the all-orbit-pair incidence matrix, plus a zero
    column-sum check."""
    n_eps = len(cell.recs)
    rx, ry, _ = pair_index(cell, bases)
    C = np.zeros((len(rx), n_eps))
    C[np.arange(len(rx)), ry] += 1.0
    C[np.arange(len(rx)), rx] -= 1.0
    colsum = np.abs(C.sum(0)).max()
    assert colsum == 0.0, colsum  # exact: mean displacement = 0
    K = C.T @ C
    return K, (rx, ry)


def ksqrt(K):
    w, V = np.linalg.eigh(K)
    w = np.clip(w, 0.0, None)
    return (V * np.sqrt(w)) @ V.T


def stats(vals):
    v = np.asarray(vals)
    return dict(
        n=int(v.size),
        mean=float(v.mean()),
        std=float(v.std()),
        median=float(np.median(v)),
        p5=float(np.percentile(v, 5)),
        p95=float(np.percentile(v, 95)),
        min=float(v.min()),
        max=float(v.max()),
    )


# ---------- verification of the factored PCA on a small subset ----------
vcell = cells[("P", "fit")]
vbases = list(range(4))
vlayer = 61
vrx, vry, _ = pair_index(vcell, vbases)
rows = sorted(set(vrx) | set(vry))
Hs = vcell.states(np.array(rows), vlayer)
rmap = {r: i for i, r in enumerate(rows)}
D = Hs[[rmap[r] for r in vry]] - Hs[[rmap[r] for r in vrx]]
sv = np.linalg.svd(D, compute_uv=False)
eig_direct = np.sort(sv**2)[::-1]
Cv = np.zeros((len(vrx), len(rows)))
Cv[np.arange(len(vrx)), [rmap[r] for r in vry]] += 1.0
Cv[np.arange(len(vrx)), [rmap[r] for r in vrx]] -= 1.0
Kv = Cv.T @ Cv
Mv = ksqrt(Kv) @ (Hs @ Hs.T) @ ksqrt(Kv)
eig_fact = np.sort(np.clip(np.linalg.eigvalsh(Mv), 0, None))[::-1]
k = min(len(eig_direct), len(eig_fact))
dev = float(np.abs(eig_direct[:k] - eig_fact[:k]).max() / max(eig_direct[0], 1e-30))
assert dev < 1e-9, dev
verification = dict(
    subset="P/fit bases 0-3, layer 61, 120 explicit displacement vectors",
    max_rel_eig_dev_direct_vs_factored=dev,
    top3_eigs_direct=eig_direct[:3].tolist(),
    top3_eigs_factored=eig_fact[:3].tolist(),
)
print(f"factored-PCA verification: max rel eig dev = {dev:.2e}")

# ------------------------------ main loop ------------------------------
results = dict(
    config_sha256=h,
    frozen_reporting_layer_set=LAYER_SET,
    layer_count_note=(
        "Issuing instruction text says '10 frozen layers'; the frozen 8C "
        "reporting set stored in results/phase8c/committed_config.json "
        ".reporting_layer_set has 12 layers [0,8,16,24,32,40,48,56,61,64,"
        "72,79]. Per instruction, the stored frozen set (12 layers) is "
        "used and the count discrepancy is noted."
    ),
    pair_definition=(
        "ordered pairs (x, g.x), g over the 5 non-identity S3 elements, "
        "x over all 6 episodes of each base orbit; y = g.x means "
        "perm_y = compose(g, perm_x) (frozen pair_rows convention); "
        "norm ratio = ||h(g.x)-h(x)||_2 / ||h(x)||_2, float64 on cached "
        "fp16 activations"
    ),
    scopes=dict(
        all_bases="all 300 base orbits per cell (literal 'all orbit pairs')",
        split_universe=(
            "cal+test bases from results/phase8c/splits.json "
            "(strict-orbit-correct universe)"
        ),
    ),
    pca_note=(
        "PCA per cell per layer over all all_bases displacement vectors "
        "(N = 300*30 = 9000); mean displacement is exactly zero by the "
        "pairing symmetry (signed incidence columns sum to 0; asserted), "
        "so uncentered second-moment PCA equals covariance PCA. Factored "
        "computation K^{1/2} H H^T K^{1/2}; verified against explicit SVD "
        "(this file, .verification)."
    ),
    verification=verification,
    acts_checksums="results/verdict/answer_position/acts/checksums.json (verified)",
    cells={},
)

for (path, vocab), cell in cells.items():
    key = f"{path}/{vocab}"
    all_bases = list(range(cell.n_bases))
    su = splits[key]["cal"] + splits[key]["test"]
    K_all, _ = build_K(cell, all_bases)
    Ks_all = ksqrt(K_all)
    rx_a2, ry_a2, gi_a = pair_index(cell, all_bases)
    su_set = set(su)
    su_mask = np.array([cell.recs[r]["base_id"] in su_set for r in rx_a2])
    cellout = dict(
        n_bases_all=len(all_bases), n_bases_split_universe=len(su), per_layer={}
    )
    for layer in LAYER_SET:
        tl = time.time()
        H = cell.states(np.arange(len(cell.recs)), layer)
        S = H @ H.T
        d2 = S[ry_a2, ry_a2] + S[rx_a2, rx_a2] - 2.0 * S[ry_a2, rx_a2]
        d2 = np.clip(d2, 0.0, None)
        ratio = np.sqrt(d2) / np.sqrt(S[rx_a2, rx_a2])
        # PCA (all_bases scope)
        M = Ks_all @ S @ Ks_all
        eig = np.sort(np.clip(np.linalg.eigvalsh(M), 0, None))[::-1]
        tot = float(eig.sum())
        tr_check = abs(tot - float(d2.sum())) / max(float(d2.sum()), 1e-30)
        assert tr_check < 1e-9, tr_check
        top5 = eig[:5]
        lay = dict(
            all_bases=dict(
                pooled=stats(ratio),
                per_g={
                    GLAB[NONID[k]]: stats(ratio[gi_a == k]) for k in range(len(NONID))
                },
            ),
            split_universe=dict(pooled=stats(ratio[su_mask])),
            pca_all_bases=dict(
                n_vectors=int(len(rx_a2)),
                total_var=tot,
                top5_eigenvalues=top5.tolist(),
                var_explained_top5=(top5 / tot).tolist(),
                cum_var_explained_top5=float(top5.sum() / tot),
                trace_crosscheck_rel_dev=tr_check,
            ),
        )
        cellout["per_layer"][str(layer)] = lay
        print(
            f"{key} L{layer}: ratio mean={lay['all_bases']['pooled']['mean']:.4f} "
            f"med={lay['all_bases']['pooled']['median']:.4f} "
            f"PC1-5 ve={['%.4f' % v for v in lay['pca_all_bases']['var_explained_top5']]} "
            f"cum={lay['pca_all_bases']['cum_var_explained_top5']:.4f} "
            f"({time.time()-tl:.0f}s)",
            flush=True,
        )
        del H, S, M
    results["cells"][key] = cellout

results["wall_s"] = round(time.time() - t0, 1)
p = lib.OUT / "robustness" / "r2_displacement.json"
json.dump(results, open(p, "w"), indent=2)
print(f"wrote {p} ({time.time()-t0:.0f}s)")

# ------------------------------- table ---------------------------------
L = LAYER_SET
lines = [
    "# Phase 8C robustness R2 — displacement norms + displacement PCA "
    "(interpretation-free)",
    "",
    "Source: results/phase8c/robustness/r2_displacement.json "
    "(this table is generated from it by phase8c_rob2_r2_displacement.py).",
    "Pairs: (x, g.x), all 5 non-identity g x all 6 episodes per base orbit, "
    "ALL bases per cell (300); ratio = ||h(g.x)-h(x)||/||h(x)||, float64 on "
    "cached fp16 acts (results/phase8a_final/acts, checksums verified).",
    "Layers: frozen 8C reporting set as stored in committed_config.json — "
    "12 layers (instruction text said '10'; count noted, stored set used).",
    "PCA: per cell per layer over all 9000 displacement vectors; mean is "
    "exactly 0 by pairing symmetry; veK = fraction of total displacement "
    "variance in PC K; cum5 = top-5 cumulative.",
    "",
]
for key in ("P/fit", "P/transfer", "G/fit", "G/transfer"):
    c = results["cells"][key]["per_layer"]
    lines.append(f"## {key}")
    lines.append("")
    lines.append(
        "| layer | mean | std | median | p5 | p95 | ve1 | ve2 | "
        "ve3 | ve4 | ve5 | cum5 |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for layer in L:
        a = c[str(layer)]["all_bases"]["pooled"]
        v = c[str(layer)]["pca_all_bases"]["var_explained_top5"]
        cum = c[str(layer)]["pca_all_bases"]["cum_var_explained_top5"]
        lines.append(
            f"| {layer} | {a['mean']:.4f} | {a['std']:.4f} | "
            f"{a['median']:.4f} | {a['p5']:.4f} | {a['p95']:.4f} | "
            + " | ".join(f"{x:.4f}" for x in v)
            + f" | {cum:.4f} |"
        )
    lines.append("")
pt = lib.OUT / "robustness" / "r2_displacement_table.md"
pt.write_text("\n".join(lines) + "\n")
print(f"wrote {pt}")

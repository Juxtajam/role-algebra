"""Phase 8C — Step 2 ( instruction B), CALIBRATION SPLIT ONLY.

A. Answer-decodability curve across all 80 layers, per cell (frozen
   first_decodable_layer semantics: cal bases halved, k-way answer probe,
   held-out acc >= 0.9).
B. Calibration content-transfer error curve: fit R on P/fit cal, eval on
   P/transfer cal, all 80 layers x frozen 6-value lambda grid (frozen
   pair_error). Crosspath (G/fit cal) curve logged as diagnostic.
C. Matched-regime shuffled-null transfer-err curve (5 seeds) at each layer
   for the layer-selection rule (guards against degenerate-layer argmin).

Outputs results/phase8c/calibration_curves.json. No test base is read.
"""

import json
import time

import numpy as np

import activation_discriminator as lib

t0 = time.time()
cells = lib.load_cells()
splits = json.load(open(lib.OUT / "splits.json"))["splits"]
cal = {k: v["cal"] for k, v in splits.items()}

out = {"lam_grid": lib.LAM_GRID}

# ---------------- A: decodability curves ----------------
out["decodability"] = {}
for key, cell in cells.items():
    name = f"{key[0]}/{key[1]}"
    curve, first, slot_curve = lib.decodability_curve(cell, cal[name])
    out["decodability"][name] = dict(
        curve=curve, first_decodable=first, slot_probe_diag=slot_curve
    )
    print(
        f"decodability {name}: first_decodable={first} "
        f"acc@first={curve[first]:.3f} acc@79={curve[-1]:.3f} "
        f"({time.time()-t0:.0f}s)"
    )

# ---------------- B + C: transfer-err curves ----------------
cf, ct, cg = cells[("P", "fit")], cells[("P", "transfer")], cells[("G", "fit")]
bases_fit = cal["P/fit"]
bases_tr = cal["P/transfer"]
bases_g = cal["G/fit"]

pairs_fit = {a: lib.pair_rows(cf, bases_fit, a) for a in lib.GENERATORS}
pairs_tr = {a: lib.pair_rows(ct, bases_tr, a) for a in lib.GENERATORS}
pairs_g = {a: lib.pair_rows(cg, bases_g, a) for a in lib.GENERATORS}

N_NULL_SEEDS = 5
real_err = np.zeros((lib.N_LAYERS, len(lib.LAM_GRID)))
cross_err = np.zeros((lib.N_LAYERS, len(lib.LAM_GRID)))
null_err = np.zeros((lib.N_LAYERS, len(lib.LAM_GRID), N_NULL_SEEDS))

for layer in range(lib.N_LAYERS):
    tl = time.time()
    per_gen = {}
    for a in lib.GENERATORS:
        rx, ry = pairs_fit[a]
        X = cf.states(rx, layer)
        Y = cf.states(ry, layer)
        ex, ey = pairs_tr[a]
        Vx, Vy = ct.states(ex, layer), ct.states(ey, layer)
        gx, gy = pairs_g[a]
        Gx, Gy = cg.states(gx, layer), cg.states(gy, layer)
        # shared eigendecomposition of XX^T across lambdas
        XXt = X @ X.T
        s, U = np.linalg.eigh(XXt)
        trG = float((X * X).sum())
        TV = Vx @ X.T
        TG = Gx @ X.T
        den_t = max(((Vy - Vx) ** 2).sum(), 1e-12)
        den_g = max(((Gy - Gx) ** 2).sum(), 1e-12)
        per_gen[a] = dict(
            X=X,
            Y=Y,
            s=s,
            U=U,
            trG=trG,
            TV=TV,
            TG=TG,
            Vy=Vy,
            Gy=Gy,
            den_t=den_t,
            den_g=den_g,
        )
    for li, lam in enumerate(lib.LAM_GRID):
        et, eg = [], []
        for a in lib.GENERATORS:
            p = per_gen[a]
            lam_eff = lam * p["trG"] / lib.D_MODEL + 1e-12
            Minv = (p["U"] / (p["s"] + lam_eff)) @ p["U"].T
            Pt = (p["TV"] @ Minv) @ p["Y"]
            et.append(((Pt - p["Vy"]) ** 2).sum() / p["den_t"])
            Pg = (p["TG"] @ Minv) @ p["Y"]
            eg.append(((Pg - p["Gy"]) ** 2).sum() / p["den_g"])
        real_err[layer, li] = np.mean(et)
        cross_err[layer, li] = np.mean(eg)
    # shuffled nulls (matched regime: same pairs, same lambda grid, same layer)
    for ns in range(N_NULL_SEEDS):
        for li, lam in enumerate(lib.LAM_GRID):
            et = []
            for ai, a in enumerate(lib.GENERATORS):
                p = per_gen[a]
                # frozen null convention: one rng per fit_maps call, consumed
                # per generator in order
                rng = np.random.default_rng(9000 + ns)
                permA = rng.permutation(len(p["Y"]))
                permB = rng.permutation(len(p["Y"]))
                perm = permA if ai == 0 else permB
                Yn = p["Y"][perm]
                lam_eff = lam * p["trG"] / lib.D_MODEL + 1e-12
                Minv = (p["U"] / (p["s"] + lam_eff)) @ p["U"].T
                Pt = (p["TV"] @ Minv) @ Yn
                et.append(((Pt - p["Vy"]) ** 2).sum() / p["den_t"])
            null_err[layer, li, ns] = np.mean(et)
    if layer % 8 == 0 or layer == lib.N_LAYERS - 1:
        li_best = int(np.argmin(real_err[layer]))
        print(
            f"layer {layer:2d}: min real err={real_err[layer, li_best]:.4f} "
            f"@lam={lib.LAM_GRID[li_best]:g} cross={cross_err[layer, li_best]:.4f} "
            f"null_med={np.median(null_err[layer, li_best]):.4f} "
            f"({time.time()-tl:.1f}s)"
        )

out["real_transfer_err"] = real_err.tolist()
out["cross_err_diag"] = cross_err.tolist()
out["null_transfer_err"] = null_err.tolist()
out["null_seeds"] = list(range(9000, 9000 + N_NULL_SEEDS))
out["matched_regime"] = dict(
    n_pairs_per_generator=len(pairs_fit[lib.GENERATORS[0]][0]),
    d_model=lib.D_MODEL,
    lam_grid=lib.LAM_GRID,
    fit_cell="P/fit cal",
    eval_cell="P/transfer cal",
)
json.dump(out, open(lib.OUT / "calibration_curves.json", "w"), indent=2)
print(f"done in {time.time()-t0:.0f}s -> {lib.OUT/'calibration_curves.json'}")

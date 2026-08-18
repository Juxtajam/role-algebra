"""ROBUSTNESS PASS R2 — R_lex deflationary baseline + identity operator I,
evaluated under conditions 1, 2, 3 (+ support-mass and nontriviality context)
on the SAME test split, same frozen layers, same metrics, against the same
frozen thresholds/nulls as the genuine fitted-R run.

R_lex construction (phase8c_res_rlex.py, unit-checked in
phase8c_res_rlex_check.py): per generator a,
    R_lex = I + (D_after - D_before) @ pinv(D_before)
over the SAME fit bases the fitted R used (P/fit TEST bases), where D_before
stacks u_{n_i} - u_{n_j} (all ordered slot pairs, canonical name triple) and
D_after their images under a. u_n = lm_head readout row of name n
(results/phase8c/name_vectors.npz key 'readout' — byte-identical to the
independently fetched results/phase8c_resolution/name_token_rows.npz).
R_lex has NO parameters fit on activations and is layer-independent; the
metrics evaluate it against activations at each frozen layer.

Identity operator I: the exact identity map (pre-recorded adjudication
extension: fitted R itself only passed C1 at identity-null level, so the
triple R_lex vs I vs fitted-R is the robustness core).

Frozen metric code reused verbatim: lib.pair_error, lib.group_law_metrics,
lib.support_mass_lexical. Both operators expose the DualRidge apply interface
(and a dummy .X row so group_law_metrics' HXt cache builds unchanged; the
cache is ignored by these operators' apply_RT).

Thresholds: the committed frozen taus (results/phase8c/committed_config.json)
— identical comparison as the fitted R. Bootstrap CIs: same seed/proto as
phase8c_test_eval.py (seed 20260807, 10k resamples over bases).

Output (written per layer as computed):
  results/phase8c/robustness/r2_rlex_identity_results.json
  results/phase8c/robustness/rlex_results.json  (same content, name per
  instruction)
"""

import json
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
import activation_discriminator as lib  # noqa: E402
from activation_discriminator import GENERATORS, PERMS  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[2]
ROB = ROOT / "results/robustness/battery"
ROB.mkdir(parents=True, exist_ok=True)

t0 = time.time()


class LexOperator:
    """R_lex = I + C @ Pinv on rows; identity off span(D_before).
    Verified against dense construction in phase8c_res_rlex_check.py."""

    def __init__(self, D_before, D_after, d):
        self.d = d
        self.P = np.linalg.pinv(D_before, rcond=1e-10)  # (d, m)
        self.C = D_after - D_before  # (m, d)
        self.X = np.zeros((1, d))  # dummy for HXt cache
        recon = D_before + (D_before @ self.P) @ self.C
        self.residual = float(
            ((recon - D_after) ** 2).sum() / max((D_after**2).sum(), 1e-12)
        )

    def apply_RT(self, V, VXt=None):
        return V + (V @ self.P) @ self.C

    def apply_R_cols(self, Q):
        return Q + self.C.T @ (self.P.T @ Q)

    def frob2_R_minus_I(self):
        G1 = self.C @ self.C.T
        G2 = self.P.T @ self.P
        return float((G1 * G2).sum())


class IdentityOperator:
    def __init__(self, d):
        self.d = d
        self.X = np.zeros((1, d))

    def apply_RT(self, V, VXt=None):
        return V

    def apply_R_cols(self, Q):
        return Q

    def frob2_R_minus_I(self):
        return 0.0


def boot_ratio(nums, dens, seed=lib.BOOT_SEED, n_boot=lib.N_BOOT):
    nums, dens = np.asarray(nums), np.asarray(dens)
    nb = nums.shape[1]
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, nb, size=(n_boot, nb))
    stats = np.mean(
        nums[:, idx].sum(axis=2) / np.maximum(dens[:, idx].sum(axis=2), 1e-12), axis=0
    )
    return [float(np.percentile(stats, q)) for q in (2.5, 97.5)]


cfg = json.load(open(ROOT / "results/verdict/discriminator/committed_config.json"))
tm = json.load(open(ROOT / "results/verdict/discriminator/test_metrics.json"))
splits = json.load(open(ROOT / "results/verdict/discriminator/splits.json"))["splits"]
LAYERS = cfg["reporting_layer_set"]
L_STAR = cfg["layer"]
TAUS = cfg["thresholds"]

cells = lib.load_cells()
cf, ct = cells[("P", "fit")], cells[("P", "transfer")]
cg, cgt = cells[("G", "fit")], cells[("G", "transfer")]
test = {k: v["test"] for k, v in splits.items()}

nv = np.load(ROOT / "results/verdict/discriminator/name_vectors.npz", allow_pickle=True)
name_vec = {n: v for n, v in zip(nv["names"], nv["readout"])}

# ---- build R_lex per generator on the fitted-R fit bases (P/fit TEST)
ops_lex = {}
for a in GENERATORS:
    Db, Da = [], []
    for b in test["P/fit"]:
        names = cf.rec(b, PERMS[0])["base"]["names"]
        U = [np.asarray(name_vec[n], dtype=np.float64) for n in names]
        for i in range(3):
            for j in range(i + 1, 3):
                Db.append(U[i] - U[j])
                Da.append(U[a[i]] - U[a[j]])
    ops_lex[a] = LexOperator(np.stack(Db), np.stack(Da), lib.D_MODEL)
ops_id = {a: IdentityOperator(lib.D_MODEL) for a in GENERATORS}

out = dict(
    construction="R_lex = I + (D_after - D_before) @ pinv(D_before); "
    "fit bases = P/fit TEST (the fitted-R fit bases); u = "
    "lm_head readout rows (name_vectors.npz['readout'])",
    construction_residual={
        lib.GEN_NAMES[GENERATORS.index(a)]: ops_lex[a].residual for a in GENERATORS
    },
    frob2_R_lex_minus_I={
        lib.GEN_NAMES[GENERATORS.index(a)]: ops_lex[a].frob2_R_minus_I()
        for a in GENERATORS
    },
    thresholds_frozen={
        k: TAUS[k]
        for k in (
            "content_transfer_err",
            "crosspath_err",
            "law_inv_defect",
            "law_braid_defect",
            "nontriv",
            "noncommute",
            "support_mass_lex",
        )
    },
    fitted_R_source="results/verdict/discriminator/test_metrics.json (stored run)",
    layers=list(LAYERS),
    verdict_layer=L_STAR,
    per_layer={},
)


def battery(ops, layer, with_ci):
    m = {}
    err, pb = lib.pair_error(ops, ct, test["P/transfer"], layer, per_base=True)
    m["content_transfer_err"] = err
    if with_ci:
        m["content_transfer_ci"] = boot_ratio(
            [p["num"] for p in pb], [p["den"] for p in pb]
        )
    err, pb = lib.pair_error(ops, cg, test["G/fit"], layer, per_base=True)
    m["crosspath_err"] = err
    if with_ci:
        m["crosspath_ci"] = boot_ratio([p["num"] for p in pb], [p["den"] for p in pb])
    m["crosspath_transfer_err_desc"] = lib.pair_error(
        ops, cgt, test["G/transfer"], layer
    )
    laws, lpb = lib.group_law_metrics(ops, ct, test["P/transfer"], layer, per_base=True)
    m.update(laws)
    if with_ci:
        m["law_inv_ci"] = boot_ratio(lpb["inv"], [lpb["hn"]] * 2)
        m["law_braid_ci"] = boot_ratio([lpb["braid_num"]], [lpb["braid_den"]])
        m["nontriv_ci"] = boot_ratio(lpb["nontriv"], [lpb["hn"]] * 2)
    m["support_mass_lex"] = lib.support_mass_lexical(
        ops, ct, test["P/transfer"], name_vec
    )
    return m


for layer in LAYERS:
    tl = time.time()
    ci = layer == L_STAR
    m_lex = battery(ops_lex, layer, ci)
    m_id = battery(ops_id, layer, ci)
    fitted = {
        k: tm["per_layer"][str(layer)].get(k)
        for k in (
            "content_transfer_err",
            "content_transfer_ci",
            "crosspath_err",
            "crosspath_ci",
            "crosspath_transfer_err_desc",
            "law_inv_defect",
            "law_braid_defect",
            "nontriv",
            "noncommute",
            "support_mass_lex",
        )
    }
    out["per_layer"][str(layer)] = dict(
        R_lex=m_lex, identity=m_id, fitted_R_stored=fitted
    )
    json.dump(out, open(ROB / "r2_rlex_identity_results.json", "w"), indent=2)
    json.dump(out, open(ROB / "rlex_results.json", "w"), indent=2)
    print(
        f"layer {layer}: C1 lex={m_lex['content_transfer_err']:.6f} "
        f"id={m_id['content_transfer_err']:.6f} "
        f"fit={fitted['content_transfer_err']:.6f} | "
        f"C2 lex={m_lex['crosspath_err']:.6f} "
        f"id={m_id['crosspath_err']:.6f} "
        f"fit={fitted['crosspath_err']:.6f} | "
        f"inv lex={m_lex['law_inv_defect']:.3e} "
        f"nontriv lex={m_lex['nontriv']:.3e} ({time.time()-tl:.0f}s)",
        flush=True,
    )

# frozen-threshold comparison at the verdict layer (raw booleans)
vl = out["per_layer"][str(L_STAR)]
comp = {}
for name in ("R_lex", "identity", "fitted_R_stored"):
    mm = vl[name]
    comp[name] = {
        "content_transfer_err_lt_tau": bool(
            mm["content_transfer_err"] < TAUS["content_transfer_err"]["tau"]
        ),
        "crosspath_err_lt_tau": bool(
            mm["crosspath_err"] < TAUS["crosspath_err"]["tau"]
        ),
        "law_inv_defect_lt_tau": bool(
            mm["law_inv_defect"] < TAUS["law_inv_defect"]["tau"]
        ),
        "law_braid_defect_lt_tau": bool(
            mm["law_braid_defect"] < TAUS["law_braid_defect"]["tau"]
        ),
        "nontriv_gt_tau": bool(mm["nontriv"] > TAUS["nontriv"]["tau"]),
        "noncommute_gt_tau": bool(mm["noncommute"] > TAUS["noncommute"]["tau"]),
        "support_mass_lex_lt_tau": bool(
            mm["support_mass_lex"] < TAUS["support_mass_lex"]["tau"]
        ),
    }
out["verdict_layer_threshold_comparison"] = comp
json.dump(out, open(ROB / "r2_rlex_identity_results.json", "w"), indent=2)
json.dump(out, open(ROB / "rlex_results.json", "w"), indent=2)
print("threshold comparison @L61:", json.dumps(comp, indent=1))
print(
    f"wrote rlex_results.json + r2_rlex_identity_results.json "
    f"({time.time()-t0:.0f}s total)"
)

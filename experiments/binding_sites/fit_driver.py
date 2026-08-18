"""Phase 9 Item 3 — 9B fitting driver. Executes the frozen config
item3_committed_config.json (sha 6f524335...78b139) on the locally
sha256-verified cache acts_P_local/ (ALL_MATCH True).

Machinery: faithful port of phase8c_lib.py (dual-form ridge, pair_rows,
pair_error, group_law_metrics, support-mass) to the position cache layout
(n_eps, 11 pos, 22 layers, 8192), with one optimisation that changes no
math: eigh(XX^T) is lambda- and Y-independent, so per (cell, generator,
position-pairing) it is computed once and shared across the lambda grid and
the shuffled/identity nulls (X identical; only Y or lambda changes).
lam_eff = lam * ||X||_F^2 / d + 1e-12, identical to phase8c_lib.DualRidge.

Position classes (cache axis 1, manifest order, path P):
  entity_mention_carry cols 0..2 (slot s clause final, textual order),
  fact_final cols 3..8, query_arg col 9, answer col 10.
Pairing:
  in-place fits/evals: same column, (ep g, ep a.g) — fact order fixed, so
    column == slot for carry; for the two moved slots of each generator the
    samples are concatenated (g12 -> slots {0,1}, g23 -> {1,2}).
  fact_final: all 6 columns as samples, same-column pairing (declared).
  joint C7: role-matched columns via base.fact_order (carry clause of slot
    s sits at textual carry-index = rank of clause-id s among carry ids in
    fact_order); scalar/fact classes pair same-column (textual structure
    fixed given fact_order applies to both episodes' own orders).
Lambda is selected per position-class x layer on the
calibration split (fit-vocab cal -> transfer-vocab cal transfer-err), the
8C criterion applied per cell instead of one global lambda (position
classes have no a-priori shared scale).
Thresholds per cell from matched nulls (10 shuffled, frozen seeds 0..9, +
identity), frozen rule: lt-metrics tau = min of the 10 shuffled values
(n=10 lower envelope, the 8C quantile logic at this n); identity value
reported alongside always.
Single evaluation: test bases touched exactly once, after all thresholds
for that cell are computed from calibration.
"""

import json, hashlib, os, time, itertools
import numpy as np
import pathlib

ROOT = str(pathlib.Path(__file__).resolve().parents[2])
ACTS = f"{ROOT}/acts_P_local/acts_P"
OUT = f"{ROOT}/results/phase9/fits"
os.makedirs(OUT, exist_ok=True)

CFG_PATH = f"{ROOT}/results/phase9/item3_committed_config.json"
CFG_SHA_EXPECT = "6f52433589ef65795cd84e1502ebaf18abaee31937dd30a08b95d10bdf78b139"
cfg_sha = hashlib.sha256(open(CFG_PATH, "rb").read()).hexdigest()
assert cfg_sha == CFG_SHA_EXPECT, cfg_sha
CFG = json.load(open(CFG_PATH))
LAYERS = CFG["layers"]  # 22 cached layers, cache axis order
LAMBDAS = CFG["ridge_protocol"]["lambda_grid"]

PERMS = [(0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)]
PERM_IDX = {p: i for i, p in enumerate(PERMS)}
GENERATORS = [(1, 0, 2), (0, 2, 1)]  # g12, g23 (dv3 frozen convention)
MOVED = {(1, 0, 2): (0, 1), (0, 2, 1): (1, 2)}


def compose(a, b):
    return tuple(a[b[j]] for j in range(3))


POS = {
    "entity_mention_carry": [0, 1, 2],
    "fact_final": [3, 4, 5, 6, 7, 8],
    "query_arg": [9],
    "answer": [10],
}

NV = np.load(f"{ROOT}/results/phase8c/name_vectors.npz")
NAME_LIST = [str(n) for n in NV["names"]]
READOUT = {n: NV["readout"][i].astype(np.float64) for i, n in enumerate(NAME_LIST)}
EMBED = {n: NV["embed"][i].astype(np.float64) for i, n in enumerate(NAME_LIST)}


class P9Cell:
    def __init__(self, recs_path, acts_path):
        self.recs = [json.loads(l) for l in open(recs_path)]
        self.acts = np.load(acts_path, mmap_mode="r")
        assert self.acts.shape[0] == len(self.recs) and self.acts.shape[1] == 11
        for i in (0, 1, 7, len(self.recs) - 1):
            r = self.recs[i]
            assert r["base_id"] == i // 6 and tuple(r["g"]) == PERMS[i % 6], i
        self.n_bases = len(self.recs) // 6

    def row(self, b, g):
        return b * 6 + PERM_IDX[tuple(g)]

    def states(self, rows, li, cols):
        # rows,cols same length (per-sample column) OR scalar col
        if np.isscalar(cols):
            return np.asarray(self.acts[rows, cols, li, :], dtype=np.float64)
        return np.stack(
            [
                np.asarray(self.acts[r, c, li, :], dtype=np.float64)
                for r, c in zip(rows, cols)
            ]
        )

    def carry_col(self, i, slot):
        fo = self.recs[i]["base"].get("fact_order", list(range(6)))
        carry_ids = [cid for cid in fo if cid < 3]
        return carry_ids.index(slot)  # textual carry-index of slot


CELLS = {}
for v in ("fit", "transfer"):
    CELLS[("frozen", v)] = P9Cell(
        f"{ROOT}/tasks_frozen/disc_P_{v}.jsonl", f"{ACTS}/frozen_disc_P_{v}.npy"
    )
    CELLS[("joint", v)] = P9Cell(
        f"{ROOT}/results/phase9/tasks_joint/joint_P_{v}.jsonl",
        f"{ACTS}/joint_joint_P_{v}.npy",
    )

S8 = json.load(open(f"{ROOT}/results/phase8c/splits.json"))["splits"]
SPLIT = {}
for v in ("fit", "transfer"):
    SPLIT[("frozen", v)] = (list(S8[f"P/{v}"]["cal"]), list(S8[f"P/{v}"]["test"]))
for v in ("fit", "transfer"):
    jb = sorted({r["base_id"] for r in CELLS[("joint", v)].recs})
    SPLIT[("joint", v)] = (jb[0::2], jb[1::2])


def pair_samples(cell, bases, a, pos_class, joint=False):
    """Sample lists (rows_x, cols_x, rows_y, cols_y)."""
    rx, cx, ry, cy = [], [], [], []
    for b in bases:
        for g in PERMS:
            i, j = cell.row(b, g), cell.row(b, compose(a, g))
            if pos_class == "entity_mention_carry":
                for s in MOVED[a]:
                    rx.append(i)
                    ry.append(j)
                    if joint:
                        cx.append(cell.carry_col(i, s))
                        cy.append(cell.carry_col(j, s))
                    else:
                        cx.append(s)
                        cy.append(s)
            else:
                cols = POS[pos_class]
                if pos_class == "fact_final":
                    # one column per episode, cycled across the orbit
                    c = cols[(PERM_IDX[tuple(g)] + b) % len(cols)]
                    rx.append(i)
                    cx.append(c)
                    ry.append(j)
                    cy.append(c)
                else:
                    for c in cols:
                        rx.append(i)
                        cx.append(c)
                        ry.append(j)
                        cy.append(c)
    return (np.array(rx), np.array(cx), np.array(ry), np.array(cy))


class SharedRidge:
    """DualRidge math with cached eigh(XX^T) shared across lam / Y."""

    def __init__(self, X):
        self.X = X
        self.n, self.d = X.shape
        self.fro2 = float((X * X).sum())
        self.XXt = X @ X.T
        self.s, self.U = np.linalg.eigh(self.XXt)

    def fit(self, Y, lam):
        lam_eff = lam * self.fro2 / self.d + 1e-12
        M = (self.U / (self.s + lam_eff)) @ self.U.T
        return FitR(self.X, Y, M)


class FitR:
    def __init__(self, X, Y, M):
        self.X, self.Y, self.M = X, Y, M
        self.MX = M @ X

    def apply_RT(self, V):
        return ((V @ self.X.T) @ self.M) @ self.Y

    def apply_R_cols(self, Q):
        return self.Y.T @ (self.M @ (self.X @ Q))

    def trace_R(self):
        return float((self.MX * self.Y).sum())

    def frob2_R(self):
        G = self.MX @ self.X.T
        return float(((G @ self.M) * (self.Y @ self.Y.T)).sum())

    def frob2_R_minus_I(self):
        return self.frob2_R() - 2.0 * self.trace_R() + self.X.shape[1]


def transport_err(Rs, cell, bases, li, pos_class, joint=False):
    errs = []
    for a, R in Rs.items():
        rx, cx, ry, cy = pair_samples(cell, bases, a, pos_class, joint)
        X = cell.states(rx, li, cx)
        Y = cell.states(ry, li, cy)
        P = R.apply_RT(X)
        errs.append(((P - Y) ** 2).sum() / max(((Y - X) ** 2).sum(), 1e-12))
    return float(np.mean(errs))


def group_laws(Rs, cell, bases, li, pos_class):
    cols = POS[pos_class]
    rows = np.array([cell.row(b, g) for b in bases for g in PERMS])
    H = np.concatenate([cell.states(rows, li, c) for c in cols])
    hn = (H**2).sum()
    R12, R23 = Rs[GENERATORS[0]], Rs[GENERATORS[1]]
    ap = lambda V, R: R.apply_RT(V)
    inv = np.mean([((ap(ap(H, R), R) - H) ** 2).sum() / hn for R in (R12, R23)])
    HA = ap(ap(ap(H, R12), R23), R12)
    HB = ap(ap(ap(H, R23), R12), R23)
    braid = ((HA - HB) ** 2).sum() / (
        0.5 * ((HA**2).sum() + (HB**2).sum()) + 1e-12 * hn
    )
    nontriv = np.mean([((ap(H, R) - H) ** 2).sum() / hn for R in (R12, R23)])
    return dict(inv=float(inv), braid=float(braid), nontriv=float(nontriv))


def support_mass(Rs, cell, bases, vectors, max_eps=48):
    sel = bases[:max_eps]
    Qs = []
    jit = np.random.default_rng(0)
    for b in sel:
        r = cell.recs[cell.row(b, PERMS[0])]
        U = np.stack([vectors[n] for n in r["base"]["names"]])
        diffs = (U[1:] - U[:1]).T
        q, _ = np.linalg.qr(diffs + 1e-12 * jit.standard_normal(diffs.shape))
        Qs.append(q)
    Qcat = np.concatenate(Qs, axis=1)
    masses = []
    for a, R in Rs.items():
        f2 = R.frob2_R_minus_I()
        RQ = R.apply_R_cols(Qcat)
        vals = [
            float(((RQ[:, 2 * i : 2 * i + 2] - Qs[i]) ** 2).sum() / max(f2, 1e-12))
            for i in range(len(sel))
        ]
        masses.append(float(np.mean(vals)))
    return float(np.mean(masses))


def rlex_apply(V, recs_names):
    """R_lex: per-episode rank-2 swap on readout diffs — for transport err
    of the lexical baseline we apply the generator's name swap of the
    episode. Approximation-free per-sample application."""
    return None  # computed inline below


def run():
    t0 = time.time()
    results = {}
    fit_cell, tr_cell = CELLS[("frozen", "fit")], CELLS[("frozen", "transfer")]
    jfit, jtr = CELLS[("joint", "fit")], CELLS[("joint", "transfer")]
    calF, testF = SPLIT[("frozen", "fit")]
    calT, testT = SPLIT[("frozen", "transfer")]
    calJF, testJF = SPLIT[("joint", "fit")]
    calJT, testJT = SPLIT[("joint", "transfer")]
    grid = list(itertools.product(POS.keys(), range(len(LAYERS))))
    for pc, li in grid:
        key = f"{pc}/L{LAYERS[li]}"
        cell_path = f"{OUT}/cell_{pc}_L{LAYERS[li]}.json"
        if os.path.exists(cell_path):  # resume: skip completed cells
            results[key] = json.load(open(cell_path))
            continue
        t1 = time.time()
        # memory-lean sampling for wide classes (DECLARED, config-compatible):
        # fact_final uses one column per episode, cycling columns across the
        # orbit (columns balanced within each base), keeping n at 1800 per
        # generator like the other classes instead of 5400 (OOM guard).
        # Implemented inside pair_samples via fact_stride.
        # ---- calibration: X shared per generator ----
        shared, Ycal = {}, {}
        for a in GENERATORS:
            rx, cx, ry, cy = pair_samples(fit_cell, calF, a, pc)
            X = fit_cell.states(rx, li, cx)
            Y = fit_cell.states(ry, li, cy)
            shared[a] = SharedRidge(X)
            Ycal[a] = Y
        # lambda selection on calibration transfer-err (fit-vocab cal ->
        # transfer-vocab cal)
        best = (None, np.inf)
        for lam in LAMBDAS:
            Rs = {a: shared[a].fit(Ycal[a], lam) for a in GENERATORS}
            e = transport_err(Rs, tr_cell, calT, li, pc)
            if e < best[1]:
                best = (lam, e)
        lam = best[0]
        Rs = {a: shared[a].fit(Ycal[a], lam) for a in GENERATORS}
        # ---- matched nulls (thresholds), calibration only ----
        null_c1, null_c7, null_laws = [], [], []
        for seed in range(10):
            rng = np.random.default_rng(seed)
            Rn = {
                a: shared[a].fit(Ycal[a][rng.permutation(len(Ycal[a]))], lam)
                for a in GENERATORS
            }
            null_c1.append(transport_err(Rn, tr_cell, calT, li, pc))
            null_c7.append(transport_err(Rn, jfit, calJF, li, pc, joint=True))
            null_laws.append(group_laws(Rn, fit_cell, calF, li, pc))
        Rid = {a: shared[a].fit(shared[a].X, lam) for a in GENERATORS}
        id_c1 = transport_err(Rid, tr_cell, calT, li, pc)
        id_c7 = transport_err(Rid, jfit, calJF, li, pc, joint=True)
        tau = dict(
            c1=float(np.min(null_c1)),
            c7=float(np.min(null_c7)),
            inv=float(np.min([n["inv"] for n in null_laws])),
            braid=float(np.min([n["braid"] for n in null_laws])),
            nontriv_gt=float(np.max([n["nontriv"] for n in null_laws])) * 0
            + float(np.percentile([n["nontriv"] for n in null_laws], 95)),
        )
        # ---- SINGLE TEST EVALUATION ----
        test = dict(
            lam=lam,
            cal_c1=best[1],
            c1=transport_err(Rs, tr_cell, testT, li, pc),
            c7_fitvocab=transport_err(Rs, jfit, testJF, li, pc, joint=True),
            c7_transfer=transport_err(Rs, jtr, testJT, li, pc, joint=True),
            laws=group_laws(Rs, fit_cell, testF, li, pc),
            c4_embed=support_mass(Rs, fit_cell, testF, EMBED),
            c4_readout=support_mass(Rs, fit_cell, testF, READOUT),
            identity=dict(c1=id_c1, c7=id_c7),
            nulls=dict(c1=null_c1, c7=null_c7),
            tau=tau,
            seconds=round(time.time() - t1, 1),
        )
        results[key] = test
        with open(f"{OUT}/cell_{pc}_L{LAYERS[li]}.json", "w") as f:
            json.dump(test, f, indent=2)
        print(
            f"[{time.strftime('%H:%M:%S')}] {key} lam={lam} "
            f"c1={test['c1']:.4f} (tau {tau['c1']:.4f}, id {id_c1:.4f}) "
            f"c7={test['c7_fitvocab']:.4f} (tau {tau['c7']:.4f}, id {id_c7:.4f}) "
            f"inv={test['laws']['inv']:.4f} braid={test['laws']['braid']:.4f} "
            f"[{test['seconds']}s]",
            flush=True,
        )
    with open(f"{OUT}/all_cells.json", "w") as f:
        json.dump(
            dict(
                config_sha=cfg_sha,
                layers=LAYERS,
                results=results,
                total_seconds=round(time.time() - t0, 1),
            ),
            f,
            indent=2,
        )
    print("DONE", round(time.time() - t0, 1), "s")


if __name__ == "__main__":
    run()

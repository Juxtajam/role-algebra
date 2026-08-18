"""Section B — nonlinear transport at Phase 9 P-path binding sites (C1 only).

Coverage extension of phase10_B_nonlinear_transport.py to the entity-mention
and query-argument positions, using the local Phase 9 P cache. C2 (cross-path)
is UNAVAILABLE here (no G binding-site cache), so per the frozen config this is
CAPPED: no H_role language is possible; the most a positive could mean is
"path-local nonlinear structure" (B5), never a role algebra. Precondition: H6
PASS.

Reuses the Phase 9 fit module (mmap cache, pairing) + the nonlinear battery.
Writes results/phase10/nonlinear/B_binding_sites.json
"""

import hashlib
import importlib.util
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
import nonlinear_transport as nl  # noqa: E402

spec = importlib.util.spec_from_file_location(
    "p9fit", ROOT / "results/binding_sites/code/phase9_item3_fit.py"
)
fit = importlib.util.module_from_spec(spec)
sys.modules["p9fit"] = fit
spec.loader.exec_module(fit)

CFG = ROOT / "phase10/nonlinear/committed_config.json"
LAYER = 61
RANKS = [32, 128, 512]
WIDTHS = [16, 64, 256]
MARGIN = 0.8
POSITIONS = ["query_arg", "answer"]  # single-column classes (clean pairing)


def check_h6():
    h6 = json.load(open(ROOT / "results/extensions/H6_results.json"))
    assert h6.get("H6_PASS"), "H6 did not pass"
    return h6["config_sha256"]


def pairs(cell, bases, pc, li):
    """(X, Y) per generator at position class pc, layer index li (frozen
    pair_samples for single-column classes)."""
    out = {}
    for a in fit.GENERATORS:
        rx, cx, ry, cy = fit.pair_samples(cell, bases, a, pc)
        out[a] = (cell.states(rx, li, cx), cell.states(ry, li, cy))
    return out


def main():
    cfg_sha = hashlib.sha256(CFG.read_bytes()).hexdigest()
    assert check_h6() == cfg_sha
    li = fit.LAYERS.index(LAYER)
    fit_cell, tr_cell = fit.CELLS[("frozen", "fit")], fit.CELLS[("frozen", "transfer")]
    calF, testF = fit.SPLIT[("frozen", "fit")]
    calT, testT = fit.SPLIT[("frozen", "transfer")]
    print(
        f"H6 PASS; config {cfg_sha[:12]}…  L{LAYER}  (C1 only; C2 UNAVAILABLE -> capped)"
    )

    out = {
        "config_sha256": cfg_sha,
        "layer": LAYER,
        "cap": "C2 unavailable (P-only cache); no H_role possible here",
        "positions": {},
    }
    print(
        f"\n{'pos':>12} {'tier':>6} | {'lin_c1':>7} | {'mlp_c1':>7} {'w':>4} | reveal_c1"
    )
    for pc in POSITIONS:
        fit_pairs = pairs(fit_cell, testF, pc, li)
        c1_pairs = pairs(tr_cell, testT, pc, li)
        poolX = np.concatenate([X for X, _ in fit_pairs.values()])
        pres = {}
        for r in RANKS:
            mu, Pb = nl.pca_basis(poolX, r)
            fitR = {
                a: (nl.reduce(X, mu, Pb), nl.reduce(Y, mu, Pb))
                for a, (X, Y) in fit_pairs.items()
            }
            c1R = {
                a: (nl.reduce(X, mu, Pb), nl.reduce(Y, mu, Pb))
                for a, (X, Y) in c1_pairs.items()
            }

            def battery(cls, ev, **kw):
                return float(
                    np.mean(
                        [
                            nl.fit_and_eval(cls, *fitR[a], *ev[a], **kw)
                            for a in fit.GENERATORS
                        ]
                    )
                )

            def tau(ev):
                nulls = []
                for s in range(10):
                    rng = np.random.default_rng(s)
                    errs = []
                    for a in fit.GENERATORS:
                        Xf, Yf = fitR[a]
                        f = nl.ridge_map(Xf, Yf[rng.permutation(len(Yf))], 1e-2)
                        errs.append(nl.transport_err(f, *ev[a]))
                    nulls.append(float(np.mean(errs)))
                return float(np.quantile(nulls, 0.05))

            lin_c1 = battery("linear", c1R)
            tau_c1 = tau(c1R)
            tier = {"linear": dict(c1=lin_c1, tau_c1=tau_c1)}
            for w in WIDTHS:
                m1 = battery("mlp", c1R, width=w)
                reveal = bool(m1 < MARGIN * min(lin_c1, tau_c1) and m1 < 0.8)
                tier[f"mlp_w{w}"] = dict(c1=m1, reveals_c1=reveal)
                print(
                    f"{pc:>12} {'r'+str(r):>6} | {lin_c1:7.3f} | "
                    f"{m1:7.3f} {w:>4} | {reveal}"
                )
            pres[f"r{r}"] = tier
        out["positions"][pc] = pres

    reveals = [
        (pc, rk, k)
        for pc, ps in out["positions"].items()
        for rk, t in ps.items()
        for k, v in t.items()
        if k.startswith("mlp") and v["reveals_c1"]
    ]
    out["any_c1_reveal"] = bool(reveals)
    out["revealing"] = reveals
    out["verdict"] = (
        "No nonlinear C1 advantage at any P-path binding site/tier — consistent "
        "with the answer-position result; the linear binding-site negative (E16) "
        "is not a low-capacity-nonlinear artifact either."
        if not reveals
        else f"C1-only nonlinear advantage at {reveals} — CAPPED to 'path-local nonlinear "
        "structure' (B5); C2 unavailable so NOT a role algebra; would need a G "
        "binding-site cache (new GPU session) to test further."
    )
    json.dump(
        out,
        open(ROOT / "results/extensions/B_binding_sites.json", "w"),
        indent=1,
    )
    print(f"\nVERDICT: {out['verdict']}")


if __name__ == "__main__":
    main()

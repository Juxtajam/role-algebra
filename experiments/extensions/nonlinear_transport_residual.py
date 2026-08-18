"""Section B — nonlinear transport on the real 8C cache.

Runs the H6-validated nonlinear battery at the 8C answer position (verdict
layer L61, both paths, both vocabularies) to test whether role transport
exists that the LINEAR H_retrieval verdict (E13) missed. Precondition: H6
must PASS (checked from results/phase10/nonlinear/H6_results.json).

Frozen decision (committed_config.json): a map class reveals nonlinear
transport iff it beats identity AND the linear anchor AND the shuffled null,
by the H6 margin, on C1 AND C2, at some capacity tier; the claim attaches to
the lowest passing tier; a top-tier-only pass is memorisation-consistent.

Reuses phase8c_lib (frozen 8C pairing, splits, cells). CPU, from cached
activations. Writes results/phase10/nonlinear/B_answer_position.json
"""

import hashlib
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))
import activation_discriminator as p8  # noqa: E402
import nonlinear_transport as nl  # noqa: E402

CFG = ROOT / "phase10/nonlinear/committed_config.json"
LAYER = 61
RANKS = [32, 128, 512]
WIDTHS = [16, 64, 256]
MARGIN = 0.8  # same as H6


def check_h6():
    h6 = json.load(open(ROOT / "results/extensions/H6_results.json"))
    assert h6.get("H6_PASS"), "H6 did not pass — section B is uninterpretable"
    return h6["config_sha256"]


def build_pairs(cell, bases, layer):
    """Per generator: (X, Y) stacked states for the frozen pairing."""
    out = {}
    for a in p8.GENERATORS:
        rx, ry = p8.pair_rows(cell, bases, a)
        out[a] = (cell.states(rx, layer), cell.states(ry, layer))
    return out


def main():
    cfg_sha = hashlib.sha256(CFG.read_bytes()).hexdigest()
    h6_sha = check_h6()
    assert h6_sha == cfg_sha, "H6 ran under a different config"
    print(f"H6 PASS; config {cfg_sha[:12]}…  layer L{LAYER}")

    cells = p8.load_cells()
    splits = json.load(open(ROOT / "results/verdict/discriminator/splits.json"))["splits"]
    fitB = splits["P/fit"]["test"]  # generators fit on P/fit TEST (8C precedent)
    c1B = splits["P/transfer"]["test"]  # C1 disjoint vocabulary
    c2B = splits["G/fit"]["test"]  # C2 cross-path

    fit_pairs = build_pairs(cells[("P", "fit")], fitB, LAYER)
    c1_pairs = build_pairs(cells[("P", "transfer")], c1B, LAYER)
    c2_pairs = build_pairs(cells[("G", "fit")], c2B, LAYER)
    poolX = np.concatenate([X for X, _ in fit_pairs.values()])

    out = {"config_sha256": cfg_sha, "layer": LAYER, "tiers": {}}
    any_reveal = False
    print(
        f"\n{'tier':>10} | {'lin_c1':>7} {'lin_c2':>7} | "
        f"{'mlp_c1':>7} {'mlp_c2':>7} {'w':>4} | reveal"
    )
    for r in RANKS:
        mu, P = nl.pca_basis(poolX, r)
        fitR = {
            a: (nl.reduce(X, mu, P), nl.reduce(Y, mu, P))
            for a, (X, Y) in fit_pairs.items()
        }
        c1R = {
            a: (nl.reduce(X, mu, P), nl.reduce(Y, mu, P))
            for a, (X, Y) in c1_pairs.items()
        }
        c2R = {
            a: (nl.reduce(X, mu, P), nl.reduce(Y, mu, P))
            for a, (X, Y) in c2_pairs.items()
        }

        def battery(cls, ev, **kw):
            return float(
                np.mean(
                    [
                        nl.fit_and_eval(cls, *fitR[a], *ev[a], **kw)
                        for a in p8.GENERATORS
                    ]
                )
            )

        def tau(cls, ev, **kw):
            nulls = []
            for s in range(10):
                rng = np.random.default_rng(s)
                errs = []
                for a in p8.GENERATORS:
                    Xf, Yf = fitR[a]
                    Yfs = Yf[rng.permutation(len(Yf))]
                    f = (
                        nl.ridge_map(Xf, Yfs, kw.get("lam", 1e-2))
                        if cls == "linear"
                        else nl.mlp_map(Xf, Yfs, kw["width"], seed=s)
                    )
                    errs.append(nl.transport_err(f, *ev[a]))
                nulls.append(float(np.mean(errs)))
            return float(np.quantile(nulls, 0.05))

        lin_c1, lin_c2 = battery("linear", c1R), battery("linear", c2R)
        tau_c1, tau_c2 = tau("linear", c1R), tau("linear", c2R)
        tier = {"linear": dict(c1=lin_c1, c2=lin_c2, tau_c1=tau_c1, tau_c2=tau_c2)}
        k1, k2 = battery("kernel", c1R), battery("kernel", c2R)
        krev = bool(
            k1 < MARGIN * min(lin_c1, tau_c1)
            and k1 < 0.8
            and k2 < MARGIN * min(lin_c2, tau_c2)
            and k2 < 0.8
        )
        any_reveal = any_reveal or krev
        tier["kernel"] = dict(c1=k1, c2=k2, reveals=krev)
        print(
            f"{'r'+str(r):>10} | {lin_c1:7.3f} {lin_c2:7.3f} | "
            f"{k1:7.3f} {k2:7.3f} {'ker':>4} | {krev}"
        )
        for w in WIDTHS:
            m1, m2 = battery("mlp", c1R, width=w), battery("mlp", c2R, width=w)
            reveal = bool(
                m1 < MARGIN * min(lin_c1, tau_c1)
                and m1 < 0.8
                and m2 < MARGIN * min(lin_c2, tau_c2)
                and m2 < 0.8
            )
            any_reveal = any_reveal or reveal
            tier[f"mlp_w{w}"] = dict(c1=m1, c2=m2, reveals=reveal)
            print(
                f"{'r'+str(r):>10} | {lin_c1:7.3f} {lin_c2:7.3f} | "
                f"{m1:7.3f} {m2:7.3f} {w:>4} | {reveal}"
            )
        out["tiers"][f"r{r}"] = tier

    # verdict
    reveals = [
        (rk, k)
        for rk, t in out["tiers"].items()
        for k, v in t.items()
        if (k.startswith("mlp") or k == "kernel") and v["reveals"]
    ]
    out["any_reveal"] = any_reveal
    out["revealing_tiers"] = reveals
    if not any_reveal:
        verdict = (
            "NO nonlinear transport beyond linear at the answer position: "
            "no map class beats the linear anchor by margin on C1 AND C2 at "
            "any tier. The linear H_retrieval negative (E13) is NOT a "
            "map-class artifact — the model has no nonlinear role operator "
            "the battery can find here either."
        )
    else:
        lowest = reveals[0]
        top_only = all(rk == f"r{RANKS[-1]}" for rk, _ in reveals)
        verdict = (
            f"Nonlinear transport revealed at {reveals}; lowest tier {lowest}. "
            + (
                "TOP-TIER-ONLY -> memorisation-consistent, not transport (B6); "
                "treat as negative pending capacity-curve confirmation."
                if top_only
                else "Enters the Section I revival pipeline: name-support check "
                "(B7), fresh-vocab/seed replication (I6), causal confirmation "
                "(I7) before any interpretation."
            )
        )
    out["verdict"] = verdict
    json.dump(
        out,
        open(ROOT / "results/extensions/B_answer_position.json", "w"),
        indent=1,
    )
    print(f"\nVERDICT: {verdict}")


if __name__ == "__main__":
    main()

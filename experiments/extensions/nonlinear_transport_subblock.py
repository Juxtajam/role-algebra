"""Phase 10 section F — transport battery on attention-output and MLP-output.

Runs the H6-validated nonlinear transport battery (identical to section B) on
the 72B attn_out and mlp_out caches at the answer position, per layer. Tests
whether a transportable role operator (linear or low-complexity nonlinear)
lives in these non-residual objects where the residual stream had none.

Precondition: H6 PASS. Config phase10/nonlinear/committed_config_F.json.
Reuses phase8c_lib pairing (compose, PERMS, splits) + nl_transport_lib.

Cache layout: results/phase10/nonlinear/actsF/{cell}_{attn,mlp}.npy,
shape (1800, len(LAYERS), 8192), fp16, answer position.

Writes results/phase10/nonlinear/F_results.json
"""

import hashlib
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))
import activation_discriminator as p8  # noqa: E402  (compose, PERMS, GENERATORS, splits)
import nonlinear_transport as nl  # noqa: E402

CFG = ROOT / "phase10/nonlinear/committed_config_F.json"
ACTS = ROOT / "results/extensions/actsF"
LAYERS = [40, 48, 56, 60, 61, 64, 68, 72]
RANKS = [32, 128, 512]
WIDTHS = [16, 64, 256]
MARGIN = 0.8


def check_h6():
    h6 = json.load(open(ROOT / "results/extensions/H6_results.json"))
    assert h6.get("H6_PASS"), "H6 did not pass — section F uninterpretable"


def rows_for(bases):
    """Frozen pairing rows per generator: (rx, ry) with y = compose(a,g)."""
    out = {}
    for a in p8.GENERATORS:
        rx, ry = [], []
        for b in bases:
            for g in p8.PERMS:
                rx.append(b * 6 + p8.PERMS.index(g))
                ry.append(b * 6 + p8.PERMS.index(p8.compose(a, g)))
        out[a] = (np.array(rx), np.array(ry))
    return out


def main():
    check_h6()
    cfg_sha = hashlib.sha256(CFG.read_bytes()).hexdigest()
    FROZEN_HASH = "84f2e54d85d6e8aa4c1474b608bef5ab69babe54353ef0ef2702d9f6ed38baef"
    meta = json.load(open(ACTS / "checksums.json"))
    assert meta["config_F_sha"] == cfg_sha, "cache built under a different F config"
    assert meta["frozen_hash"] == FROZEN_HASH, "cache built on wrong frozen episodes"
    # verify per-file sha256 matches the in-session checksums (bytes intact)
    for name, rec in meta["files"].items():
        got = hashlib.sha256((ACTS / f"{name}.npy").read_bytes()).hexdigest()
        assert got == rec["sha256"], f"sha mismatch {name}: {got} != {rec['sha256']}"
    print(
        f"H6 PASS; config_F {cfg_sha[:12]}…; cache frozen_hash {meta['frozen_hash'][:12]}…; "
        f"{len(meta['files'])} files sha-verified"
    )

    splits = json.load(open(ROOT / "results/verdict/discriminator/splits.json"))["splits"]
    fitB = splits["P/fit"]["test"]
    c1B = splits["P/transfer"]["test"]
    c2B = splits["G/fit"]["test"]

    caches = {}
    for cell in ("disc_P_fit", "disc_P_transfer", "disc_G_fit"):
        for kind in ("attn", "mlp"):
            caches[(cell, kind)] = np.load(ACTS / f"{cell}_{kind}.npy", mmap_mode="r")

    out = {"config_F_sha256": cfg_sha, "objects": {}}
    print(
        f"\n{'object':>5} {'layer':>5} {'tier':>5} | {'lin C1':>7} {'lin C2':>7} | "
        f"{'best nl C1':>10} {'best nl C2':>10} | reveal"
    )
    any_reveal_global = False
    for kind in ("attn", "mlp"):
        obj = {}
        for L in LAYERS:
            li = LAYERS.index(L)

            def pairs(cell, bases):
                z = caches[(cell, kind)]
                d = {}
                for a in p8.GENERATORS:
                    rx, ry = rows_for(bases)[a]
                    d[a] = (
                        np.asarray(z[rx, li, :], np.float64),
                        np.asarray(z[ry, li, :], np.float64),
                    )
                return d

            fit_pairs = pairs("disc_P_fit", fitB)
            c1_pairs = pairs("disc_P_transfer", c1B)
            c2_pairs = pairs("disc_G_fit", c2B)
            poolX = np.concatenate([X for X, _ in fit_pairs.values()])
            lres = {}
            for r in RANKS:
                mu, P = nl.pca_basis(poolX, r)
                fr = {
                    a: (nl.reduce(X, mu, P), nl.reduce(Y, mu, P))
                    for a, (X, Y) in fit_pairs.items()
                }
                e1 = {
                    a: (nl.reduce(X, mu, P), nl.reduce(Y, mu, P))
                    for a, (X, Y) in c1_pairs.items()
                }
                e2 = {
                    a: (nl.reduce(X, mu, P), nl.reduce(Y, mu, P))
                    for a, (X, Y) in c2_pairs.items()
                }

                def bat(cls, ev, **kw):
                    return float(
                        np.mean(
                            [
                                nl.fit_and_eval(cls, *fr[a], *ev[a], **kw)
                                for a in p8.GENERATORS
                            ]
                        )
                    )

                def tau(ev):
                    ns = []
                    for s in range(10):
                        rng = np.random.default_rng(s)
                        errs = []
                        for a in p8.GENERATORS:
                            Xf, Yf = fr[a]
                            f = nl.ridge_map(Xf, Yf[rng.permutation(len(Yf))], 1e-2)
                            errs.append(nl.transport_err(f, *ev[a]))
                        ns.append(float(np.mean(errs)))
                    return float(np.quantile(ns, 0.05))

                lin1, lin2 = bat("linear", e1), bat("linear", e2)
                t1, t2 = tau(e1), tau(e2)
                tier = {"linear": dict(c1=lin1, c2=lin2, tau_c1=t1, tau_c2=t2)}
                nl_c1s, nl_c2s = [], []
                for w in WIDTHS:
                    m1, m2 = bat("mlp", e1, width=w), bat("mlp", e2, width=w)
                    rev = bool(
                        m1 < MARGIN * min(lin1, t1)
                        and m1 < 0.8
                        and m2 < MARGIN * min(lin2, t2)
                        and m2 < 0.8
                    )
                    tier[f"mlp_w{w}"] = dict(c1=m1, c2=m2, reveals=rev)
                    nl_c1s.append(m1)
                    nl_c2s.append(m2)
                k1, k2 = bat("kernel", e1), bat("kernel", e2)
                krev = bool(
                    k1 < MARGIN * min(lin1, t1)
                    and k1 < 0.8
                    and k2 < MARGIN * min(lin2, t2)
                    and k2 < 0.8
                )
                tier["kernel"] = dict(c1=k1, c2=k2, reveals=krev)
                nl_c1s.append(k1)
                nl_c2s.append(k2)
                rev_any = any(
                    v.get("reveals") for kk, v in tier.items() if kk != "linear"
                )
                any_reveal_global = any_reveal_global or rev_any
                lres[f"r{r}"] = tier
                print(
                    f"{kind:>5} {L:>5} {'r'+str(r):>5} | {lin1:7.3f} {lin2:7.3f} | "
                    f"{min(nl_c1s):10.3f} {min(nl_c2s):10.3f} | {rev_any}"
                )
            obj[f"L{L}"] = lres
        out["objects"][kind] = obj

    reveals = [
        (kind, L, r, k)
        for kind, o in out["objects"].items()
        for L, ls in o.items()
        for r, t in ls.items()
        for k, v in t.items()
        if k != "linear" and v.get("reveals")
    ]
    out["any_reveal"] = any_reveal_global
    out["revealing"] = reveals
    out["verdict"] = (
        "No transportable role operator (linear or low-complexity nonlinear) in the "
        "attention output or the MLP output at the answer position, at any layer or "
        "tier. The residual-stream negative (E13/E16/E21) extends to these non-residual "
        "objects: role structure is not hiding in the attn/mlp writes either."
        if not reveals
        else f"Transport revealed in {reveals} — refines the scope: role structure in the "
        f"attn/mlp write. Enters Section I (fresh-vocab replication, causal confirmation) "
        f"before interpretation; a top-tier-only pass is memorisation-consistent (B6)."
    )
    json.dump(
        out, open(ROOT / "results/extensions/F_results.json", "w"), indent=1
    )
    print(f"\nVERDICT: {out['verdict']}")


if __name__ == "__main__":
    main()

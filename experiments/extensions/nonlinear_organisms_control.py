"""H6 — positive control for the nonlinear transport battery.

Plants a NONLINEAR role code (S-role-nonlinear) and checks that the LINEAR
battery FAILS it while the nonlinear (MLP) battery RECOVERS it, at Stage-1
SNRs. Controls: S-role (linear; both recover) and S-retrieval (neither).

If H6 does not pass, the section-B results on the real cache are
uninterpretable and must not be read (frozen precondition in
phase10/nonlinear/committed_config.json).

Writes results/phase10/nonlinear/H6_results.json
"""

import hashlib
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from synth.model import Frame, compose, transpositions, ROLE_DIM, CONT_DIM  # noqa: E402
from synth.organisms import SRole, SRetrieval  # noqa: E402
import nonlinear_transport as nl  # noqa: E402


class SRoleNL(SRole):
    """Content-entangled nonlinear role code (config amendment 2):
    role_block(x) = tanh(2 (A @ content(x) + vertex(slot(x)))), A fixed random.
    Transport slot j->g(j) is nonlinear and content-dependent, so no linear
    map transports it, while an MLP can. Role info preserved."""

    name = "S-role-nonlinear"

    def __init__(self, frame, snr, seed=0):
        super().__init__(frame, snr, seed)
        self.A = self._org_rng.standard_normal((ROLE_DIM, CONT_DIM)) / np.sqrt(CONT_DIM)

    def role_blocks(self, eps):
        v = self.f.vertices[self.answers(eps)]  # (n, ROLE_DIM)
        c = np.stack([self.f.content[p][cid] for (p, cid, _) in eps])  # (n, CONT_DIM)
        return np.tanh(2.0 * (c @ self.A.T + v))


CFG = ROOT / "phase10/nonlinear/committed_config.json"
LAYER = 2
SNRS = [10, 3]
RANKS = [16, 32]
WIDTHS = [16, 64]
MARGIN = 0.8  # a reveal must beat the linear anchor / null by >= 20%


def phi_factory(d, ref_states, seed=20260815):
    """Fixed smooth invertible nonlinearity (config amendment 1):
    phi(z) = O^T[ u + tanh(2u) ] @ O,  u = (O z - mu)/sigma,  with (mu, sigma)
    frozen per-dim mean/std of the ROTATED reference states O @ ref^T. The
    standardization puts u at O(1) so tanh(2u) is genuinely nonlinear;
    monotonic in u -> invertible on range; role info preserved. O fixed
    random orthogonal."""
    rng = np.random.default_rng(seed)
    O, _ = np.linalg.qr(rng.standard_normal((d, d)))
    U = ref_states @ O.T
    mu, sigma = U.mean(0), U.std(0) + 1e-9

    def phi(Z):
        u = (Z @ O.T - mu) / sigma
        return (u + np.tanh(2 * u)) @ O

    return phi


def pair_eps(org, bases, a):
    xs, ys = [], []
    for b in bases:
        orb = org.orbit(b)
        for g, ep in orb.items():
            xs.append(ep)
            ys.append(orb[compose(a, g)])
    return xs, ys


def states_of(org, eps, phi):
    h = org.states(eps, LAYER)
    return phi(h) if phi is not None else h


def run_organism(org, phi, label):
    gens = transpositions(org.k)
    fitB = org.bases("P", "fit", "cal")
    c1B = org.bases("P", "transfer", "test")
    c2B = org.bases("G", "fit", "test")

    # pooled fit X for the PCA basis (both generators)
    poolX = []
    fit_pairs = {}
    for a in gens:
        xs, ys = pair_eps(org, fitB, a)
        X = states_of(org, xs, phi)
        Y = states_of(org, ys, phi)
        fit_pairs[a] = (X, Y)
        poolX.append(X)
    poolX = np.concatenate(poolX)

    def eval_pairs(bases):
        out = {}
        for a in gens:
            xs, ys = pair_eps(org, bases, a)
            out[a] = (states_of(org, xs, phi), states_of(org, ys, phi))
        return out

    c1_pairs, c2_pairs = eval_pairs(c1B), eval_pairs(c2B)

    res = {"tiers": {}}
    for r in RANKS:
        mu, P = nl.pca_basis(poolX, r)
        # reduce all
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

        def battery(cls, evalR, **kw):
            errs = [nl.fit_and_eval(cls, *fitR[a], *evalR[a], **kw) for a in gens]
            return float(np.mean(errs))

        def shuffled_tau(cls, evalR, **kw):
            nulls = []
            for s in range(10):
                rng = np.random.default_rng(s)
                errs = []
                for a in gens:
                    Xf, Yf = fitR[a]
                    Yfs = Yf[rng.permutation(len(Yf))]
                    f = (
                        nl.ridge_map(Xf, Yfs, kw.get("lam", 1e-2))
                        if cls == "linear"
                        else (
                            nl.mlp_map(Xf, Yfs, kw["width"], seed=s)
                            if cls == "mlp"
                            else nl.kernel_map(Xf, Yfs)
                        )
                    )
                    errs.append(nl.transport_err(f, *evalR[a]))
                nulls.append(float(np.mean(errs)))
            return float(np.quantile(nulls, 0.05))

        lin_c1 = battery("linear", c1R)
        lin_c2 = battery("linear", c2R)
        tau_c1 = shuffled_tau("linear", c1R)
        tau_c2 = shuffled_tau("linear", c2R)
        tier = {
            "linear": {"c1": lin_c1, "c2": lin_c2, "tau_c1": tau_c1, "tau_c2": tau_c2}
        }
        # kernel ridge arm (config map class 'kernel'); reveal same rule
        k_c1 = battery("kernel", c1R)
        k_c2 = battery("kernel", c2R)
        tier["kernel"] = {
            "c1": k_c1,
            "c2": k_c2,
            "reveals": bool(
                k_c1 < MARGIN * min(lin_c1, tau_c1)
                and k_c1 < 0.8
                and k_c2 < MARGIN * min(lin_c2, tau_c2)
                and k_c2 < 0.8
            ),
        }
        for w in WIDTHS:
            m_c1 = battery("mlp", c1R, width=w)
            m_c2 = battery("mlp", c2R, width=w)
            # reveal: the MLP beats identity AND beats the linear anchor / null
            # by a relative MARGIN, on C1 AND C2 (so regression-to-mean noise,
            # where mlp ~= linear ~= 0.577, does NOT count).
            reveals = bool(
                m_c1 < MARGIN * min(lin_c1, tau_c1)
                and m_c1 < 0.8
                and m_c2 < MARGIN * min(lin_c2, tau_c2)
                and m_c2 < 0.8
            )
            tier[f"mlp_w{w}"] = {"c1": m_c1, "c2": m_c2, "reveals": reveals}
        res["tiers"][f"r{r}"] = tier
    return res


def main():
    cfg_sha = hashlib.sha256(CFG.read_bytes()).hexdigest()
    print(f"config sha256={cfg_sha[:16]}…  layer={LAYER}")
    frame = Frame(k=3)
    out = {
        "config_sha256": cfg_sha,
        "layer": LAYER,
        "amendments": [
            "amendment1 (015a6195, superseded)",
            "amendment2 (2085c7e6, content-entangled role code)",
        ],
        "organisms": {},
    }
    for snr in SNRS:
        specs = [
            ("S-role", SRole(frame, snr)),
            ("S-role-nonlinear", SRoleNL(frame, snr)),
            ("S-retrieval", SRetrieval(frame, snr)),
        ]
        for label, org in specs:
            r = run_organism(org, None, label)
            out["organisms"][f"{label}@{snr}"] = r
            # summary line
            best = {}
            for rk, tier in r["tiers"].items():
                lin = tier["linear"]
                mlps = {k: v for k, v in tier.items() if k.startswith("mlp")}
                any_rev = any(v["reveals"] for v in mlps.values())
                best[rk] = (lin["c1"], min(v["c1"] for v in mlps.values()), any_rev)
            line = " | ".join(
                f"{rk}: lin_c1={b[0]:.3f} mlp_c1={b[1]:.3f} reveal={b[2]}"
                for rk, b in best.items()
            )
            print(f"  {label}@{snr}: {line}")

    # H6 pass criterion (frozen; the meaningful validation is that the
    # nonlinear battery shows an advantage over linear ONLY where a nonlinear
    # code is planted).
    def linear_recovers(o):  # linear transports well on C1 at some rank
        return any(t["linear"]["c1"] < 0.1 for t in o["tiers"].values())

    def mlp_reveals(o):  # MLP beats linear by margin on C1 AND C2 somewhere
        return any(
            v["reveals"]
            for t in o["tiers"].values()
            for k, v in t.items()
            if k.startswith("mlp") or k == "kernel"
        )

    checks = {}
    for snr in SNRS:
        srole = out["organisms"][f"S-role@{snr}"]
        srnl = out["organisms"][f"S-role-nonlinear@{snr}"]
        sret = out["organisms"][f"S-retrieval@{snr}"]
        checks[str(snr)] = dict(
            srole_linear_recovers=linear_recovers(srole),
            srole_no_nonlinear_advantage=not mlp_reveals(srole),
            nonlinear_detected_by_mlp=mlp_reveals(srnl),
            retrieval_nothing_recovers=(
                not linear_recovers(sret) and not mlp_reveals(sret)
            ),
        )
    out["H6_checks"] = checks
    # pass: at every SNR, the four checks hold
    passed = all(all(v.values()) for v in checks.values())
    out["H6_PASS"] = bool(passed)
    json.dump(
        out, open(ROOT / "results/extensions/H6_results.json", "w"), indent=1
    )
    print("\nH6 checks per SNR:")
    for snr, c in checks.items():
        print(f"  SNR {snr}: {c}")
    print(
        f"\nH6 {'PASS — nonlinear battery validated; section B may proceed' if passed else 'FAIL — section B uninterpretable, do not read'}"
    )
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()

"""Stage 1 organisms (spec v3, "Organisms").

Every organism exposes the uniform *system* interface consumed by
shared/discriminator.py:

    k, d, n_layers, generators, all_perms, crosspath_available
    orbit(base)                -> {g: episode}          episode = (path, cid, g)
    bases(path, vocab, split)  -> list of (path, cid)
    states(eps, layer)         -> (n, d) fresh-noise states
    answers(eps)               -> (n,) answer slots
    decode(eps)                -> (n,) natural-run predictions (final layer)
    decode_from(H, layer, eps) -> (n,) run remaining layers from patched
                                  states H at `layer`, then decode
    u_vectors(eps, layer)      -> (n, k, d) per-episode u_i for the support
                                  test (span{u_i - u_j})
    content_labels(eps)        -> (n,) content-group labels for the probes
    self_decode_accuracy()     -> gate: must be >= 0.99 before the
                                  discriminator runs
"""

import numpy as np

from synth.model import MASTER_SEED, PATHS, ROLE_DIM, Frame, perms, transpositions


class SynthOrganism:
    name = "?"

    def __init__(self, frame: Frame, snr: float, seed: int = 0):
        self.f = frame
        self.snr = snr
        self.k, self.d, self.n_layers = frame.k, frame.d, frame.n_layers
        self.generators = transpositions(frame.k)
        self.all_perms = perms(frame.k)
        self.crosspath_available = True
        self._rng = np.random.default_rng((MASTER_SEED, seed, int(snr * 1000)))
        import zlib

        org_id = zlib.crc32(
            self.name.encode()
        )  # stable across processes (hash() is not)
        self._org_rng = np.random.default_rng((MASTER_SEED, 555, org_id))

    # -- episode bookkeeping ------------------------------------------------
    def orbit(self, base):
        path, cid = base
        return {g: (path, cid, g) for g in self.all_perms}

    def bases(self, path, vocab, split):
        return self.f.bases(path, vocab, split)

    def answers(self, eps):
        return np.array([self.f.answer_slot(p, c, g) for (p, c, g) in eps])

    def content_labels(self, eps):
        return np.array([self.f.content_label(p, c) for (p, c, _) in eps])

    # -- state generation ----------------------------------------------------
    def role_blocks(self, eps):
        raise NotImplementedError

    def codes(self, eps):
        rb = self.role_blocks(eps)
        cb = np.stack([self.f.content[p][c] for (p, c, _) in eps])
        return np.concatenate([rb, cb], axis=1)

    def states(self, eps, layer):
        code = self.codes(eps)
        h = code @ self.f.M[layer].T
        role_norm = np.linalg.norm(code[:, :ROLE_DIM], axis=1, keepdims=True)
        sigma = role_norm / (self.snr * np.sqrt(self.d))
        return h + self._rng.standard_normal(h.shape) * sigma

    def run_from(self, H, layer):
        """Apply the layer transitions T_l = M_{l+1} M_l^T up to the final layer."""
        for l in range(layer, self.n_layers - 1):
            H = (H @ self.f.M[l]) @ self.f.M[l + 1].T
        return H

    # -- decoding -------------------------------------------------------------
    def _decode_states(self, H, eps):
        raise NotImplementedError

    def decode(self, eps):
        return self._decode_states(self.states(eps, self.n_layers - 1), eps)

    def decode_from(self, H, layer, eps):
        return self._decode_states(self.run_from(H, layer), eps)

    # -- support-test vectors ---------------------------------------------------
    def u_vectors(self, eps, layer):
        """Empirical slot-conditional mean role vector, mapped to state space.
        For S-role / S-shared this recovers the role vertices; for organisms
        with no slot-consistent role vector the means collapse toward zero."""
        if not hasattr(self, "_u_cache"):
            self._u_cache = {}
        if layer not in self._u_cache:
            cal = [
                ep
                for b in self.bases("P", "fit", "cal")
                for ep in self.orbit(b).values()
            ]
            rb = self.role_blocks(cal)
            ans = self.answers(cal)
            means = np.stack([rb[ans == j].mean(axis=0) for j in range(self.k)])
            self._u_cache[layer] = means @ self.f.M[layer][:, :ROLE_DIM].T
        u = self._u_cache[layer]
        return np.broadcast_to(u, (len(eps),) + u.shape)

    # -- the self-decode gate ------------------------------------------
    def self_decode_accuracy(self, n_draws=4):
        eps = [
            ep
            for path in PATHS
            for vocab in ("fit", "transfer")
            for b in self.bases(path, vocab, "test")
            for ep in self.orbit(b).values()
        ]
        ans = self.answers(eps)
        correct = sum((self.decode(eps) == ans).sum() for _ in range(n_draws))
        return correct / (n_draws * len(eps))


class SRole(SynthOrganism):
    """role_code(g.x) = rho(g) role_code(x): vertex of the standard (k-1)-dim
    representation for the episode's answer slot. Must return H_role."""

    name = "S-role"

    def role_blocks(self, eps):
        return self.f.vertices[self.answers(eps)]

    def _decode_states(self, H, eps):
        logits = (H @ self.f.M[self.n_layers - 1][:, :ROLE_DIM]) @ self.f.vertices.T
        return logits.argmax(axis=1)


class SShared(SynthOrganism):
    """k fixed global vectors permuted by g. A swap operator on span{v_i}
    satisfies the S_k relations, so this likely returns H_role — expected,
    reported as a stated limitation, never gated or tuned against."""

    name = "S-shared"

    def __init__(self, frame, snr, seed=0):
        super().__init__(frame, snr, seed)
        w = self._org_rng.standard_normal((self.k, ROLE_DIM))
        self.w = w / np.linalg.norm(w, axis=1, keepdims=True)

    def role_blocks(self, eps):
        return self.w[self.answers(eps)]

    def _decode_states(self, H, eps):
        logits = (H @ self.f.M[self.n_layers - 1][:, :ROLE_DIM]) @ self.w.T
        return logits.argmax(axis=1)


class SRetrieval(SynthOrganism):
    """role_code(b, g) = z[b, g] drawn independently per (base, permutation);
    no global R_g exists. The decoder is an explicit per-episode lookup
    (nearest stored code -> that episode's tabulated answer), so the organism
    solves its task perfectly with no role algebra. Must return H_retrieval."""

    name = "S-retrieval"

    def __init__(self, frame, snr, seed=0):
        super().__init__(frame, snr, seed)
        self._table_eps = [
            ep
            for path in PATHS
            for vocab in ("fit", "transfer")
            for b in self.bases(path, vocab, "all")
            for ep in self.orbit(b).values()
        ]
        self._z = {}
        for ep in self._table_eps:
            z = self._org_rng.standard_normal(ROLE_DIM)
            self._z[ep] = z / np.linalg.norm(z)
        self._codes = self.codes(self._table_eps)  # lookup keys
        self._answers = self.answers(self._table_eps)  # lookup values

    def role_blocks(self, eps):
        return np.stack([self._z[ep] for ep in eps])

    def _decode_states(self, H, eps):
        code_est = H @ self.f.M[self.n_layers - 1]
        nearest = (code_est @ self._codes.T).argmax(axis=1)
        return self._answers[nearest]


class SPosition(SynthOrganism):
    """State encodes only selected-token identity (the content block) and a
    position index; the decoder reads position. Slots map to positions by a
    per-base scrambled assignment, so no role variable exists and no single
    linear map tracks g across base problems. Must return H_retrieval."""

    name = "S-position"

    def __init__(self, frame, snr, seed=0):
        super().__init__(frame, snr, seed)
        self._pos = {}  # base -> (k,) position indices in [0, ROLE_DIM)
        for path in PATHS:
            for b in self.bases(path, "fit", "all") + self.bases(
                path, "transfer", "all"
            ):
                self._pos[b] = self._org_rng.choice(
                    ROLE_DIM, size=self.k, replace=False
                )

    def role_blocks(self, eps):
        rb = np.zeros((len(eps), ROLE_DIM))
        ans = self.answers(eps)
        for i, (p, c, g) in enumerate(eps):
            rb[i, self._pos[(p, c)][ans[i]]] = 1.0
        return rb

    def _decode_states(self, H, eps):
        role_est = H @ self.f.M[self.n_layers - 1][:, :ROLE_DIM]
        preds = np.empty(len(eps), dtype=int)
        for i, (p, c, g) in enumerate(eps):
            preds[i] = role_est[i, self._pos[(p, c)]].argmax()
        return preds


ORGANISMS = {o.name: o for o in (SRole, SShared, SRetrieval, SPosition)}
# Verdicts required (S-shared is reported, not gated).
REQUIRED = {
    "S-role": {10: "H_role", 3: "H_role"},
    "S-retrieval": {10: "H_retrieval", 3: "H_retrieval", 1: "H_retrieval"},
    "S-position": {10: "H_retrieval", 3: "H_retrieval", 1: "H_retrieval"},
}

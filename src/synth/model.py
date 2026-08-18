"""Stage 1 synthetic linear system (spec v3, "The synthetic model").

State:  h_l(x) = M_l @ [role_code(x); content_code(x)] + noise_l
        T_l    = M_{l+1} @ M_l.T
M_l has orthonormal columns, fixed seed, constant across organisms.
content_code is invariant under g. Noise isotropic Gaussian with
SNR = ||role component|| / ||noise||, SNR in {10, 3, 1}.

Two paths P and G share role structure but have disjoint content
vocabularies and different content->answer wiring. Content vocabulary is
split CONTENT_fit / CONTENT_transfer, 128 each, and each half is further
split calibration / test (disjoint base problems, spec "Verdict" step 1).

An episode is (path, content_id, g) with g in S_k acting on the k
episode-local answer slots; the base problem is (path, content_id).
"""

import itertools

import numpy as np

D = 64
N_LAYERS = 4
ROLE_DIM = 16  # role block of the code (S-position needs room for 16 position indices)
CONT_DIM = 24
CODE_DIM = ROLE_DIM + CONT_DIM
N_CONTENT = 128  # per (path, fit/transfer half)
MASTER_SEED = 20260806
PATHS = ("P", "G")


def perms(k):
    return [tuple(p) for p in itertools.permutations(range(k))]


def compose(a, b):
    """(a o b)[j] = a[b[j]] — apply b, then a."""
    return tuple(a[b[j]] for j in range(len(a)))


def transpositions(k):
    """Adjacent transpositions g_12, g_23, ... — the generators the
    discriminator fits maps for."""
    out = []
    for i in range(k - 1):
        t = list(range(k))
        t[i], t[i + 1] = t[i + 1], t[i]
        out.append(tuple(t))
    return out


def orthonormal_cols(rng, rows, cols):
    q, _ = np.linalg.qr(rng.standard_normal((rows, cols)))
    return q[:, :cols]


class Frame:
    """Fixed random frame shared by ALL organisms: M_l, content codes,
    content->answer wiring, and the standard-representation vertices."""

    def __init__(self, k=3, d=D, n_layers=N_LAYERS, seed=MASTER_SEED):
        self.k, self.d, self.n_layers = k, d, n_layers
        rng = np.random.default_rng(seed)
        self.M = [orthonormal_cols(rng, d, CODE_DIM) for _ in range(n_layers)]
        # content codes: an 8-dim group one-hot block (the attribute the
        # condition-6 content probe reads) plus a 16-dim unique identity part;
        # separate vocabulary per path
        self.content = {}
        self.j0 = {}
        for path in PATHS:
            cid = np.arange(2 * N_CONTENT)
            group = 0.7 * np.eye(8)[cid % 8]
            uniq = rng.standard_normal((2 * N_CONTENT, CONT_DIM - 8))
            uniq = 0.7 * uniq / np.linalg.norm(uniq, axis=1, keepdims=True)
            self.content[path] = np.concatenate([group, uniq], axis=1)
            # content -> answer-slot wiring, different per path
            self.j0[path] = rng.integers(0, k, size=2 * N_CONTENT)
        # standard (k-1)-dim representation of S_k in a fixed random
        # orthonormal basis, embedded in the ROLE_DIM block (used by S-role).
        ones = np.ones((k, 1)) / np.sqrt(k)
        q, _ = np.linalg.qr(np.hstack([ones, rng.standard_normal((k, k - 1))]))
        self.Bc = q[:, 1:]  # k x (k-1), spans mean-zero subspace
        self.Q = orthonormal_cols(rng, ROLE_DIM, k - 1)  # embed into role block
        # vertices v_j = Q @ Bc.T e_j ; satisfies v_{g(j)} = rho(g) v_j
        self.vertices = (self.Q @ self.Bc.T).T  # (k, ROLE_DIM)

    def content_label(self, path, cid):
        return int(cid % 8)

    def rho(self, g):
        """rho(g) acting on the ROLE_DIM block."""
        P = np.zeros((self.k, self.k))
        for j in range(self.k):
            P[g[j], j] = 1.0
        return self.Q @ self.Bc.T @ P @ self.Bc @ self.Q.T

    # ---- episode bookkeeping -------------------------------------------
    # content ids: [0, N)          = fit,      [N, 2N) = transfer
    # within each half: first 50%  = calibration, rest = test
    def bases(self, path, vocab, split):
        lo = 0 if vocab == "fit" else N_CONTENT
        half = N_CONTENT // 2
        if split == "cal":
            ids = range(lo, lo + half)
        elif split == "test":
            ids = range(lo + half, lo + 2 * half)
        else:  # "all"
            ids = range(lo, lo + 2 * half)
        return [(path, c) for c in ids]

    def answer_slot(self, path, cid, g):
        return g[self.j0[path][cid]]

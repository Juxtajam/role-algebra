"""Nonlinear transport battery.

Tests whether role transport exists that a LINEAR operator misses. Given
matched pairs (h(x), h(g.x)) per generator g, fit a family of maps of
increasing capacity and measure relative transport error against the linear
incumbent and the do-nothing identity.

All maps operate in a PCA-reduced space (basis fit on pooled TRAIN x only,
per the design "PCA-r -> small nonlinear map -> reconstruction"), so every
class is compared on the same footing and the raw-d underdetermined regime is
avoided (the design's primary variant). Transport error is the relative,
identity-scores-1 metric used throughout the programme, computed in the
reduced space:

    err(f) = || f(Xr_test) - Yr_test ||^2 / || Yr_test - Xr_test ||^2

Map classes:
  linear  : ridge R (the incumbent / anchor; equals the existing linear test
            restricted to the PCA subspace)
  mlp     : 1-hidden-layer tanh MLP, width w, Adam + weight decay
  kernel  : kernel ridge regression, RBF, bandwidth = median pairwise distance

Conditions:
  C1  fit on fit-vocab TRAIN, eval on transfer-vocab TEST  (disjoint vocab)
  C2  fit on path-P TRAIN,   eval on path-G  TEST          (cross-path)

Decision (frozen; see committed_config.json): a class "reveals nonlinear
transport" iff, at some capacity tier, its error beats BOTH identity (1.0)
AND the linear anchor AND the shuffled-null FPR threshold, on C1 AND C2. The
claim attaches to the LOWEST passing tier (B6). Name-support (B7) and
same-path-only (B5) are checked on any success.
"""

import numpy as np
import torch

torch.manual_seed(0)


def pca_basis(X, r):
    """Top-r PCA basis of centered X (n,d); returns (mu, P) with P (d,r)."""
    mu = X.mean(0)
    Xc = X - mu
    # economy SVD; right singular vectors are PCA directions
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    return mu, Vt[:r].T


def reduce(X, mu, P):
    return (X - mu) @ P


def ridge_map(Xr, Yr, lam):
    d = Xr.shape[1]
    G = Xr.T @ Xr
    lam_eff = lam * np.trace(G) / d + 1e-12
    R = np.linalg.solve(G + lam_eff * np.eye(d), Xr.T @ Yr)  # (r,r), applies as x@R
    return lambda Z: Z @ R


def mlp_map(Xr, Yr, width, epochs=400, lr=1e-2, wd=1e-4, seed=0):
    torch.manual_seed(seed)
    r = Xr.shape[1]
    X = torch.tensor(Xr, dtype=torch.float32)
    Y = torch.tensor(Yr, dtype=torch.float32)
    net = torch.nn.Sequential(
        torch.nn.Linear(r, width), torch.nn.Tanh(), torch.nn.Linear(width, r)
    )
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
    lossf = torch.nn.MSELoss()
    for _ in range(epochs):
        opt.zero_grad()
        loss = lossf(net(X), Y)
        loss.backward()
        opt.step()
    net.eval()

    def apply(Z):
        with torch.no_grad():
            return net(torch.tensor(Z, dtype=torch.float32)).numpy()

    return apply


def kernel_map(Xr, Yr, lam=1e-2):
    """RBF kernel ridge, bandwidth = median pairwise distance (heuristic)."""
    from scipy.spatial.distance import pdist, squareform

    Dm = squareform(pdist(Xr))
    gamma = 1.0 / (2.0 * (np.median(Dm[Dm > 0]) ** 2) + 1e-12)
    K = np.exp(-gamma * Dm**2)
    A = np.linalg.solve(K + lam * np.eye(len(Xr)), Yr)  # (n, r)
    Xtr = Xr

    def apply(Z):
        d2 = ((Z[:, None, :] - Xtr[None, :, :]) ** 2).sum(-1)
        Kz = np.exp(-gamma * d2)
        return Kz @ A

    return apply


def transport_err(f, Xr, Yr):
    return float(((f(Xr) - Yr) ** 2).sum() / max(((Yr - Xr) ** 2).sum(), 1e-12))


def fit_and_eval(cls, Xr_fit, Yr_fit, Xr_eval, Yr_eval, **kw):
    if cls == "linear":
        f = ridge_map(Xr_fit, Yr_fit, kw.get("lam", 1e-2))
    elif cls == "mlp":
        f = mlp_map(Xr_fit, Yr_fit, kw["width"], seed=kw.get("seed", 0))
    elif cls == "kernel":
        f = kernel_map(Xr_fit, Yr_fit, kw.get("lam", 1e-2))
    else:
        raise ValueError(cls)
    return transport_err(f, Xr_eval, Yr_eval)


def name_support_fraction(f_linear_R, name_subspace_Q):
    """B7: fraction of a linear map's (R-I) energy in the name subspace.
    Only defined for the linear anchor (nonlinear maps have no single R);
    for a nonlinear success we instead report the Jacobian-at-mean support
    (computed by the caller). Kept here for the linear reference."""
    RmI = f_linear_R - np.eye(len(f_linear_R))
    num = ((RmI @ name_subspace_Q) ** 2).sum()
    den = max((RmI**2).sum(), 1e-12)
    return float(num / den)

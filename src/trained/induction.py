"""Phase 5 — induction pretraining corpus + verification.

CORPUS v2. The v1 corpus (32 unique tokens repeated once at fixed offset 32)
was VOIDED before Step 2 verification was accepted: with learned absolute
positions and a constant repeat offset, the model learned a purely
POSITIONAL attend-to-(t-31) head (probe: copy acc 1.000 at period 32, 0.001
at periods 21/11) that passes the literal mass thresholds only because
position and content coincide on that distribution — the T0 lesson. Such a
head is not the circuit required ("attend from a
token to the token following its previous occurrence"), and could not be
recruited by task episodes, which have no fixed-offset structure. Declared
deviation; v1 volume outputs purged.

v2: per sequence, draw a prefix of L unique tokens (L uniform in
[L_MIN, L_MAX] = [16, 32]) from the full task vocabulary (PAD excluded —
it is the LM-loss ignore_index; everything else eligible so no embedding
is cold), then repeat the prefix cyclically to fill the 64-token window.
The repeat offset now varies per sequence, so locating the previous
occurrence of the current token requires CONTENT matching — the canonical
induction computation. Every position t >= L is a supervised copy
opportunity under full-sequence LM loss. No task episodes.

Verification (Step 2, thresholds pre-declared, identical
to Check 3) probes only SECOND-PASS positions t in [L, min(2L-1, 62)],
where the current token has occurred exactly ONCE before (at t-L, prefix
tokens are unique) so the previous-occurrence successor is unambiguous:
  behavioural — full-vocab argmax accuracy at targets t+1 must exceed 0.9
    on held-out sequences;
  mechanistic — some head's mean attention mass from t to t-L+1 must be
    >= 0.25 AND >= 5x the uniform-over-attendable-context baseline.
"""

import math

import numpy as np
import torch

from trained.data import VOCAB
from trained.model import D_MODEL, N_HEADS

IND_LEN = 64
L_MIN, L_MAX = 16, 32

# pre-declared thresholds (spec Step 2; identical to Check 3)
BEHAV_THRESH = 0.9
EDGE_ABS, EDGE_REL = 0.25, 5.0


def _probe_positions(L):
    """Second-pass from-positions for prefix length L (unambiguous single
    prior occurrence; target position t+1 <= 63)."""
    return np.arange(L, min(2 * L - 1, IND_LEN - 2) + 1)


N_PROBE_MAX = max(len(_probe_positions(L)) for L in range(L_MIN, L_MAX + 1))

# uniform-over-attendable-context baseline: 1/(t+1) averaged over all probe
# positions of all L (a constant of the corpus design, fixed in advance)
UNIFORM_BASELINE = float(
    np.mean(
        [1.0 / (t + 1) for L in range(L_MIN, L_MAX + 1) for t in _probe_positions(L)]
    )
)


def sample_batch(rng, batch):
    """-> (seqs (batch, 64) int64, Ls (batch,) int64). Prefix of L unique
    random non-PAD tokens over the full vocab, tiled cyclically to 64."""
    r = rng.random((batch, VOCAB - 1))
    prefixes = np.argsort(r, axis=1)[:, :L_MAX] + 1  # ids 1..VOCAB-1
    Ls = rng.integers(L_MIN, L_MAX + 1, size=batch)
    idx = np.arange(IND_LEN)[None, :] % Ls[:, None]  # (batch, 64) cyclic
    seqs = np.take_along_axis(prefixes, idx, axis=1).astype(np.int64)
    return seqs, Ls.astype(np.int64)


def probe_index_arrays(Ls):
    """Padded probe indices for a batch: (from_idx, to_idx, valid_mask),
    each (batch, N_PROBE_MAX). from t (2nd pass), to t-L+1 (prev-occurrence
    successor), target token position t+1."""
    B = len(Ls)
    from_idx = np.zeros((B, N_PROBE_MAX), dtype=np.int64)
    to_idx = np.zeros((B, N_PROBE_MAX), dtype=np.int64)
    valid = np.zeros((B, N_PROBE_MAX), dtype=bool)
    for i, L in enumerate(Ls):
        p = _probe_positions(int(L))
        from_idx[i, : len(p)] = p
        to_idx[i, : len(p)] = p - int(L) + 1
        valid[i, : len(p)] = True
    return from_idx, to_idx, valid


def build_eval_set(seed, n=1024):
    """Held-out sequences: rng stream (seed, 55501), disjoint from any
    training stream ((seed, 77) etc.)."""
    rng = np.random.default_rng((seed, 55501))
    seqs, Ls = sample_batch(rng, n)
    return dict(seqs=seqs, Ls=Ls, probes=probe_index_arrays(Ls))


@torch.no_grad()
def copy_accuracy(model, ev, device, batch=256):
    """Behavioural check: full-vocab argmax accuracy at unambiguous copy
    positions (second pass), no candidate masking."""
    model.eval()
    seqs = ev["seqs"]
    from_idx, _, valid = ev["probes"]
    correct, total = 0, 0
    for i in range(0, len(seqs), batch):
        t = torch.as_tensor(seqs[i : i + batch], device=device)
        fi = torch.as_tensor(from_idx[i : i + batch], device=device)
        vm = torch.as_tensor(valid[i : i + batch], device=device)
        logits = model(t)
        preds = logits.gather(
            1, fi.unsqueeze(-1).expand(-1, -1, logits.shape[-1])
        ).argmax(-1)
        targets = t.gather(1, fi + 1)
        ok = (preds == targets) & vm
        correct += int(ok.sum().item())
        total += int(vm.sum().item())
    return correct / total


@torch.no_grad()
def induction_attention(model, ev, device, batch=128, limit=None):
    """Mechanistic check: per-(layer, head) mean attention mass from each
    probe position t to t-L+1. Returns (n_layers, N_HEADS) array."""
    model.eval()
    seqs = ev["seqs"] if limit is None else ev["seqs"][:limit]
    from_idx, to_idx, valid = ev["probes"]
    if limit is not None:
        from_idx, to_idx, valid = from_idx[:limit], to_idx[:limit], valid[:limit]
    sums = np.zeros((model.n_layers, N_HEADS))
    n_probes = 0
    for i in range(0, len(seqs), batch):
        toks = torch.as_tensor(seqs[i : i + batch], device=device)
        fi = torch.as_tensor(from_idx[i : i + batch], device=device)
        ti = torch.as_tensor(to_idx[i : i + batch], device=device)
        vm = torch.as_tensor(valid[i : i + batch], device=device, dtype=torch.float32)
        B, T = toks.shape
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=device), 1)
        x = model.emb[toks] + model.pos[:T]
        rows = torch.arange(B, device=device)[:, None]
        for li, block in enumerate(model.blocks):
            h = block.ln1(x)
            q, k, v = block.qkv(h).chunk(3, dim=-1)
            hd = D_MODEL // N_HEADS
            q, k, v = (t.view(B, T, N_HEADS, hd).transpose(1, 2) for t in (q, k, v))
            att = (q @ k.transpose(-2, -1)) / math.sqrt(hd)
            att = att.masked_fill(mask, float("-inf")).softmax(dim=-1)
            # (B, P, H): mass at (from, to) per probe per head
            m = att.permute(0, 2, 3, 1)[rows, fi, ti]  # (B, P, H)
            sums[li] += (m * vm[..., None]).sum(dim=(0, 1)).cpu().numpy()
            x = x + block.proj((att @ v).transpose(1, 2).reshape(B, T, D_MODEL))
            x = x + block.mlp(block.ln2(x))
        n_probes += int(vm.sum().item())
    return sums / n_probes


def verify_induction(model, ev, device):
    """Full Step 2 verification. 'pass' requires BOTH checks (a seed
    failing either does not advance)."""
    acc = copy_accuracy(model, ev, device)
    tab = induction_attention(model, ev, device)
    li, h = np.unravel_index(tab.argmax(), tab.shape)
    mx = float(tab.max())
    behav = bool(acc > BEHAV_THRESH)
    mech = bool(mx >= EDGE_ABS and mx >= EDGE_REL * UNIFORM_BASELINE)
    return dict(
        copy_acc=float(acc),
        behavioural_pass=behav,
        head_table=[[float(v) for v in row] for row in tab],
        max_mass=mx,
        max_head=[int(li), int(h)],
        x_baseline=float(mx / UNIFORM_BASELINE),
        baseline=float(UNIFORM_BASELINE),
        mechanistic_pass=mech,
        thresholds=dict(behavioural=BEHAV_THRESH, abs=EDGE_ABS, rel=EDGE_REL),
        n_eval_seqs=int(len(ev["seqs"])),
        **{"pass": bool(behav and mech)}
    )

"""Positional-shortcut probe on the Phase 5 pretrained checkpoint.

The Step 1 corpus (v1) repeats at a FIXED offset of 32. With learned
absolute positions a layer-0 head can implement 'attend to t-31'
positionally. Probe: evaluate copy behaviour on repeated sequences with a
DIFFERENT period L (tokens repeat with period L != 32). A content-based
induction head still copies; a positional t-31 head fails.
"""

import pathlib
import math
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "src"))
import numpy as np
import torch

from trained.data import VOCAB
from trained.model import TinyTransformer, D_MODEL, N_HEADS

IND_LEN = 64


def periodic_batch(rng, batch, period):
    """Unique random non-PAD tokens 0..period-1, tiled to length 64."""
    r = rng.random((batch, VOCAB - 1))
    base = np.argsort(r, axis=1)[:, :period] + 1
    reps = int(np.ceil(IND_LEN / period))
    return np.tile(base, (1, reps))[:, :IND_LEN].astype(np.int64)


@torch.no_grad()
def probe(model, seqs, period):
    """Copy acc at positions >= period (target = token period back +0),
    and mean max-head attention mass from t to t-period+1."""
    toks = torch.as_tensor(seqs)
    B, T = toks.shape
    from_pos = np.arange(period, T - 1)  # predict positions from_pos -> from_pos+1
    logits = model(toks)
    preds = logits[:, from_pos].argmax(dim=-1)
    targets = toks[:, from_pos + 1]
    acc = float((preds == targets).float().mean())

    mask = torch.triu(torch.ones(T, T, dtype=torch.bool), 1)
    x = model.emb[toks] + model.pos[:T]
    to_pos = from_pos - (period - 1)
    fi = torch.as_tensor(from_pos)
    ti = torch.as_tensor(to_pos)
    best = np.zeros((model.n_layers, N_HEADS))
    for li, block in enumerate(model.blocks):
        h = block.ln1(x)
        q, k, v = block.qkv(h).chunk(3, dim=-1)
        hd = D_MODEL // N_HEADS
        q, k, v = (t.view(B, T, N_HEADS, hd).transpose(1, 2) for t in (q, k, v))
        att = (q @ k.transpose(-2, -1)) / math.sqrt(hd)
        att = att.masked_fill(mask, float("-inf")).softmax(dim=-1)
        m = (
            att[:, :, fi, :]
            .gather(3, ti.view(1, 1, -1, 1).expand(B, N_HEADS, -1, 1))
            .squeeze(3)
        )
        best[li] = m.mean(dim=(0, 2)).numpy()
        x = x + block.proj((att @ v).transpose(1, 2).reshape(B, T, D_MODEL))
        x = x + block.mlp(block.ln2(x))
    return acc, best


ck = torch.load("/tmp/p5_seed0.pt", map_location="cpu")
model = TinyTransformer(seed=0, n_layers=4)
model.load_state_dict(ck["model"])
model.eval()
print(f"checkpoint step {ck['step']}")

rng = np.random.default_rng(999)
for period in (32, 21, 16, 11):
    seqs = periodic_batch(rng, 256, period)
    acc, tab = probe(model, seqs, period)
    li, h = np.unravel_index(tab.argmax(), tab.shape)
    print(
        f"period {period:>2}: copy_acc {acc:.3f}  max ind-edge mass "
        f"{tab.max():.3f} (L{li} h{h})"
    )

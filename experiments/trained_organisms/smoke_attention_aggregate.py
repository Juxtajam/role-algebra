"""Local smoke test for itemA_aggregate_attention.py core logic.

Runs the same attention_capture / edge_stats code path on a fresh
(untrained) TinyTransformer with N_EP=8, and cross-checks the direct
per-head table and one composed pair against a brute-force loop.
"""

import pathlib
import math, sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "src"))
import numpy as np
import torch
from trained import data as D
from trained.model import TinyTransformer, masked_answer_preds, D_MODEL, N_HEADS

N_EP = 8


def attention_capture(model, toks_batch):
    toks = torch.as_tensor(toks_batch)
    B, T = toks.shape
    mask = torch.triu(torch.ones(T, T, dtype=torch.bool), 1)
    x = model.emb[toks] + model.pos[:T]
    atts = []
    with torch.no_grad():
        for block in model.blocks:
            h = block.ln1(x)
            q, k, v = block.qkv(h).chunk(3, dim=-1)
            hd = D_MODEL // N_HEADS
            q, k, v = (t.view(B, T, N_HEADS, hd).transpose(1, 2) for t in (q, k, v))
            att = (q @ k.transpose(-2, -1)) / math.sqrt(hd)
            att = att.masked_fill(mask, float("-inf")).softmax(dim=-1)
            atts.append(att)
            x = x + block.proj((att @ v).transpose(1, 2).reshape(B, T, D_MODEL))
            x = x + block.mlp(block.ln2(x))
    return atts


rng = np.random.default_rng(123)
qt = D.A_PS
toks_l, argpos, matchpos, qmarkpos, apos_l, ans_l, cands_l = [], [], [], [], [], [], []
for _ in range(N_EP):
    b = D.sample_base("T1", rng, force_qtok=qt)
    g = D.PERMS[int(rng.integers(len(D.PERMS)))]
    toks, apos, ans, cands = D.render(b, g)
    seq = toks.tolist()
    qpos = seq.index(qt)
    arg_id = seq[qpos + 1]
    mpos, j = None, 1
    while seq[j] in (D.HAS, D.CARRY, D.GUARD):
        if seq[j] == D.HAS and seq[j + 1] == arg_id:
            mpos = j + 1
        j += 4
    assert mpos is not None
    toks_l.append(toks)
    argpos.append(qpos + 1)
    matchpos.append(mpos)
    qmarkpos.append(apos)
    apos_l.append(apos)
    ans_l.append(ans)
    cands_l.append(cands)

model = TinyTransformer(seed=0, n_layers=4)
model.eval()
toks_b = np.stack(toks_l)
atts = attention_capture(model, toks_b)
B = N_EP
rows = torch.arange(B)
fp = torch.as_tensor(argpos)
tp = torch.as_tensor(matchpos)

# --- direct table: vectorised vs brute force ---
tab = np.zeros((4, N_HEADS))
for li in range(4):
    tab[li] = atts[li][rows, :, fp, tp].mean(0).numpy()
tab_bf = np.zeros((4, N_HEADS))
for li in range(4):
    for h in range(N_HEADS):
        tab_bf[li, h] = np.mean(
            [float(atts[li][i, h, argpos[i], matchpos[i]]) for i in range(B)]
        )
assert np.allclose(tab, tab_bf, atol=1e-7), "direct table mismatch"
print("direct table OK; aggregate", tab.sum())

# --- composed pair (l=1 -> l=2): vectorised vs brute force ---
l = 1
M_hi = atts[l + 1].mean(1)
M_lo = atts[l].mean(1)
row = M_hi[rows, fp, :]
col = M_lo[rows, :, tp]
comp_ep = (row * col).sum(1)
comp_bf = []
for i in range(B):
    s = 0.0
    for j in range(48):
        a_hi = float(atts[l + 1][i, :, argpos[i], j].mean())
        a_lo = float(atts[l][i, :, j, matchpos[i]].mean())
        s += a_hi * a_lo
    comp_bf.append(s)
assert np.allclose(comp_ep.numpy(), comp_bf, atol=1e-6), "composed mismatch"
print("composed head-mean OK:", float(comp_ep.mean()))

# --- head-pair einsum vs brute force ---
R = atts[l + 1][rows, :, fp, :]
C = atts[l][rows, :, :, tp]
pairmat = torch.einsum("bht,bgt->bhg", R, C).mean(0).numpy()
bf = np.zeros((N_HEADS, N_HEADS))
for h1 in range(N_HEADS):
    for h2 in range(N_HEADS):
        vals = []
        for i in range(B):
            s = sum(
                float(atts[l + 1][i, h1, argpos[i], j])
                * float(atts[l][i, h2, j, matchpos[i]])
                for j in range(48)
            )
            vals.append(s)
        bf[h1, h2] = np.mean(vals)
assert np.allclose(pairmat, bf, atol=1e-6), "head-pair mismatch"
print("head-pair einsum OK; max", pairmat.max())

# --- causal sanity: mass to positions after from_pos must be 0 ---
i = 0
assert float(atts[0][i, :, argpos[i], argpos[i] + 1 :].sum()) == 0.0
print("causal mask OK")

# --- masked preds runs ---
preds = masked_answer_preds(
    model, toks_b, np.array(apos_l), np.array([list(c) for c in cands_l])
)
acc = float((preds == np.array(ans_l)).mean())
print(f"untrained masked acc {acc:.3f} (expect ~1/3)")
print("SMOKE PASS")

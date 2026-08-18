import pathlib
import math
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "src"))
import numpy as np
import torch

from trained import data as D
from trained import induction as I
from trained.model import TinyTransformer, lm_loss, make_optimizer, D_MODEL, N_HEADS

rng = np.random.default_rng(0)
seqs, Ls = I.sample_batch(rng, 512)
assert seqs.shape == (512, 64) and seqs.dtype == np.int64
assert (seqs != D.PAD).all() and seqs.min() >= 1 and seqs.max() < D.VOCAB
assert Ls.min() >= I.L_MIN and Ls.max() <= I.L_MAX
for s, L in zip(seqs[:100], Ls[:100]):
    assert len(set(s[:L].tolist())) == L, "prefix not unique"
    assert (s == np.tile(s[:L], 5)[:64]).all(), "not cyclic"
assert len(np.unique(Ls)) == I.L_MAX - I.L_MIN + 1
big, _ = I.sample_batch(rng, 4000)
assert len(np.unique(big)) == D.VOCAB - 1, "vocab coverage incomplete"
# probe positions: second pass only, unambiguous
fi, ti, vm = I.probe_index_arrays(Ls)
for i, L in enumerate(Ls[:100]):
    p = fi[i][vm[i]]
    assert p.min() == L and p.max() == min(2 * L - 1, 62)
    assert ((ti[i][vm[i]] == p - L + 1)).all()
    # the token at each probe position occurred exactly once before
    for t in p:
        prior = (seqs[i, :t] == seqs[i, t]).sum()
        assert prior == 1, (i, t, prior)
        # and target t+1 equals the successor of that occurrence
        assert seqs[i, t + 1] == seqs[i, t - L + 1]
print(
    f"corpus v2 OK (baseline={I.UNIFORM_BASELINE:.4f}, " f"N_PROBE_MAX={I.N_PROBE_MAX})"
)

m = TinyTransformer(seed=0, n_layers=4)
ev = I.build_eval_set(0, n=64)
acc0 = I.copy_accuracy(m, ev, "cpu")
tab0 = I.induction_attention(m, ev, "cpu")
print(
    f"untrained: copy_acc={acc0:.4f} max_mass={tab0.max():.4f} "
    f"({tab0.max()/I.UNIFORM_BASELINE:.1f}x baseline)"
)
assert acc0 < 0.05 and tab0.max() < 0.15

# brute-force cross-check of induction_attention on 4 sequences
sub = dict(
    seqs=ev["seqs"][:4], Ls=ev["Ls"][:4], probes=tuple(a[:4] for a in ev["probes"])
)
toks = torch.as_tensor(sub["seqs"])
B, T = toks.shape
mask = torch.triu(torch.ones(T, T, dtype=torch.bool), 1)
x = m.emb[toks] + m.pos[:T]
fi4, ti4, vm4 = sub["probes"]
sums = np.zeros((4, N_HEADS))
tot = vm4.sum()
with torch.no_grad():
    for li, block in enumerate(m.blocks):
        h = block.ln1(x)
        q, k, v = block.qkv(h).chunk(3, dim=-1)
        hd = D_MODEL // N_HEADS
        q, k, v = (t.view(B, T, N_HEADS, hd).transpose(1, 2) for t in (q, k, v))
        att = (q @ k.transpose(-2, -1)) / math.sqrt(hd)
        att = att.masked_fill(mask, float("-inf")).softmax(dim=-1)
        for hh in range(N_HEADS):
            s = 0.0
            for i in range(B):
                for j in range(fi4.shape[1]):
                    if vm4[i, j]:
                        s += float(att[i, hh, fi4[i, j], ti4[i, j]])
            sums[li, hh] = s / tot
        x = x + block.proj((att @ v).transpose(1, 2).reshape(B, T, D_MODEL))
        x = x + block.mlp(block.ln2(x))
tab_fast = I.induction_attention(m, sub, "cpu")
assert np.allclose(sums, tab_fast, atol=1e-6), np.abs(sums - tab_fast).max()
print("induction_attention matches brute-force loop")

# CPU training probe: content-matching is required now, so this should be
# slower than v1's 100-step positional solve; train up to 1500 steps
opt = make_optimizer(m, lr=1e-3, weight_decay=0.01)
rng2 = np.random.default_rng((0, 77))
m.train()
for step in range(1500):
    s, _ = I.sample_batch(rng2, 64)
    t = torch.as_tensor(s)
    loss = lm_loss(m(t), t)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
    if (step + 1) % 300 == 0:
        a = I.copy_accuracy(m, ev, "cpu")
        tb = I.induction_attention(m, ev, "cpu")
        li, h = np.unravel_index(tb.argmax(), tb.shape)
        print(
            f"step {step+1}: loss {loss.item():.3f} copy_acc {a:.3f} "
            f"max_mass {tb.max():.3f} (L{li} h{h})"
        )
        m.train()

# anti-positional check: novel period L=13 (outside the training range,
# and 64 % 13 != 0) — a content-based head should still copy at the
# unambiguous second-pass positions
r3 = np.random.default_rng(999)
pref = np.argsort(r3.random((256, D.VOCAB - 1)), axis=1)[:, :13] + 1
seq13 = np.tile(pref, (1, 5))[:, :64].astype(np.int64)
L13 = np.full(256, 13, dtype=np.int64)
p13 = np.arange(13, 25)
fi3 = np.tile(p13, (256, 1))
ti3 = fi3 - 12
vm3 = np.ones_like(fi3, dtype=bool)
ev13 = dict(seqs=seq13, Ls=L13, probes=(fi3, ti3, vm3))
a13 = I.copy_accuracy(m, ev13, "cpu")
t13 = I.induction_attention(m, ev13, "cpu")
print(f"novel period 13: copy_acc {a13:.3f} max_mass {t13.max():.3f}")
print("smoke v2 OK")

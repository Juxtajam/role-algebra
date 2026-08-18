"""Local CPU smoke test of the Phase 4b freeze fix (fast, small steps).

Checks, on CPU with tiny batches:
 1. new init RNG-identical to legacy init (reconstruct legacy inline);
 2. legacy checkpoint state_dict loads (simulated legacy dict with 'emb');
 3. make_optimizer: two param groups, emb_names in the wd=0.0 group;
 4. 30 training steps: frozen name rows bit-exact; with fit-pool rows opened,
    fit rows move, other name rows bit-exact;
 5. legacy-style optimizer shrinks name-row norms (fault reproduction).
"""

import sys

sys.path.insert(0, "src")

import numpy as np
import torch

from trained import data as D
from trained.model import (
    TinyTransformer,
    lm_loss,
    make_optimizer,
    verify_tied_names,
    D_MODEL,
)

# 1. init identity vs legacy construction
torch.manual_seed(3)
full = torch.randn(D.VOCAB, D_MODEL) * 0.02
target = full[: D.NAME0].norm(dim=1).mean()
names = torch.randn(D.VOCAB - D.NAME0, D_MODEL)
full[D.NAME0 :] = names / names.norm(dim=1, keepdim=True) * target
m = TinyTransformer(seed=3, n_layers=4)
assert torch.equal(m.emb, full), "init not RNG-identical to legacy"
print("1. init bit-identical to legacy: OK")

# 2. legacy state dict load
legacy_sd = {k: v.clone() for k, v in m.state_dict().items()}
legacy_sd["emb"] = torch.cat([legacy_sd.pop("emb_main"), legacy_sd.pop("emb_names")])
legacy_sd["grad_mask"] = torch.zeros(D.VOCAB, 1)
legacy_sd["grad_mask"][: D.NAME0] = 1.0
del legacy_sd["names_grad_mask"]
m2 = TinyTransformer(seed=0, n_layers=4)
r = m2.load_state_dict(legacy_sd, strict=True)
assert not r.missing_keys and not r.unexpected_keys, r
assert torch.equal(m2.emb, m.emb)
# legacy grad_mask [NAME0:] was 0 -> names_grad_mask all zero
assert m2.names_grad_mask.sum() == 0
verify_tied_names(m2)
print("2. legacy checkpoint compat + tied round-trip: OK")

# 3. optimizer groups
opt = make_optimizer(m, lr=1e-3, weight_decay=0.01)
wds = [pg["weight_decay"] for pg in opt.param_groups]
assert wds == [0.01, 0.0], wds
assert opt.param_groups[1]["params"][0] is m.emb_names
n_params = sum(p.numel() for pg in opt.param_groups for p in pg["params"])
assert n_params == sum(p.numel() for p in m.parameters())
print("3. param groups: OK", wds)


# 4. short training, frozen vs opened rows
def steps(model, opt, n=30):
    rng = np.random.default_rng((0, 77))
    pool = [D.sample_base("T1", rng) for _ in range(256)]
    for _ in range(n):
        idx = rng.integers(0, len(pool), size=32)
        gs = [D.PERMS[i] for i in rng.integers(0, len(D.PERMS), size=32)]
        toks, _, _, _ = D.render_batch([pool[i] for i in idx], gs)
        loss = lm_loss(model(torch.as_tensor(toks)), torch.as_tensor(toks))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()


m3 = TinyTransformer(seed=0, n_layers=4)
opt3 = make_optimizer(m3)
names0 = m3.emb_names.detach().clone()
steps(m3, opt3)
assert torch.equal(m3.emb_names.detach(), names0), "frozen rows moved!"
print("4a. all-frozen: name rows bit-exact after 30 steps: OK")

m4 = TinyTransformer(seed=0, n_layers=4)
with torch.no_grad():
    m4.names_grad_mask[:24] = 1.0  # Item 3: open fit-pool rows
opt4 = make_optimizer(m4)
names0 = m4.emb_names.detach().clone()
steps(m4, opt4)
fit_moved = not torch.equal(m4.emb_names.detach()[:24], names0[:24])
rest_exact = torch.equal(m4.emb_names.detach()[24:], names0[24:])
assert fit_moved and rest_exact, (fit_moved, rest_exact)
print("4b. fit rows opened: fit rows move, rows 24+ bit-exact: OK")

# 5. legacy fault reproduction
m5 = TinyTransformer(seed=0, n_layers=4)
opt5 = torch.optim.AdamW(m5.parameters(), lr=1e-3, weight_decay=0.01)
names0 = m5.emb_names.detach().clone()
steps(m5, opt5)
ratio = m5.emb_names.detach().norm(dim=1) / names0.norm(dim=1)
exp = (1 - 1e-3 * 0.01) ** 30
print(
    f"5. legacy optimizer shrink after 30 steps: mean ratio {ratio.mean():.8f} "
    f"(analytic {exp:.8f}) std {ratio.std():.2e}"
)
assert ratio.mean() < 1.0 and abs(ratio.mean() - exp) < 1e-5

print("ALL SMOKE CHECKS PASSED")

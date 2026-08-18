"""Stage 2 model (spec v3, "Task and model" / architecture).

4 layers, d_model 128, 4 heads, d_ff 512, pre-LN, fp32, learned positional
embeddings, max seq len 64. Name embeddings are frozen, tied to the
unembedding, randomly initialised and normalised — transfer names are never
trained but can still be emitted through the tied unembedding. The
round-trip of every name embedding through the unembedding is verified
before training.

PHASE 4b FREEZE FIX (blocker, 2026-08-07): the old implementation kept
one emb tensor and zeroed name-row gradients with a hook, but AdamW's
DECOUPLED weight decay multiplies the whole parameter tensor by
(1 - lr*wd) every step regardless of the gradient, so "frozen" name rows
shrank uniformly ~0.80x in norm over 50k steps (Item 1 finding). The
embedding is now split into two parameters — emb_main (rows < NAME0) and
emb_names (all name rows) — and make_optimizer() puts emb_names in an
explicit parameter group with weight_decay=0.0. Frozen name rows are now
bit-exact constant across training (Adam moments stay zero under a zeroed
gradient, and no decay applies). The RNG draw order at init is identical
to the old code, so initial weights are bit-identical; a load-state-dict
pre-hook splits the legacy 'emb' key so old checkpoints still load.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from trained.data import NAME0, PAD, VOCAB

D_MODEL, N_LAYERS, N_HEADS, D_FF, MAX_LEN = 128, 4, 4, 512, 64


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(D_MODEL)
        self.ln2 = nn.LayerNorm(D_MODEL)
        self.qkv = nn.Linear(D_MODEL, 3 * D_MODEL)
        self.proj = nn.Linear(D_MODEL, D_MODEL)
        self.mlp = nn.Sequential(
            nn.Linear(D_MODEL, D_FF), nn.GELU(), nn.Linear(D_FF, D_MODEL)
        )

    def forward(self, x, mask):
        B, T, _ = x.shape
        q, k, v = self.qkv(self.ln1(x)).chunk(3, dim=-1)
        hd = D_MODEL // N_HEADS
        q, k, v = (t.view(B, T, N_HEADS, hd).transpose(1, 2) for t in (q, k, v))
        att = (q @ k.transpose(-2, -1)) / math.sqrt(hd)
        att = att.masked_fill(mask, float("-inf")).softmax(dim=-1)
        x = x + self.proj((att @ v).transpose(1, 2).reshape(B, T, D_MODEL))
        return x + self.mlp(self.ln2(x))


class TinyTransformer(nn.Module):
    def __init__(self, seed=0, n_layers=N_LAYERS):
        super().__init__()
        self.n_layers = n_layers
        torch.manual_seed(seed)
        # identical RNG consumption to the legacy single-tensor init:
        full = torch.randn(VOCAB, D_MODEL) * 0.02
        # name rows: random, normalised to the typical embedding norm
        target = full[:NAME0].norm(dim=1).mean()
        names = torch.randn(VOCAB - NAME0, D_MODEL)
        names = names / names.norm(dim=1, keepdim=True) * target
        # split parameters: name rows live in their own tensor so they can be
        # excluded from weight decay via an explicit optimizer group
        self.emb_main = nn.Parameter(full[:NAME0].clone())
        self.emb_names = nn.Parameter(names)
        # freeze name rows via a gradient mask (ties survive: logits use emb.T);
        # rows can be selectively unfrozen (Item 3 opens the fit-pool rows)
        self.register_buffer("names_grad_mask", torch.zeros(VOCAB - NAME0, 1))
        self.emb_names.register_hook(lambda g: g * self.names_grad_mask)
        self._register_load_state_dict_pre_hook(self._legacy_emb_compat)
        self.pos = nn.Parameter(torch.randn(MAX_LEN, D_MODEL) * 0.02)
        self.blocks = nn.ModuleList([Block() for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(D_MODEL)

    @property
    def emb(self):
        """Full (VOCAB, D_MODEL) embedding; the tied unembedding is emb.T.
        Differentiable: grads flow into both underlying parameters."""
        return torch.cat([self.emb_main, self.emb_names], dim=0)

    def _legacy_emb_compat(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        """Split the legacy single 'emb' tensor (and full-vocab grad_mask)
        from pre-Phase-4b checkpoints into the new parameter layout."""
        k = prefix + "emb"
        if k in state_dict:
            full = state_dict.pop(k)
            state_dict[prefix + "emb_main"] = full[:NAME0]
            state_dict[prefix + "emb_names"] = full[NAME0:]
        gm = prefix + "grad_mask"
        if gm in state_dict:
            state_dict[prefix + "names_grad_mask"] = state_dict.pop(gm)[NAME0:]

    def forward(self, tokens, capture=False, patch=None):
        """tokens (B, T). capture -> also return resid_post per layer.
        patch = (layer, positions(B,), vectors(B, d)): replace resid_post of
        that layer at those positions, then run the remaining layers."""
        B, T = tokens.shape
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=tokens.device), 1)
        emb = self.emb
        x = emb[tokens] + self.pos[:T]
        resids = []
        rows = torch.arange(B, device=tokens.device)
        for li, block in enumerate(self.blocks):
            x = block(x, mask)
            if patch is not None and patch[0] == li:
                x = x.clone()
                x[rows, patch[1]] = patch[2]
            if capture:
                resids.append(x)
        logits = self.ln_f(x) @ emb.T
        return (logits, resids) if capture else logits


def make_optimizer(model, lr=1e-3, weight_decay=0.01):
    """AdamW with the required explicit parameter groups (Phase 4b):
    emb_names (ALL name rows, frozen or trainable) gets weight_decay=0.0 so
    frozen name rows are exactly constant across training, and trainable
    fit-pool rows (Item 3) differ from the frozen control in gradient flow
    ONLY, not in decay. Everything else keeps the standing value."""
    others = [p for n, p in model.named_parameters() if n != "emb_names"]
    return torch.optim.AdamW(
        [
            dict(params=others, weight_decay=weight_decay),
            dict(params=[model.emb_names], weight_decay=0.0),
        ],
        lr=lr,
        weight_decay=weight_decay,
    )


def verify_tied_names(model, n_check=1000):
    """Spec: verify before training that passing a name embedding through the
    tied unembedding recovers the correct token."""
    with torch.no_grad():
        ids = torch.arange(NAME0, VOCAB, device=model.emb.device)[:n_check]
        logits = model.emb[ids] @ model.emb.T
        ok = (logits.argmax(dim=1) == ids).float().mean().item()
    assert ok == 1.0, f"tied-name round-trip failed: {ok:.4f}"
    return ok


def lm_loss(logits, tokens):
    """Full-sequence LM loss (answer-token accuracy is reported separately)."""
    return F.cross_entropy(
        logits[:, :-1].reshape(-1, VOCAB), tokens[:, 1:].reshape(-1), ignore_index=PAD
    )


@torch.no_grad()
def masked_answer_preds(model, tokens, answer_pos, candidates, batch=512):
    """Episode-local candidate masking: logits restricted to the k names (or
    symbols) present in the episode; returns the predicted token."""
    model.eval()
    device = next(model.parameters()).device
    out = []
    for i in range(0, len(tokens), batch):
        t = torch.as_tensor(tokens[i : i + batch], device=device)
        ap = torch.as_tensor(answer_pos[i : i + batch], device=device)
        cd = torch.as_tensor(candidates[i : i + batch], device=device)
        logits = model(t)
        rows = torch.arange(len(t), device=device)
        sel = logits[rows, ap]  # (b, V) at the answer position
        cand_logits = sel.gather(1, cd)  # (b, k)
        out.append(
            cd.gather(1, cand_logits.argmax(1, keepdim=True)).squeeze(1).cpu().numpy()
        )
    import numpy as np

    return np.concatenate(out)

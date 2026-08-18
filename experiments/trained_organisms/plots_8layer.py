"""Phase 6 Item B figures.

1. figures/phase6_composed_vs_control.png — composed held-out trajectory,
   8-layer (Item B) vs 4-layer Phase 5 control, same axes, per seed.
2. figures/phase6_joint_trajectories.png — 8L per-seed task metrics +
   induction survival, same layout as the Phase 5 figure.
3. figures/phase6_pretraining8.png — 8L pretraining trajectories.
Run after results/stage2/induction8/** has been pulled locally.
"""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

b8 = "results/trained_organisms/induction8"
b4 = "results/trained_organisms/induction"

# --- 1. composed held-out: 8L vs 4L control ---------------------------------
fig, axes = plt.subplots(3, 1, figsize=(11, 12), sharex=True)
for s, ax in zip((0, 1, 2), axes):
    t8 = json.load(open(f"{b8}/finetune/seed{s}/trajectory.json"))
    t4 = json.load(open(f"{b4}/finetune/seed{s}/trajectory.json"))
    ax.plot(
        [h["step"] for h in t8],
        [h["held_episode"] for h in t8],
        color="tab:green",
        lw=1.6,
        label="composed held (8L, Item B)",
    )
    ax.plot(
        [h["step"] for h in t4],
        [h["held_episode"] for h in t4],
        color="tab:green",
        lw=1.0,
        ls="--",
        alpha=0.7,
        label="composed held (4L, Phase 5 control)",
    )
    ax.plot(
        [h["step"] for h in t8],
        [h["aux_min"] for h in t8],
        color="tab:blue",
        lw=1.0,
        label="aux_min held (8L)",
    )
    ax.plot(
        [h["step"] for h in t4],
        [h["aux_min"] for h in t4],
        color="tab:blue",
        lw=0.8,
        ls="--",
        alpha=0.7,
        label="aux_min held (4L control)",
    )
    ax.plot(
        [h["step"] for h in t8],
        [h["train_composed"] for h in t8],
        color="olive",
        lw=0.8,
        alpha=0.7,
        label="train composed (8L)",
    )
    ax.axhline(1 / 3, color="gray", ls=":", lw=0.8)
    ax.axhline(0.95, color="black", ls=":", lw=0.8)
    ax.axhline(0.5, color="tab:red", ls=":", lw=0.8)
    ax.set_ylim(-0.02, 1.08)
    ax.set_ylabel(f"seed{s}")
    ax.grid(alpha=0.25)
    if s == 0:
        ax.legend(ncol=3, fontsize=8, loc="center right")
axes[-1].set_xlabel("fine-tuning step (standard T1, hard 50k, refresh on)")
fig.suptitle(
    "Item B: composed held-out, 8-layer vs 4-layer Phase 5 control\n"
    "(gray = chance 1/3; red = 0.5 diagnostic trigger; black = 0.95 branch)",
    fontsize=11,
)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig("figures/phase6_composed_vs_control.png", dpi=150)
print("wrote figures/phase6_composed_vs_control.png")

# --- 2. joint trajectories (8L) ----------------------------------------------
fig, axes = plt.subplots(3, 1, figsize=(11, 13), sharex=True)
for s, ax in zip((0, 1, 2), axes):
    tr = json.load(open(f"{b8}/finetune/seed{s}/trajectory.json"))
    steps = [h["step"] for h in tr]
    for t, c in (("A_PS", "tab:blue"), ("A_SN", "tab:cyan"), ("A_NS", "tab:purple")):
        ax.plot(
            steps, [h["aux_held"][t] for h in tr], color=c, lw=1.2, label=f"held {t}"
        )
    ax.plot(
        steps,
        [h["held_episode"] for h in tr],
        color="tab:green",
        lw=1.2,
        label="held composed_P",
    )
    ax.plot(
        steps,
        [h["transfer_episode"] for h in tr],
        color="olive",
        lw=0.8,
        alpha=0.6,
        label="transfer composed",
    )
    ax.plot(
        steps,
        [h["ind_copy_acc"] for h in tr],
        color="tab:red",
        lw=1.6,
        label="induction copy acc (held)",
    )
    ax.plot(
        steps,
        [h["ind_max_mass"] for h in tr],
        color="tab:orange",
        lw=1.6,
        label="induction head mass (max)",
    )
    ax.plot(
        steps,
        [h["ind_pretrained_head_mass"] for h in tr],
        color="tab:orange",
        lw=1.0,
        ls="--",
        label="mass at pretrained head",
    )
    ax.axhline(1 / 3, color="gray", ls=":", lw=0.8)
    ax.axhline(0.95, color="black", ls=":", lw=0.8)
    ax.axhline(0.25, color="tab:orange", ls=":", lw=0.8)
    lost = next((h["step"] for h in tr if not h["ind_mechanistic"]), None)
    if lost:
        ax.axvline(lost, color="tab:red", ls="--", lw=1.0)
        ax.text(
            lost,
            1.02,
            f"induction lost @{lost}",
            color="tab:red",
            fontsize=8,
            ha="center",
        )
    ax.set_ylim(-0.02, 1.08)
    ax.set_ylabel(f"seed{s}")
    ax.grid(alpha=0.25)
    if s == 0:
        ax.legend(ncol=3, fontsize=8, loc="center right")
axes[-1].set_xlabel("fine-tuning step (standard T1, hard 50k, refresh on)")
fig.suptitle(
    "Item B (8 layers): task metrics and induction survival on the " "same time axis",
    fontsize=11,
)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig("figures/phase6_joint_trajectories.png", dpi=150)
print("wrote figures/phase6_joint_trajectories.png")

# --- 3. pretraining (8L) ------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 4.5))
for s, c in zip((0, 1, 2), ("tab:blue", "tab:green", "tab:red")):
    tr = json.load(open(f"{b8}/pretrain/seed{s}/trajectory.json"))
    steps = [h["step"] for h in tr]
    ax.plot(
        steps, [h["copy_acc"] for h in tr], color=c, lw=1.2, label=f"seed{s} copy acc"
    )
    ax.plot(
        steps,
        [h["max_ind_mass"] for h in tr],
        color=c,
        lw=1.2,
        ls="--",
        label=f"seed{s} head mass",
    )
ax.axhline(0.9, color="black", ls=":", lw=0.8)
ax.axhline(0.25, color="tab:orange", ls=":", lw=0.8)
ax.set_xlabel("pretraining step (induction corpus v2, 8 layers)")
ax.set_ylim(0, 1.05)
ax.legend(ncol=3, fontsize=8)
ax.grid(alpha=0.25)
ax.set_title("Item B step 1: 8-layer pretraining — copy accuracy and max head mass")
fig.tight_layout()
fig.savefig("figures/phase6_pretraining8.png", dpi=150)
print("wrote figures/phase6_pretraining8.png")

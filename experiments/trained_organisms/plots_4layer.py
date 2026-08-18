import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

base = "results/trained_organisms/induction"
fig, axes = plt.subplots(3, 1, figsize=(11, 13), sharex=True)
for s, ax in zip((0, 1, 2), axes):
    tr = json.load(open(f"{base}/finetune/seed{s}/trajectory.json"))
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
            lost, 1.02, f"head lost @{lost}", color="tab:red", fontsize=8, ha="center"
        )
    ax.set_ylim(-0.02, 1.08)
    ax.set_ylabel(f"seed{s}")
    ax.grid(alpha=0.25)
    if s == 0:
        ax.legend(ncol=3, fontsize=8, loc="center right")
axes[-1].set_xlabel("fine-tuning step (standard T1, hard 50k, pool refresh on)")
fig.suptitle(
    "Phase 5: task metrics and induction-head survival on the same "
    "time axis\n(gray dotted = chance 1/3; black dotted = 0.95 branch; "
    "orange dotted = 0.25 mass threshold)",
    fontsize=11,
)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig("figures/phase5_joint_trajectories.png", dpi=150)
print("wrote figures/phase5_joint_trajectories.png")

# pretraining trajectories (compact, one panel)
fig2, ax = plt.subplots(figsize=(9, 4.5))
for s, c in zip((0, 1, 2), ("tab:blue", "tab:green", "tab:red")):
    tr = json.load(open(f"{base}/pretrain/seed{s}/trajectory.json"))
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
ax.set_xlabel("pretraining step (induction corpus v2)")
ax.set_ylim(0, 1.05)
ax.legend(ncol=3, fontsize=8)
ax.grid(alpha=0.25)
ax.set_title("Phase 5 Step 2: pretraining — copy accuracy and max induction-head mass")
fig2.tight_layout()
fig2.savefig("figures/phase5_pretraining.png", dpi=150)
print("wrote figures/phase5_pretraining.png")

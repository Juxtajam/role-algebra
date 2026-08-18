"""7.4 — Trajectory plots from the saved trajectory JSONs (T0, T1; T2/T3 never trained)."""

import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

root = pathlib.Path(__file__).resolve().parents[2] / "results/trained_organisms"
figdir = pathlib.Path(__file__).resolve().parents[2] / "figures"
figdir.mkdir(exist_ok=True)

orgs = ["T0", "T1"]
fig, axes = plt.subplots(1, len(orgs), figsize=(11, 4), sharey=True)
for ax, org in zip(axes, orgs):
    for s in (0, 1, 2):
        p = root / org / f"seed{s}" / "trajectory.json"
        if not p.exists():
            continue
        h = json.load(open(p))
        steps = [r["step"] for r in h]
        ax.plot(steps, [r["held_episode"] for r in h], label=f"seed {s} held")
        ax.plot(
            steps,
            [r["transfer_episode"] for r in h],
            "--",
            alpha=0.6,
            label=f"seed {s} transfer",
        )
    ax.set_title(org + (" (leaked)" if org == "T0" else ""))
    ax.set_xlabel("step")
    ax.axhline(0.95, color="gray", lw=0.5)
    ax.axhline(1 / 3, color="red", lw=0.5, ls=":")
    ax.set_ylim(0, 1.05)
axes[0].set_ylabel("candidate-masked answer accuracy")
axes[0].legend(fontsize=7)
axes[1].legend(fontsize=7)
fig.suptitle(
    "Training trajectories — T2/T3 not trained (stopped early); red dotted = chance (1/3)"
)
fig.tight_layout()
fig.savefig(figdir / "trajectories.png", dpi=140)

# loss vs accuracy dissociation for T1
fig2, ax1 = plt.subplots(figsize=(7, 4))
ax2 = ax1.twinx()
for s in (0, 1, 2):
    h = json.load(open(root / "T1" / f"seed{s}" / "trajectory.json"))
    steps = [r["step"] for r in h]
    ax1.plot(steps, [r["loss"] for r in h], alpha=0.8, label=f"seed {s} loss")
    ax2.plot(steps, [r["held_episode"] for r in h], "--", alpha=0.6)
ax1.set_xlabel("step")
ax1.set_ylabel("full-sequence LM loss (solid)")
ax2.set_ylabel("composed held accuracy (dashed)")
ax2.set_ylim(0, 1.05)
ax2.axhline(1 / 3, color="red", lw=0.5, ls=":")
ax1.legend(fontsize=8)
ax1.set_title(
    "T1 loss/accuracy dissociation: loss falls, composed accuracy pinned at chance"
)
fig2.tight_layout()
fig2.savefig(figdir / "t1_loss_vs_accuracy.png", dpi=140)
print("wrote", list(str(p) for p in figdir.glob("*.png")))

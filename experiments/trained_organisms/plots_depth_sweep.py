"""Depth sweep figure: trajectories per depth + final accuracy bar."""

import json, pathlib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

root = pathlib.Path(__file__).resolve().parents[2] / "results/trained_organisms/depth_sweep"
figdir = pathlib.Path(__file__).resolve().parents[2] / "figures"

fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
for ax, L in zip(axes, (4, 6, 8)):
    for s in (0, 1, 2):
        h = json.load(open(root / f"T1_L{L}/seed{s}/trajectory.json"))
        ax.plot(
            [r["step"] for r in h],
            [r["held_episode"] for r in h],
            label=f"seed {s}",
            alpha=0.8,
        )
    ax.axhline(
        1 / 3, color="red", lw=0.6, ls=":", label="chance (1/3)" if L == 4 else None
    )
    ax.axhline(0.95, color="gray", lw=0.5)
    ax.set_title(f"T1, {L} layers")
    ax.set_xlabel("step")
    ax.set_ylim(0, 1.05)
axes[0].set_ylabel("composed held accuracy (candidate-masked)")
axes[0].legend(fontsize=8)
fig.suptitle(
    "Depth sweep — T1 at 4/6/8 layers, hard 50k steps: no seed at any depth left chance"
)
fig.tight_layout()
fig.savefig(figdir / "depth_sweep.png", dpi=140)
print("wrote", figdir / "depth_sweep.png")

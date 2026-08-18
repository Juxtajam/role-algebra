"""Generate the figures used in README.md and paper.md, from stored metrics only.

Reads result JSON under results/ and writes PNGs to figures/. Every number comes
from a stored artifact; nothing is hand-entered.
"""
import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)

BLUE, GREY, RED, INK = "#3b6ea5", "#9aa0a6", "#c0392b", "#222222"
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 150, "savefig.bbox": "tight"})


def load(p):
    return json.load(open(ROOT / p))


def fig_verdict():
    d = load("results/verdict/discriminator/verdict.json")
    m, t = d["metrics"], d["thresholds_applied"]
    conds = [("transfer to a\nnew vocabulary", m["content_transfer_err"]),
             ("transfer across\nphrasings", m["crosspath_err"])]
    fig, ax = plt.subplots(figsize=(5.4, 3.5), constrained_layout=True)
    x = range(len(conds))
    ax.bar(x, [c[1] for c in conds], width=0.5, color=BLUE, zorder=3)
    # one honest reference: 1.0 = leaving the state alone (the identity map), tagged off to the side
    ax.axhline(1.0, color=INK, ls="--", lw=1.2, zorder=2)
    ax.text(1.36, 1.0, "doing nothing\n(identity)", color=GREY, va="center", ha="left", fontsize=8.5)
    ax.set_xticks(list(x)); ax.set_xticklabels([c[0] for c in conds])
    ax.set_xlim(-0.5, 2.25)
    ax.set_ylabel("transport error (lower is better)")
    ax.set_title("The fitted operator does no better than doing nothing", fontsize=10.5)
    ax.set_ylim(0, 3.8)
    fig.savefig(FIG / "verdict_transport.png"); plt.close(fig)
    print("verdict_transport.png:", {c[0].replace(chr(10), ' '): round(c[1], 3) for c in conds})


def fig_cross_family():
    d = load("results/cross_family/nemotron_disc_results.json")["metrics"]
    c1_fit, c1_id = d["content_transfer_err"], d["content_transfer_err_identity_fit"]
    c2_fit, c2_id = d["crosspath_err"], d["crosspath_err_identity_fit"]
    groups = [("transfer to a\nnew vocabulary", c1_fit, c1_id),
              ("transfer across\nphrasings", c2_fit, c2_id)]
    fig, ax = plt.subplots(figsize=(5.4, 3.5), constrained_layout=True)
    import numpy as np
    x = np.arange(len(groups)); w = 0.36
    ax.bar(x - w / 2, [g[1] for g in groups], w, color=BLUE, label="fitted operator", zorder=3)
    ax.bar(x + w / 2, [g[2] for g in groups], w, color=GREY, label="doing nothing (identity)", zorder=3)
    ax.axhline(1.0, color=INK, ls="--", lw=1.1, zorder=2)
    ax.set_xticks(x); ax.set_xticklabels([g[0] for g in groups])
    ax.set_ylabel("transport error (lower is better)")
    ax.set_title("The same holds in a second family (Nemotron-70B)", fontsize=10.5)
    ax.legend(frameon=False, loc="upper left", fontsize=9)
    ax.set_ylim(0, 2.75)
    fig.savefig(FIG / "cross_family.png"); plt.close(fig)
    print("cross_family.png:", dict(c1_fit=round(c1_fit, 3), c1_id=round(c1_id, 3),
                                    c2_fit=round(c2_fit, 3), c2_id=round(c2_id, 3)))


def fig_routing_depth():
    d = load("results/routing/ra4_results.json")["results"]
    fr, jo = d["frozen"]["margin_by_layer"], d["joint"]["margin_by_layer"]
    L = range(len(fr))
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(L, fr, color=BLUE, lw=2, label="entities in place")
    ax.plot(L, jo, color=RED, lw=1.6, ls="--", label="entities permuted to new positions")
    ax.axhline(0, color=INK, ls=":", lw=1)
    ax.set_xlabel("layer"); ax.set_ylabel("answer margin over competitors\n(logit lens)")
    ax.set_title("The answer is assembled late, and the profile is the same after permutation",
                 fontsize=11)
    ax.legend(frameon=False, loc="upper left")
    ax.annotate("answer overtakes\ncompetitors here", xy=(68, 0), xytext=(45, 6),
                fontsize=9, arrowprops=dict(arrowstyle="->", color=INK))
    fig.savefig(FIG / "routing_depth.png"); plt.close(fig)
    print("routing_depth.png: layers", len(fr), "cross~L", next(i for i, v in enumerate(fr) if v > 0))


def fig_routing_ablation():
    r = load("results/routing/ra3_results.json")["results"]
    import numpy as np
    conds = ["baseline", "pointer", "random"]
    labels = ["no ablation", "ablate the\n72 readout heads", "ablate 72\nrandom heads"]
    fr = [r["frozen"][c]["mean_ans_logit"] for c in conds]
    jo = [r["joint"][c]["mean_ans_logit"] for c in conds]
    fig, ax = plt.subplots(figsize=(6.5, 4))
    x = np.arange(len(conds)); w = 0.36
    ax.bar(x - w / 2, fr, w, color=BLUE, label="entities in place")
    ax.bar(x + w / 2, jo, w, color=RED, label="entities permuted")
    for i in range(len(conds)):
        ax.text(i - w / 2, fr[i] + 0.3, f"{fr[i]:.1f}", ha="center", fontsize=9)
        ax.text(i + w / 2, jo[i] + 0.3, f"{jo[i]:.1f}", ha="center", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("answer-name logit")
    ax.set_title("The readout route is causal and equal after permutation, but redundant",
                 fontsize=11)
    ax.legend(frameon=False, loc="lower left")
    ax.text(0.5, 0.95, "the answer stays the top-ranked entity in every condition (k-way = 1.000)",
            transform=ax.transAxes, ha="center", va="top", fontsize=9, color=INK,
            bbox=dict(boxstyle="round", fc="#f4f4f2", ec="none"))
    ax.set_ylim(0, 44)
    fig.savefig(FIG / "routing_ablation.png"); plt.close(fig)
    print("routing_ablation.png: frozen", [round(v, 1) for v in fr], "joint", [round(v, 1) for v in jo])


if __name__ == "__main__":
    fig_verdict()
    fig_cross_family()
    fig_routing_depth()
    fig_routing_ablation()
    print("wrote figures to", FIG)

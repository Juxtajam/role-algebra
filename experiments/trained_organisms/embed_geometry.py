"""Item 1 — Embedding geometry diagnostic (Phase 4 spec). No training.

Pairwise cosine similarities among:
  * the 24 frozen fit-pool name embeddings (T1 pool = names 0..23),
  * the 16 symbol embeddings,
  * the 16 property embeddings,
reported per seed (0,1,2): min / max / mean and the full histogram.

Vectors are read from the actual T1 final checkpoints on the dv3-results
volume (stage2/T1/seed{s}/ckpt.pt). For names this equals the init value —
frozenness is verified explicitly against a fresh TinyTransformer(seed).
Symbols/properties are trainable, so both init and final geometries are
reported (the final one is what the match had to discriminate at the end;
the init one is what training started from).
"""

import modal
import pathlib

app = modal.App("dv3-item1-embed-geometry")
vol = modal.Volume.from_name("dv3-results")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.4.1", "numpy")
    .add_local_dir(
        str(pathlib.Path(__file__).resolve().parents[2] / "src"),
        remote_path="/root/dv3",
    )
)


@app.function(image=image, timeout=1800, volumes={"/results": vol})
def geometry():
    import os, sys

    os.environ["DV3_RESULTS"] = "/results"
    sys.path.insert(0, "/root/dv3")
    os.chdir("/root/dv3")

    import numpy as np
    import torch
    from trained import data as D
    from trained.model import TinyTransformer

    FIT_POOL = 24  # T1 pool
    BIN_W = 0.1
    edges = np.arange(-1.0, 1.0 + 1e-9, BIN_W)

    def cos_offdiag(E):
        En = E / np.linalg.norm(E, axis=1, keepdims=True)
        C = En @ En.T
        iu = np.triu_indices(len(E), k=1)
        return C[iu]

    def stats_and_hist(v):
        hist, _ = np.histogram(v, bins=edges)
        return dict(
            n_pairs=int(v.size),
            min=float(v.min()),
            max=float(v.max()),
            mean=float(v.mean()),
            std=float(v.std()),
            frac_abs_gt_0_5=float((np.abs(v) > 0.5).mean()),
            frac_abs_gt_0_3=float((np.abs(v) > 0.3).mean()),
            hist=hist.tolist(),
            bin_edges=[round(float(e), 2) for e in edges],
        )

    def fmt_hist(h):
        lines = []
        for i, c in enumerate(h["hist"]):
            if c == 0:
                continue
            lo, hi = h["bin_edges"][i], h["bin_edges"][i + 1]
            lines.append(f"      [{lo:+.1f},{hi:+.1f}): {c:4d}  {'#' * min(c, 60)}")
        return "\n".join(lines)

    out = {}
    report = []
    for seed in (0, 1, 2):
        ck = torch.load(f"/results/stage2/T1/seed{seed}/ckpt.pt", map_location="cpu")
        E_final = ck["model"]["emb"].numpy()
        ref = TinyTransformer(seed=seed, n_layers=4)
        E_init = ref.emb.detach().numpy()

        # frozenness check: name rows identical init vs final
        name_rows_final = E_final[D.NAME0 :]
        name_rows_init = E_init[D.NAME0 :]
        frozen_ok = bool(np.array_equal(name_rows_final, name_rows_init))
        max_dev = float(np.abs(name_rows_final - name_rows_init).max())

        names_fit = E_final[D.NAME0 : D.NAME0 + FIT_POOL]
        syms_f = E_final[D.SYM0 : D.SYM0 + D.N_SYMS]
        props_f = E_final[D.PROP0 : D.PROP0 + D.N_PROPS]
        syms_i = E_init[D.SYM0 : D.SYM0 + D.N_SYMS]
        props_i = E_init[D.PROP0 : D.PROP0 + D.N_PROPS]

        groups = {
            "names_fit24_frozen": stats_and_hist(cos_offdiag(names_fit)),
            "symbols16_final": stats_and_hist(cos_offdiag(syms_f)),
            "symbols16_init": stats_and_hist(cos_offdiag(syms_i)),
            "properties16_final": stats_and_hist(cos_offdiag(props_f)),
            "properties16_init": stats_and_hist(cos_offdiag(props_i)),
        }
        out[f"seed{seed}"] = dict(
            ckpt_step=int(ck["step"]),
            name_rows_frozen=frozen_ok,
            name_rows_max_dev=max_dev,
            groups=groups,
        )

        report.append("=" * 90)
        report.append(
            f"ITEM 1 — T1/seed{seed} (final ckpt step {ck['step']}) — "
            f"name rows frozen: {frozen_ok} (max |dev| {max_dev:.2e})"
        )
        report.append("=" * 90)
        for gname, g in groups.items():
            report.append(
                f"  {gname}: pairs={g['n_pairs']}  min={g['min']:+.4f}  "
                f"max={g['max']:+.4f}  mean={g['mean']:+.4f}  std={g['std']:.4f}  "
                f"|cos|>0.3: {g['frac_abs_gt_0_3']:.3f}  |cos|>0.5: {g['frac_abs_gt_0_5']:.3f}"
            )
            report.append(fmt_hist(g))

    # reference scale: expected |cos| of two random unit vectors in d=128
    report.append("")
    report.append(
        "Reference: for random unit vectors in d=128, E[cos]=0, "
        "sd = 1/sqrt(128) = 0.088; |cos| > 0.3 is a >3.4-sigma event."
    )

    txt = "\n".join(report)
    print(txt)
    import json
    from pathlib import Path

    Path("/results/checks/item1_embedding_geometry.json").parent.mkdir(
        parents=True, exist_ok=True
    )
    Path("/results/checks/item1_embedding_geometry.json").write_text(
        json.dumps(out, indent=2)
    )
    Path("/results/checks/item1_embedding_geometry.txt").write_text(txt)
    vol.commit()
    return out


@app.local_entrypoint()
def main():
    geometry.remote()

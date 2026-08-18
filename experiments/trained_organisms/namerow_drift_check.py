"""Item 1 addendum — characterise the small drift in the 'frozen' name rows.
Hypothesis: AdamW decoupled weight decay multiplies the whole emb tensor by
(1 - lr*wd) every step regardless of the zeroed gradient, so name rows are
frozen in DIRECTION but uniformly shrunk in norm. Verify: per-row cosine
between init and final name rows, and per-row norm ratio.
"""

import modal
import pathlib

app = modal.App("dv3-item1-addendum")
vol = modal.Volume.from_name("dv3-results")
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.4.1", "numpy")
    .add_local_dir(
        str(pathlib.Path(__file__).resolve().parents[2] / "src"),
        remote_path="/root/dv3",
    )
)


@app.function(image=image, timeout=900, volumes={"/results": vol})
def drift():
    import sys

    sys.path.insert(0, "/root/dv3")
    import numpy as np, torch
    from trained import data as D
    from trained.model import TinyTransformer

    lines = []
    for seed in (0, 1, 2):
        ck = torch.load(f"/results/stage2/T1/seed{seed}/ckpt.pt", map_location="cpu")
        Ef = ck["model"]["emb"].numpy()[D.NAME0 :]
        Ei = TinyTransformer(seed=seed, n_layers=4).emb.detach().numpy()[D.NAME0 :]
        cos = (Ef * Ei).sum(1) / (
            np.linalg.norm(Ef, axis=1) * np.linalg.norm(Ei, axis=1)
        )
        ratio = np.linalg.norm(Ef, axis=1) / np.linalg.norm(Ei, axis=1)
        lines.append(
            f"seed{seed} step {ck['step']}: name-row cos(init,final) "
            f"min={cos.min():.8f} mean={cos.mean():.8f} | "
            f"norm ratio min={ratio.min():.5f} max={ratio.max():.5f} "
            f"mean={ratio.mean():.5f} std={ratio.std():.2e}"
        )
    txt = "\n".join(lines)
    print(txt)
    from pathlib import Path

    Path("/results/checks/item1_namerow_drift.txt").write_text(txt)
    vol.commit()
    return txt


@app.local_entrypoint()
def main():
    drift.remote()

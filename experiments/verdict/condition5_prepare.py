"""Phase 8C — Condition 5 preparation ( instruction C): compute the FULL
patch set from cached activations BEFORE booking the GPU session.

Frozen procedure (committed_config.condition5):
  real:  R fit on P/fit TEST bases (frozen lambda/layer); for each
         P/transfer TEST base, g in all_perms[:2], generator a: patch
         vector v = R_a h_L(x=orb[g]) into episode y = orb[a o g]'s forward
         pass at resid_post[L], answer position; agree = pred_patch ==
         pred_nat(y) (stored 8A-final greedy preds).
  nulls: same with R from shuffled fits (seeds 2000*r+L, r=0..9) and the
         identity fit, on P/fit CAL bases, evaluated on P/transfer CAL
         bases. tau(transport_agree) = 0.95 quantile of the pooled null
         agreements (frozen gt rule).

Outputs (uploaded to dv3-results:phase8c/):
  cond5_patch_vectors.npy   (n_patch, 8192) float16
  cond5_manifest.json       per-item: eval episode row, fit_id, generator
"""

import hashlib
import json

import numpy as np

import activation_discriminator as lib
from activation_discriminator import GENERATORS, GEN_NAMES, PERMS, compose

cfg_txt = (lib.OUT / "committed_config.json").read_text()
assert (
    hashlib.sha256(cfg_txt.encode()).hexdigest()
    == (lib.OUT / "committed_config.sha256").read_text().strip()
)
cfg = json.loads(cfg_txt)
LAYER, LAM = cfg["patch_layer"], cfg["lambda"]
N_SHUF = 10

cells = lib.load_cells()
splits = json.load(open(lib.OUT / "splits.json"))["splits"]
cf, ct = cells[("P", "fit")], cells[("P", "transfer")]

fits = [("real", None, 0, splits["P/fit"]["test"], splits["P/transfer"]["test"])]
for r in range(N_SHUF):
    fits.append(
        (
            f"shuf{r}",
            "shuffled",
            2000 * r + LAYER,
            splits["P/fit"]["cal"],
            splits["P/transfer"]["cal"],
        )
    )
fits.append(
    ("identity", "identity", 0, splits["P/fit"]["cal"], splits["P/transfer"]["cal"])
)

vectors, manifest = [], []
for fit_id, mode, seed, fit_bases, eval_bases in fits:
    Rs = lib.fit_generators(cf, fit_bases, LAYER, LAM, null_mode=mode, null_seed=seed)
    for a in GENERATORS:
        rows_x, rows_y = [], []
        for b in eval_bases:
            for g in PERMS[:2]:  # frozen all_perms[:2]
                rows_x.append(ct.row(b, g))
                rows_y.append(ct.row(b, compose(a, g)))
        X = ct.states(np.array(rows_x), LAYER)
        V = Rs[a].apply_RT(X)  # R_a h_L(x)
        for i, ry in enumerate(rows_y):
            manifest.append(
                dict(
                    fit=fit_id,
                    gen=GEN_NAMES[GENERATORS.index(a)],
                    eval_row=int(ry),
                    src_row=int(rows_x[i]),
                    idx=len(vectors) + i,
                )
            )
        vectors.append(V.astype(np.float16))
    print(f"fit {fit_id}: {sum(len(v) for v in vectors)} vectors so far")

Vall = np.concatenate(vectors, axis=0)
np.save(lib.OUT / "cond5_patch_vectors.npy", Vall)
meta = dict(
    layer=LAYER,
    **{"lambda": LAM},
    config_sha256=(lib.OUT / "committed_config.sha256").read_text().strip(),
    eval_cell="disc_P_transfer",
    n_items=len(manifest),
    perms_per_base=2,
    n_shuffled=N_SHUF,
    natural_preds="preds_disc_P_transfer.json (stored 8A-final greedy)",
    items=manifest,
)
json.dump(meta, open(lib.OUT / "cond5_manifest.json", "w"))
print(
    f"patch set: {Vall.shape} float16 "
    f"({Vall.nbytes/1e6:.0f} MB), {len(manifest)} items"
)
h = lib.sha_file(lib.OUT / "cond5_patch_vectors.npy")
print("patch vectors sha256:", h)
(lib.OUT / "cond5_patch_vectors.sha256").write_text(h + "\n")

"""Phase 8C resolution — item 2 prep: fetch unembedding (lm_head.weight) and
input-embedding (model.embed_tokens.weight) rows for the 24 verified
answer-candidate name tokens of Qwen/Qwen2.5-72B-Instruct at the pinned
revision, WITHOUT downloading full shards.

Method (declared): resolve the safetensors shard for each tensor from
model.safetensors.index.json at the pinned revision; read the shard's JSON
header via an HTTP Range request; compute per-row byte offsets
(row i occupies [start + i*d*2, start + (i+1)*d*2) in bf16); fetch each of the
24 rows with a Range request; decode bf16 -> float32. Verifies tensor shapes
against config.json (vocab 152064 [lm_head] / d 8192) and records sha256 of
the assembled arrays.
"""

import json, struct, hashlib, pathlib
import numpy as np
import requests

REPO = "Qwen/Qwen2.5-72B-Instruct"
REV = "495f39366efef23836d0cfae4fbe635880d2be31"
BASE = f"https://huggingface.co/{REPO}/resolve/{REV}"
OUT = pathlib.Path(__file__).resolve().parents[2] / "results/robustness/resolution"
OUT.mkdir(parents=True, exist_ok=True)

pools = json.load(
    open(
        str(
            pathlib.Path(__file__).resolve().parents[2]
            / "results/verdict/gate/tasks/name_pools.json"
        )
    )
)
names = pools["fit"] + pools["transfer"]
ids = {n: pools["verified"][n]["token_id"] for n in names}
print("names:", len(names), "ids:", sorted(ids.values())[:3], "...")

sess = requests.Session()


def get(url, headers=None):
    r = sess.get(url, headers=headers or {}, timeout=120, allow_redirects=True)
    r.raise_for_status()
    return r


cfg = get(f"{BASE}/config.json").json()
print(
    "config: hidden",
    cfg["hidden_size"],
    "vocab",
    cfg["vocab_size"],
    "tie",
    cfg.get("tie_word_embeddings"),
)
d = cfg["hidden_size"]

index = get(f"{BASE}/model.safetensors.index.json").json()["weight_map"]
targets = {
    "lm_head.weight": index["lm_head.weight"],
    "model.embed_tokens.weight": index["model.embed_tokens.weight"],
}
print("shards:", targets)


def shard_header(shard):
    url = f"{BASE}/{shard}"
    r = get(url, headers={"Range": "bytes=0-7"})
    n = struct.unpack("<Q", r.content)[0]
    hdr = json.loads(get(url, headers={"Range": f"bytes=8-{8+n-1}"}).content)
    return url, 8 + n, hdr


out = {}
meta = {
    "repo": REPO,
    "revision": REV,
    "d_model": d,
    "method": "HTTP Range requests against pinned-revision safetensors shards; "
    "per-row bf16 slices decoded to float32; no full shard downloaded",
    "tensors": {},
}
for tname, shard in targets.items():
    url, data0, hdr = shard_header(shard)
    info = hdr[tname]
    assert info["dtype"] == "BF16", info
    shape = info["shape"]
    assert shape[1] == d, shape
    t0 = data0 + info["data_offsets"][0]
    rows = {}
    for nm, tid in ids.items():
        assert tid < shape[0]
        s = t0 + tid * d * 2
        raw = get(url, headers={"Range": f"bytes={s}-{s + d*2 - 1}"}).content
        assert len(raw) == d * 2
        u16 = np.frombuffer(raw, dtype=np.uint16)
        f32 = (u16.astype(np.uint32) << 16).view(np.float32)
        rows[nm] = f32
    M = np.stack([rows[nm] for nm in names])  # (24, d) ordered fit then transfer
    key = "lm_head" if tname == "lm_head.weight" else "embed_tokens"
    out[key] = M
    meta["tensors"][key] = {
        "tensor": tname,
        "shard": shard,
        "shape_full": shape,
        "rows_order": names,
        "token_ids": [ids[n] for n in names],
        "sha256": hashlib.sha256(M.tobytes()).hexdigest(),
        "row_norm_min": float(np.linalg.norm(M, axis=1).min()),
        "row_norm_max": float(np.linalg.norm(M, axis=1).max()),
    }
    print(
        key,
        shard,
        shape,
        "norms",
        round(meta["tensors"][key]["row_norm_min"], 3),
        round(meta["tensors"][key]["row_norm_max"], 3),
    )

np.savez(
    OUT / "name_token_rows.npz",
    lm_head=out["lm_head"],
    embed_tokens=out["embed_tokens"],
    names=np.array(names),
    token_ids=np.array([ids[n] for n in names]),
)
json.dump(meta, open(OUT / "name_token_rows_meta.json", "w"), indent=2)

# sanity: cosine structure
for key, M in out.items():
    Mn = M / np.linalg.norm(M, axis=1, keepdims=True)
    C = Mn @ Mn.T
    off = C[~np.eye(24, dtype=bool)]
    print(f"{key}: offdiag cos mean {off.mean():.3f} max {off.max():.3f}")
print("saved", OUT / "name_token_rows.npz")

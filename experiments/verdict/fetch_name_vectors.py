"""Phase 8C — fetch the 24 name READOUT vectors (lm_head.weight rows) of
Qwen2.5-72B-Instruct @ pinned revision, via HTTP range reads on the
safetensors shards (24 x 16 KB instead of 145 GB). Also fetches the input
embedding rows (secondary diagnostic). Verifies dtype/shape and saves
results/phase8c/name_vectors.npz."""

import json
import pathlib
import struct

import numpy as np
import requests

MODEL = "Qwen/Qwen2.5-72B-Instruct"
REV = "495f39366efef23836d0cfae4fbe635880d2be31"
BASE = f"https://huggingface.co/{MODEL}/resolve/{REV}"
D = 8192
ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "results/verdict/discriminator"
OUT.mkdir(parents=True, exist_ok=True)

pools = json.load(open(ROOT / "results/verdict/gate/tasks/name_pools.json"))
names = pools["fit"] + pools["transfer"]
tid = {n: pools["verified"][n]["token_id"] for n in names}

sess = requests.Session()


def get(url, headers=None):
    r = sess.get(url, headers=headers, timeout=60)
    r.raise_for_status()
    return r


def get_range(url, start, length):
    r = get(url, headers={"Range": f"bytes={start}-{start + length - 1}"})
    assert len(r.content) == length, (len(r.content), length)
    return r.content


def shard_header(url):
    n = struct.unpack("<Q", get_range(url, 0, 8))[0]
    hdr = json.loads(get_range(url, 8, n))
    return hdr, 8 + n


def bf16_rows(url, hdr, data_off, tensor, row_ids, d=D):
    info = hdr[tensor]
    assert info["dtype"] == "BF16", info
    shape = info["shape"]
    assert shape[1] == d, shape
    t0 = data_off + info["data_offsets"][0]
    rows = {}
    for r in row_ids:
        raw = get_range(url, t0 + r * d * 2, d * 2)
        u16 = np.frombuffer(raw, dtype=np.uint16)
        f32 = (u16.astype(np.uint32) << 16).view(np.float32)
        rows[r] = f32.astype(np.float64)
    return rows, shape


idx = get(f"{BASE}/model.safetensors.index.json").json()
wmap = idx["weight_map"]
print("lm_head shard:", wmap.get("lm_head.weight"))
print("embed shard:", wmap.get("model.embed_tokens.weight"))

out = {}
for tensor, key in (
    ("lm_head.weight", "readout"),
    ("model.embed_tokens.weight", "embed"),
):
    shard = wmap[tensor]
    url = f"{BASE}/{shard}"
    hdr, off = shard_header(url)
    rows, shape = bf16_rows(url, hdr, off, tensor, sorted(tid.values()))
    print(f"{tensor}: shape={shape} fetched {len(rows)} rows")
    M = np.stack([rows[tid[n]] for n in names])
    out[key] = M
    print(
        f"  {key}: norms min/max = {np.linalg.norm(M, axis=1).min():.3f}/"
        f"{np.linalg.norm(M, axis=1).max():.3f}"
    )

np.savez(
    OUT / "name_vectors.npz",
    names=np.array(names),
    token_ids=np.array([tid[n] for n in names]),
    readout=out["readout"],
    embed=out["embed"],
    model=MODEL,
    revision=REV,
)
print("saved", OUT / "name_vectors.npz")

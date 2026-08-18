"""Shared paths/helpers for the Track B discriminator driver scripts."""
import hashlib
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "dv3_pkg"))

import trained_discriminator as dl  # noqa: E402

OUT = ROOT / "results/trained_organisms/large/disc"
CFG_PATH = ROOT / "phase10/trackB/disc_committed_config.json"
CFG_SHA = ROOT / "phase10/trackB/disc_committed_config.sha256"


def sha256_file(p):
    return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()


def verify_cfg():
    h = sha256_file(CFG_PATH)
    rec = CFG_SHA.read_text().split()[0]
    assert h == rec, f"config sha mismatch: {h} != {rec}"
    return h


def org_dir(seed, arm):
    d = OUT / f"seed{seed}_{arm}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def load_set(seed, arm, name):
    z = np.load(org_dir(seed, arm) / f"{name}.npz")
    return {k: z[k] for k in z.files}


def cal_test_split(strict_ok):
    """Even filtered-index -> cal, odd -> test; equalised counts (design 5.2)."""
    cal, test = strict_ok[0::2], strict_ok[1::2]
    n = min(len(cal), len(test))
    return cal[:n], test[:n]


def xy_fn(z, position, layer):
    if position == "answer":
        return lambda bases, a: dl.XY_answer(z["acts"], bases, a, layer)
    return lambda bases, a: dl.XY_carry(z["carr"], bases, a, layer)


def states_at(z, position, bases, layer):
    idx = np.concatenate([[dl.ep_index(b, g) for g in dl.PERMS3]
                          for b in bases]).astype(int)
    if position == "answer":
        return z["acts"][idx, layer]
    return z["carr"][idx][:, :, layer].reshape(-1, dl.D)


def gated_organisms():
    fa = json.load(open(ROOT / "results/trained_organisms/large/formed_analysis.json"))
    gated = []
    for s, row in fa["per_seed"].items():
        for arm in ("R", "C"):
            if row.get(arm, {}).get("FORMED_final"):
                gated.append((int(s), arm))
    return sorted(gated)

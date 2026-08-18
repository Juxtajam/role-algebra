"""Progress logging + results I/O.

Every module logs through here so a long Colab run is visibly alive:
lines go to stdout AND are appended to <results>/progress.log, which
survives runtime disconnects when results live on Google Drive.
"""

import json
import os
import pathlib
import time


def results_dir() -> pathlib.Path:
    d = pathlib.Path(os.environ.get("DV3_RESULTS", "results"))
    d.mkdir(parents=True, exist_ok=True)
    return d


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(results_dir() / "progress.log", "a") as f:
        f.write(line + "\n")


def save_json(relpath: str, obj) -> pathlib.Path:
    p = results_dir() / relpath
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(obj, f, indent=2, default=_coerce)
    log(f"wrote {p}")
    return p


def load_json(relpath: str):
    with open(results_dir() / relpath) as f:
        return json.load(f)


def exists(relpath: str) -> bool:
    return (results_dir() / relpath).exists()


def _coerce(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return str(x)

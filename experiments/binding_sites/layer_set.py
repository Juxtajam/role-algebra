"""Phase 9 STAGE 1 (local prep) — Task 4: pre-register the layer set.

Every 4th layer [0, 4, ..., 76] UNION the frozen 8C reporting layer set
read from results/phase8c/committed_config.json (reporting_layer_set).
Writes results/phase9/layer_set.json.
"""

import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[2]
P9 = ROOT / "results/binding_sites"

cfg = json.load(open(ROOT / "results/verdict/discriminator/committed_config.json"))
every4 = list(range(0, 77, 4))  # 0,4,...,76
c8 = cfg["reporting_layer_set"]
layers = sorted(set(every4) | set(c8))

rec = dict(
    phase="9-stage1-prep",
    rule=(
        "every 4th layer [0,4,...,76] UNION frozen 8C reporting_layer_set "
        "from results/phase8c/committed_config.json"
    ),
    every_4th=every4,
    phase8c_reporting_set=c8,
    layers=layers,
    count=len(layers),
    n_model_layers=80,
    convention=(
        "layer l = resid_post of decoder layer l = "
        "hidden_states[l+1] (8A-final convention)"
    ),
)
json.dump(rec, open(P9 / "layer_set.json", "w"), indent=2)
print("layers:", layers)
print("count:", len(layers))
print("written:", P9 / "layer_set.json")

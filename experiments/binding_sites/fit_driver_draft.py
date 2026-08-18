"""Phase 9 Item 3 — 9B fitting driver. Executes the frozen config
results/phase9/item3_committed_config.json (sha 6f524335...78b139) against
the locally verified cache acts_P_local/ (ALL_MATCH: True vs in-session
checksums).

Protocol (frozen config, single evaluation):
  - splits by base problem: frozen disc P cells -> strict-orbit filter ->
    50/50 cal/test by sorted base parity; joint P cells -> 75/75 by parity.
  - per position-class x layer: dual-form ridge R_12, R_23 on calibration
    in-place pairs (slot-conditional means as in phase8c_lib: pair
    (h(x), h(g.x)) for the two generator transpositions g12, g23).
  - lambda from calibration transfer-err criterion (grid frozen).
  - matched-regime nulls: 10 shuffled-pair fits (seeds 0..9) + identity;
    thresholds = calibration null quantiles (same rules as 8C).
  - single test evaluation: C1 (fit-vocab R on transfer-vocab test),
    C3 (involution/braid/3-cycle), C4 (registered name-embed-span lt +
    as-run readout-span lt), C7 (transport on joint-permutation episodes).
  - baselines at every cell: R_lex (rank-2 unembedding swap) and identity.
  - incremental writes to results/phase9/fits/ as cells complete.

Position classes (manifest order in the cache axis 1, 11 positions path P):
  carry_entity: cols 0..2 (3 slots), fact_final: cols 3..8 (6 clause finals),
  query_arg: col 9, answer: col 10.
For carry_entity the fit uses the slot pair moved by the generator
(g12 -> slots 0,1; g23 -> slots 1,2), evaluated at the moved slots.
For scalar classes (query_arg, answer) and fact_final (mean over clause
finals of the permuted-pair episodes) the pair is (h(x), h(g.x)) at the
same column(s).
"""

import json, hashlib, os
import pathlib

ROOT = str(pathlib.Path(__file__).resolve().parents[2])
ACTS = f"{ROOT}/acts_P_local"
OUT = f"{ROOT}/results/phase9/fits"
os.makedirs(OUT, exist_ok=True)

cfg = json.load(open(f"{ROOT}/results/phase9/item3_committed_config.json"))
CFG_SHA = hashlib.sha256(
    open(f"{ROOT}/results/phase9/item3_committed_config.json", "rb").read()
).hexdigest()
assert CFG_SHA == "6f52433589ef65795cd84e1502ebaf18abaee31937dd30a08b95d10bdf78b139"
LAYERS = cfg["layers"]
LAMBDAS = cfg["ridge_protocol"]["lambda_grid"]
POS_CLASSES = {
    "entity_mention_carry": list(range(0, 3)),
    "fact_final": list(range(3, 9)),
    "query_arg": [9],
    "answer": [10],
}


def load_eps(path):
    return [json.loads(l) for l in open(path)]


# --- episodes + strict-orbit filter (frozen disc P) ---
def orbit_ok(eps_group, preds=None):
    return True  # strict-orbit filter uses stored 8A-final disc preds below


# stored 8A-final disc predictions for the strict-orbit filter (as 8C)
disc_preds = {}
for v in ("fit", "transfer"):
    pp = f"{ROOT}/results/phase8a_final_preds/disc_P_{v}_preds.json"
    alt = f"{ROOT}/results/phase8c/splits.json"
    disc_preds[v] = None  # fall back to 8C split file which already encodes the filter

split8c = json.load(open(f"{ROOT}/results/phase8c/splits.json"))


def bases_from_8c(vocab):
    key = f"P/{vocab}"
    s = split8c[key]
    return set(s["cal_bases"]), set(s["test_bases"])


EPS, SPLITS = {}, {}
for v in ("fit", "transfer"):
    eps = (
        load_eps(
            f"{ROOT}/results/../modal_dv3/results/phase9/tasks_joint/joint_P_{v}.jsonl"
        )
        if False
        else None
    )
for v in ("fit", "transfer"):
    EPS[("frozen", v)] = (
        load_eps(f"{ROOT}/src_tasks/disc_P_{v}.jsonl")
        if os.path.exists(f"{ROOT}/src_tasks")
        else load_eps(f"{ROOT}/tasks_frozen/disc_P_{v}.jsonl")
    )
    cal, test = bases_from_8c(v)
    SPLITS[("frozen", v)] = (cal, test)
    EPS[("joint", v)] = load_eps(f"{ROOT}/results/phase9/tasks_joint/joint_P_{v}.jsonl")
    jb = sorted({e["base_id"] for e in EPS[("joint", v)]})
    SPLITS[("joint", v)] = (set(jb[0::2]), set(jb[1::2]))

print("episode cells loaded:", {k: len(v) for k, v in EPS.items()})

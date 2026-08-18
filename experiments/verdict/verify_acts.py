"""Phase 8C — Step 0b: verify sha256 of the 8 downloaded activation files
against acts/checksums.json (the in-session verified digests)."""

import hashlib, json, pathlib

d = pathlib.Path(__file__).resolve().parents[2] / "results/verdict/answer_position/acts"
meta = json.load(open(d / "checksums.json"))
assert (
    meta["content_hash"]
    == "84f2e54d85d6e8aa4c1474b608bef5ab69babe54353ef0ef2702d9f6ed38baef"
)
ok = True
out = {}
for fn, rec in sorted(meta["files"].items()):
    h = hashlib.sha256((d / fn).read_bytes()).hexdigest()
    match = h == rec["sha256"]
    ok &= match
    out[fn] = dict(expected=rec["sha256"], got=h, match=match, shape=rec["shape"])
    print(f"{fn}: {'OK' if match else 'MISMATCH'} {h[:16]}")
json.dump(
    dict(all_match=ok, content_hash=meta["content_hash"], files=out),
    open(
        str(
            pathlib.Path(__file__).resolve().parents[2]
            / "results/verdict/discriminator_acts_verify.json"
        ),
        "w",
    ),
    indent=2,
)
assert ok
print("STEP 0b (activation checksums): PASS — 8/8")

"""RA1 analysis — is role binding a ROUTING algebra? (paper/routing.md)

From the captured answer-position attention masses on entity-name tokens
(frozen in-place + joint name+order P episodes):

1. Pointer heads: heads whose answer-position attention concentrates on the
   ANSWER entity (the entity filling the queried role), selectively vs the
   other entities. Identified on the FROZEN set.
2. Concentration: how few heads carry the routing.
3. Role-tracking (the test): on the JOINT set — where the answer entity has
   MOVED position — do the pointer heads still attend to the answer entity
   (track it by content) or not (attend a fixed position)? If the answer-entity
   attention and its selectivity survive the joint permutation, the readout
   follows the answer rather than a fixed position. These are late copy heads,
   so this describes how the retrieval reads out, not a role operator.

Reads results/phase10/routing/attn/{frozen_Pfit,joint_Pfit}.npz + meta.json.
Writes results/phase10/routing/ra1_results.json.
"""

import json
import pathlib

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
ATT = ROOT / "results/routing/attn"


def load(tag):
    z = np.load(ATT / f"{tag}.npz")
    mass, ai, valid = z["mass"], z["answer_idx"], z["valid"]
    mass, ai = mass[valid], ai[valid]  # (n, L, H, k), (n,)
    n, L, H, k = mass.shape
    # mass on the ANSWER entity: gather along the slot axis by answer_idx
    ans = np.take_along_axis(mass, ai[:, None, None, None], axis=3)[..., 0]  # (n,L,H)
    # mean over the (k-1) non-answer entities
    tot = mass.sum(-1)  # (n,L,H)
    other = (tot - ans) / (k - 1)
    return dict(
        n=n,
        L=L,
        H=H,
        k=k,
        answer_mass=ans.mean(0),
        other_mass=other.mean(0),
        answer_mass_raw=ans,
        other_mass_raw=other,
    )


def main():
    meta = json.load(open(ATT / "meta.json"))
    fr, jo = load("frozen_Pfit"), load("joint_Pfit")
    L, H = fr["L"], fr["H"]
    print(
        f"{meta['model'].split('/')[-1]}: {L} layers x {H} heads; "
        f"frozen n={fr['n']}, joint n={jo['n']}"
    )

    # selectivity = answer_mass - other_mass (per head), on the FROZEN set
    sel_fr = fr["answer_mass"] - fr["other_mass"]  # (L,H)
    # top pointer heads by frozen answer-entity selectivity
    flat = [
        (
            int(l),
            int(h),
            float(sel_fr[l, h]),
            float(fr["answer_mass"][l, h]),
            float(fr["other_mass"][l, h]),
        )
        for l in range(L)
        for h in range(H)
    ]
    flat.sort(key=lambda x: -x[2])
    top = flat[:15]

    # role-equivariance on the JOINT set for these pointer heads
    print(
        f"\n{'layer.head':>10} | {'FROZEN ans/oth/sel':>26} | {'JOINT ans/oth/sel':>26} | role-follows?"
    )
    pointer_rows = []
    for l, h, s, am, om in top:
        jam, jom = float(jo["answer_mass"][l, h]), float(jo["other_mass"][l, h])
        jsel = jam - jom
        # role-following: on joint, still attends the answer entity selectively
        # (keeps a positive selectivity a meaningful fraction of the frozen one)
        follows = bool(jsel > 0.05 and jsel > 0.4 * s)
        pointer_rows.append(
            dict(
                layer=l,
                head=h,
                frozen_answer=am,
                frozen_other=om,
                frozen_sel=s,
                joint_answer=jam,
                joint_other=jom,
                joint_sel=jsel,
                role_follows=follows,
            )
        )
        print(
            f"{l:>4}.{h:<5} | {am:.3f}/{om:.3f}/{s:+.3f}      | "
            f"{jam:.3f}/{jom:.3f}/{jsel:+.3f}      | {follows}"
        )

    # concentration: fraction of total answer-selectivity carried by top-5 heads
    all_sel = np.clip(sel_fr, 0, None)
    frac_top5 = float(np.sort(all_sel.ravel())[-5:].sum() / max(all_sel.sum(), 1e-9))
    n_strong = int((sel_fr > 0.1).sum())

    # headline: do the pointer heads route by role on the joint set?
    n_follow = sum(r["role_follows"] for r in pointer_rows)
    top_head = pointer_rows[0]
    verdict = (
        "Role-tracking readout: the pointer heads keep attending the answer entity after "
        "the joint permutation moves it, so the readout follows the answer by content rather "
        "than by a fixed position. These are late copy heads reading out an answer computed "
        "upstream, so this is a description of how the retrieval reads out, not a transportable "
        "role operator; see the causal-and-redundant follow-up (ra3)."
        if n_follow >= max(1, len(pointer_rows) // 3)
        else "The readout does not track the answer cleanly under the joint permutation "
        "(position-mixed or diffuse)."
    )

    out = dict(
        model=meta["model"],
        layers=L,
        heads=H,
        frozen_n=fr["n"],
        joint_n=jo["n"],
        n_strong_pointer_heads_frozen=n_strong,
        top5_selectivity_fraction=frac_top5,
        pointer_heads=pointer_rows,
        n_pointer_heads_role_following=n_follow,
        verdict=verdict,
    )
    json.dump(
        out, open(ROOT / "results/routing/ra1_results.json", "w"), indent=1
    )
    print(
        f"\nstrong pointer heads (frozen sel > 0.1): {n_strong}; "
        f"top-5 carry {frac_top5:.1%} of positive selectivity"
    )
    print(
        f"pointer heads that route by ROLE on the joint set: {n_follow}/{len(pointer_rows)}"
    )
    print(f"\nVERDICT: {verdict}")


if __name__ == "__main__":
    main()

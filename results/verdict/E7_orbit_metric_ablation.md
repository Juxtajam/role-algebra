# E7 — Orbit-metric ablation (descriptive; stored gate results)

| model | cell | episode | strict-orbit | consistency | ep−strict | hidden? |
|---|---|---|---|---|---|---|
| Qwen2.5-72B (8A frozen) | P/fit | 1.000 | 1.000 | 1.000 | +0.000 | — |
| Qwen2.5-72B (8A frozen) | P/transfer | 1.000 | 1.000 | 1.000 | +0.000 | — |
| Qwen2.5-72B (8A frozen) | G/fit | 0.988 | 0.980 | 0.990 | +0.008 | — |
| Qwen2.5-72B (8A frozen) | G/transfer | 0.983 | 0.970 | 0.970 | +0.013 | — |
| Qwen2.5-72B (9 joint-perm) | P/fit | 1.000 | 1.000 | 1.000 | +0.000 | — |
| Qwen2.5-72B (9 joint-perm) | P/transfer | 1.000 | 1.000 | 1.000 | +0.000 | — |
| Qwen2.5-72B (9 joint-perm) | G/fit | 0.984 | 0.933 | 0.933 | +0.051 | — |
| Qwen2.5-72B (9 joint-perm) | G/transfer | 0.988 | 0.940 | 0.940 | +0.048 | — |
| Nemotron-70B (10A) | P/fit | 0.952 | 0.850 | 0.850 | +0.102 | **yes** |
| Nemotron-70B (10A) | P/transfer | 0.908 | 0.760 | 0.760 | +0.148 | — |
| Nemotron-70B (10A) | G/fit | 0.933 | 0.870 | 0.910 | +0.063 | — |
| Nemotron-70B (10A) | G/transfer | 0.963 | 0.930 | 0.940 | +0.033 | — |
| Llama-3.3-70B (10A) | P/fit | 1.000 | 1.000 | 1.000 | +0.000 | — |
| Llama-3.3-70B (10A) | P/transfer | 1.000 | 1.000 | 1.000 | +0.000 | — |
| Llama-3.3-70B (10A) | G/fit | 0.930 | 0.830 | 0.830 | +0.100 | — |
| Llama-3.3-70B (10A) | G/transfer | 0.943 | 0.900 | 0.920 | +0.043 | — |

**Max episode−strict gap:** 0.100. **Cells passing episode-acc 0.95 but failing strict-orbit 0.90:** 0 (none).

Excluding the Nemotron pre-D10 markdown artifact, episode accuracy overstates orbit competence by up to 0.100 on genuine cells. The sharpest genuine case is Qwen2.5-72B under joint permutation (G-cells): episode accuracy 0.984/0.988 reads as near-ceiling, while strict-orbit 0.933/0.940 exposes the role-consistency failure that carried the Phase 9 H_retrieval_everywhere finding. A gate on episode accuracy alone would have passed it. This is why the gate is conjunctive over all three metrics.
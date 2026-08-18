# Role Algebra

*By Justas Miliauskas*

![One episode from the task](figures/task.png)

The project aims to uncover whether language models that can use/assign interchangeable roles do this via a reusable symmetrical circuit, or produce consistent answers using chained look-up. The full writeup is in [`paper.md`](paper.md).

## What it shows

When performing a task that binds several interchangeable entities to roles, and presented with a question utilizing those, the (sufficiently large, e.g. Qwen 72B) LLM answers correctly under all permutations. I initially posited that such behavior could be performed by using a reusable role representation. To test this, I fit an unconstrained linear operator between the activations of a model in one assignment and the activations of a permuted version (such a method was validated on synthetic organisms). The operator was tested for transfer across content, phrasing and obeying the group laws. Unfortunately, it does not. Probing at the answer positions and at every layer, the fitted operator was equivalent to not changing the activations at all. Therefore, the mechanism is likely chained associative retrieval (via binding IDs, perhaps), and not a transportable role operator.

![The two hypotheses, and the measurement](figures/hypotheses.png)

The negative held up against a lot of other tried methods, like different model families, checking attention and MLP outputs, and low-complexity nonlinear maps. I also briefly checked out the attention routing, and it failed to recover the operator too.

## Reproducing

Set up the analysis environment.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

One reproduction is fully self-contained and needs nothing but the repository. It re-validates the
instrument from scratch on freshly drawn synthetic organisms.

```bash
python experiments/instrument/revalidate_instrument.py
```

The verdicts on the large model recompute from cached activations. Fetch them from the compute volume
first, then run the analysis.

```bash
modal volume get dv3-results phase10/routing/attn results/routing/attn
python experiments/routing/attention_analysis.py
```

The captures and trained-model runs execute on Modal. Those scripts end in `_modal.py` and pin their
own container versions in the file. Run scripts from the repository root.

## Layout

```
paper.md          the writeup
src/              the library: the discriminator, calibration, synthetic and trained organisms
experiments/      analysis code, grouped by the stage of the investigation
  instrument/       the discriminator and its validation on synthetic organisms
  trained_organisms/ the detour training small models from scratch
  verdict/          the behavioural gate and the main discriminator verdict on the 72B
  robustness/       the robustness battery and the deflationary baselines
  binding_sites/    all positions and layers on the 72B
  extensions/       nonlinear maps and the attention and MLP outputs
  cross_family/     Llama and Nemotron
  larger_group/     k=4
  forced_reuse/     the permutation-transfer task
  routing/          the attention-routing analysis
results/          stored metrics, grouped the same way (large arrays are git-ignored)
figures/          the figures used in the writeup, and the script that makes them
tasks_frozen/     the frozen task set
```

The large activation and attention arrays are not in git. They are re-fetchable from the compute
volume the analysis reads from. The stored metrics under `results/` carry the numbers the writeup
cites. The `experiments/` and `results/` groups line up one to one.

## License

Released under the [MIT License](LICENSE).

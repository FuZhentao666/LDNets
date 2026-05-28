# Reference Result Audit

Date: 2026-05-28

Purpose: keep paper/notebook reference metrics separate from local reproduction,
same-budget controls, and new architecture experiments. This prevents comparing
results from different cases, budgets, sample splits, or model definitions.

## Key Finding

The quoted paper result with test NRMSE `1.88e-5` and Pearson dissimilarity
`3.30e-9` is a Test Case `1a` result, not Test Case `1c`.

It should not be used as the reference target for Case `1c`. Case `1c` has a
separate author/notebook reference around NRMSE `2.039e-2` and Pearson
dissimilarity `1.152e-2` for `data_1c_fmax1.0.npy`.

Primary sources:

- Paper: Regazzoni et al., Nature Communications 2024,
  https://www.nature.com/articles/s41467-024-45323-x
- Supplementary Information:
  https://static-content.springer.com/esm/art%3A10.1038%2Fs41467-024-45323-x/MediaObjects/41467_2024_45323_MOESM1_ESM.pdf
- Code/notebook anchors:
  - `src/TestCase_1.py`
  - `src/TestCase_1c.ipynb`
  - `runs/case1/TestCase_1c_metrics.json`
  - `PROGRESS.md`

## Paper Result Anchor

| Case | Paper context | Train samples | Test samples | Latent dim | Optimizer budget | Paper metric | Equivalence class |
| --- | --- | ---: | ---: | ---: | --- | --- | --- |
| `1a` | finite latent dimension, constant parameters | 100 | 500 unseen | 2 | BFGS `50000` in Supplementary Table 2 | test NRMSE `1.88e-5`, train NRMSE `1.81e-5`, test Pearson dissim. `3.30e-9`, train Pearson dissim. `3.00e-9` | `paper_reference_case1a` |

Notes:

- The paper paragraph quoted by the user appears under Test Case `1a`.
- Supplementary Table 2 reports this very small error for Case `1a` only after
  increasing BFGS epochs to `50000`.
- The project's normal Case `1a` notebook/runner reference is not this
  `50000` BFGS extreme-accuracy setting; the local reproducible runner currently
  records the author notebook reference `8.576e-3` for its default Case `1a`
  budget.

## Test Case 1c Author / Notebook Reference

| Field | Reference value |
| --- | --- |
| Case | `1c`, infinite latent dimension setting |
| Data file | `data/ADR/data_1c_fmax1.0.npy` |
| Dataset shape | `output (1000, 101, 100)`, `forcing (1000, 101, 3)` |
| Raw time grid | `t = 0.0 ... 10.0`, `101` stored instants |
| Raw space grid | `x = -1.0 ... 0.98`, `100` stored points |
| Runner resampling step | `dt = 5e-2` |
| Sample split in local notebook/runner | train `0:100`, valid `100:200`, test `200:300` |
| Local test sample count | `100` samples, not the quoted `500` Case `1a` test samples |
| Input signals | amplitude, phase, frequency |
| Nominal normalization ranges | amplitude `[0, 2]`, phase `[-4, 4]`, frequency `[0.25, 1]` |
| Observed forcing value range in file | mins `[-0.3896, -6.0531, 0.2574]`, maxs `[2.2705, 5.2312, 0.9926]` |
| Latent states | `4` |
| Dynamics network | Dense `16`, Dense `16`, Dense `4` |
| Reconstruction network | Dense `8`, Dense `8`, Dense `1` |
| Initial latent state | zero latent init, `state = tf.zeros(...)` |
| Regularization | `alpha_reg = 2.7e-4` |
| Optimizer budget | Adam `200`, learning rate `1e-2`, then BFGS `1800` |
| Reference NRMSE | `2.039e-2` |
| Reference Pearson dissimilarity | `1.152e-2` |
| Equivalence class | `paper_notebook_reference_case1c` |

Important caveat:

- Some notebook markdown text may describe a different Adam count, but the
  executable notebook/code path uses Adam `200` and BFGS `1800`. Use executable
  code and output metrics as the reference.

## Local Reproduction Of Author Case 1c Setup

| Case | Runner | Budget | Batch samples | Local NRMSE | Reference NRMSE | Local Pearson dissim. | Reference Pearson dissim. | Comparable? | Equivalence class |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| `1c` | `src/TestCase_1.py` | Adam `200` + BFGS `1800` | `25` | `2.059e-2` | `2.039e-2` | `1.175e-2` | `1.152e-2` | Yes | `author_reproduction_case1c` |

Interpretation:

- This is the current local result that should be compared with the Case `1c`
  author/notebook reference.
- The reproduction is close to the recorded reference. Small differences are
  expected from runtime/library details and chunked full-batch evaluation.
- This result is not expected to match the quoted Case `1a` `1.88e-5` metric.

## Project Experiments That Are Not Paper-Equivalent

| Family | Case | Budget | Sensor ratio | Seeds | Result summary | Comparable to paper quote? | Required caveat | Equivalence class |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Original LDNet Adam-only control | `1c` | Adam `200` + BFGS `0` | full | `0` | NRMSE `8.883e-2`, Pearson `2.489e-1` | No | Same-budget diagnostic only; not author-budget baseline. | `same_budget_control` |
| Stage 1 JEPA transfer | `1c` | Adam `200` + BFGS `0` | `1.00/0.20/0.05` | `0` | NRMSE around `7.656e-2 / 8.180e-2 / 7.701e-2` | No | Code-path and transfer diagnostic only. | `diagnostic_new_architecture` |
| `no_jepa_sparse` near-convergence | `1c` | Adam `4000` + BFGS `150` | `1.00/0.50/0.20/0.10/0.05` | `0/1/2` | mean NRMSE range `0.0267-0.0283` | No | Different architecture: observation encoder and inferred initial latent state. Current Case `1c` result is weaker than original reference `~0.0204`. | `new_architecture_experiment` |

## Why Current Sparse / JEPA Case 1c Is Not Equivalent

Original Case `1c`:

- Uses original LDNet with zero initial latent state.
- Learns latent dynamics only from input signals and reconstruction loss.
- Uses author-selected Case `1c` architecture and regularization.

Sparse / JEPA runner:

- Uses an observation encoder to infer the initial latent state from sensor
  context, even when `lambda_jepa = 0`.
- Adds encoder/teacher/predictor modules to the model class.
- Uses sensor-ratio sweeps and a different optimizer schedule in the recent
  main-branch matrix.

Therefore, sparse / JEPA Case `1c` results should be presented as method
experiments, not as paper reproduction.

## Practical Comparison Rules

Use these rules when reading or reporting results:

1. Compare the quoted paper `1.88e-5` metric only against Case `1a` experiments
   that intentionally reproduce the same high-BFGS setting.
2. Compare Case `1c` local reproduction against the Case `1c` notebook/reference
   values `2.039e-2` and `1.152e-2`.
3. Mark every result with an `equivalence_class`:
   - `paper_reference_case1a`
   - `paper_notebook_reference_case1c`
   - `author_reproduction_case1c`
   - `same_budget_control`
   - `new_architecture_experiment`
   - `smoke_or_diagnostic`
4. Do not claim sparse encoder improves all Case 1 variants until Case `1c`
   reaches or exceeds the original Case `1c` reference under a controlled
   comparison.
5. Before expanding to Case `2/3`, keep Case `1c` as a diagnostic target for
   latent dimension, regularization, optimizer budget, and signal conditioning.

## Immediate Next Checks

Recommended next experiments before broad claims:

1. Reproduce Case `1a` paper quote only if needed:
   - original runner,
   - Case `1a`,
   - 100 train / 500 test if the data split is implemented,
   - BFGS `50000`,
   - report train and test metrics separately.
2. Diagnose Case `1c` sparse limitation:
   - original `TestCase_1.py --case 1c` versus sparse runner under matched
     Adam/BFGS budgets,
   - latent dim sweep around `4/5/7`,
   - BFGS `500/1000/1800` probes,
   - preserve `alpha_reg=2.7e-4` unless explicitly running a regularization
     ablation.

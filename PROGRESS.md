# LDNets Reproduction Progress

Current status on 2026-05-14: the host GPU stack is usable. Earlier "GPU blocked" notes below were caused by Codex default sandbox visibility and are superseded by the later "Sandbox vs Host GPU Access" and "Case 1 GPU Reproduction" sections. From Codex, GPU checks and training must use `sandbox_permissions=require_escalated`; otherwise `/dev/nvidia*` can be hidden even when the user shell and host GPU are healthy.

## 2026-05-14 Baseline Setup

### Repository

- Source: https://github.com/FrancescoRegazzoni/LDNets
- Local path: `/home/fzt/projects/LDNets/LDNets-repo`
- Commit: `f76e6acf75004faf266c34ab1846340626e98a0b`
- Paper: Learning the intrinsic dynamics of spatio-temporal processes through Latent Dynamics Networks, Nature Communications 2024.

The original project is compact:

- `src/utils.py`: normalization, dataset conversion, plotting, and case-specific data loaders.
- `src/optimization.py`: TensorFlow gradient handling, variable stitching, Adam loop, and SciPy BFGS wrapper.
- `src/TestCase_1a.ipynb`, `src/TestCase_1b.ipynb`, `src/TestCase_1c.ipynb`: ADR notebook experiments.
- `src/TestCase_2.py`: 2D Navier-Stokes experiment.
- `src/TestCase_3.py`: 1D electrophysiology/action-potential experiment.
- `src/TestCase_4.py`: reentry experiment.

### Server Snapshot

- OS/kernel: Ubuntu 22.04 lineage, kernel `6.8.0-111-generic`.
- CPU memory: about 251 GiB total.
- Disk available under project path: about 1.7 TiB.
- Early default-sandbox observation showed two NVIDIA GPUs in PCI output, but `nvidia-smi` failed there.
- Later host/escalated validation showed both RTX 4090 GPUs are usable; the default sandbox hid `/dev/nvidia*`.
- NVIDIA kernel module version: `595.71.05`.
- `lspci -nnk -d 10de:` shows both GPU functions bound to the `nvidia` kernel driver.

Early sandbox-only GPU symptom, later corrected:

```bash
nvidia-smi
# NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver.
```

In the default sandbox, TensorFlow 2.8 reported:

```text
Could not load dynamic library 'libcudart.so.11.0'
failed call to cuInit: CUDA_ERROR_NO_DEVICE
gpus []
```

This was not a host driver failure. Later validation from the user shell and Codex escalated commands showed `nvidia-smi` and TensorFlow GPU detection both work. Treat default-sandbox GPU failures as sandbox visibility limitations unless an escalated check also fails.

Attempted local repair:

```bash
nvidia-modprobe -c=0 -c=1
sudo -n nvidia-modprobe -c=0 -c=1
```

`nvidia-modprobe` is installed but owned/setuid as `nobody:nogroup`, so it did not create `/dev/nvidia*` in the sandbox. This was part of the early sandbox diagnosis; later escalated/host validation showed administrator driver repair was not required for this project.

### Python Environment

Created conda environment:

```bash
conda create -y -n ldnets-py39 python=3.9
conda run -n ldnets-py39 python -m pip install -r requirements.txt
conda install -y -n ldnets-py39 -c conda-forge cudatoolkit=11.2 cudnn=8.1
```

Verified package versions:

- Python `3.9.25`
- TensorFlow `2.8.0`
- NumPy `1.22.3`
- SciPy `1.7.3`
- pandas `1.4.2`
- Matplotlib `3.5.1`
- CUDA runtime `cudatoolkit 11.2.2`
- cuDNN `8.1.0.77`

Important local note: this server has ROS environment variables exported globally. For clean LDNets runs, unset ROS Python/library paths:

```bash
env -u PYTHONPATH LD_LIBRARY_PATH=/home/fzt/miniconda3/envs/ldnets-py39/lib \
  MPLCONFIGDIR=/tmp/matplotlib-ldnets \
  /home/fzt/miniconda3/envs/ldnets-py39/bin/python -m pip check
```

With those variables unset, `pip check` reports no broken requirements.

Use the explicit `LD_LIBRARY_PATH` above for TensorFlow runs. Without it, TensorFlow 2.8 does not find the conda-provided `libcudart.so.11.0`.

### Data

Full Zenodo data download started from:

```bash
curl -L --fail --continue-at - https://zenodo.org/records/10436827/files/data.zip -o data.zip
```

After download completes:

```bash
unzip data.zip -d data
```

Completed:

- `data.zip`: about 7.1 GiB.
- `data/`: about 8.0 GiB.
- Key paths verified present.

Expected case paths used by the code:

- `../data/ADR/data_1a.npy`
- `../data/ADR/data_1b.npy`
- `../data/ADR/data_1c_fmax1.0.npy`
- `../data/NS/T20_80samples.npy`
- `../data/NS/T20_20samples.npy`
- `../data/NS/T40_10samples.npy`
- `../data/AP1D/APsolution*.npy`
- `../data/AP1D/APsetting*.csv`
- `../data/reentry/sample_*.mat`

### Reproduction Entry Points

Run from `src/`, because the authors use `../data/...` relative paths:

```bash
cd /home/fzt/projects/LDNets/LDNets-repo/src
env -u PYTHONPATH LD_LIBRARY_PATH=/home/fzt/miniconda3/envs/ldnets-py39/lib \
  MPLCONFIGDIR=/tmp/matplotlib-ldnets \
  /home/fzt/miniconda3/envs/ldnets-py39/bin/python TestCase_2.py
```

Python script cases:

- `TestCase_2.py`: Adam 200 + BFGS 10000, output `TestCase2.png`.
- `TestCase_3.py`: Adam 200 + BFGS 5000, output `TestCase3.png`.
- `TestCase_4.py`: Adam 200 + BFGS 4000, output `TestCase4_sample*.png`.

Notebook cases:

- `TestCase_1a.ipynb`: ADR `data_1a.npy`, latent states 2, Adam 50 + BFGS 150.
- `TestCase_1b.ipynb`: ADR `data_1b.npy`, latent states 2, Adam 50 + BFGS 150.
- `TestCase_1c.ipynb`: ADR `data_1c_fmax1.0.npy`, latent states 4, Adam 200 + BFGS 1800.

Reference notebook outputs embedded by the author:

- TestCase 1a: NRMSE `8.576e-03`, Pearson dissimilarity `6.835e-04`.
- TestCase 1b: NRMSE `2.440e-02`, Pearson dissimilarity `1.052e-02`.
- TestCase 1c: NRMSE `2.039e-02`, Pearson dissimilarity `1.152e-02`.

### Smoke Tests

Data preprocessing smoke test passed with TestCase 1a settings on two ADR samples:

```text
samples 2 times 101 points 5 out_shape (2, 101, 5, 1)
```

Training-chain smoke test passed on CPU with a tiny LDNet:

- loaded ADR `data_1a.npy`;
- processed 2 samples and 5 subsampled points;
- built small `NNdyn` and `NNrec`;
- initialized `OptimizationProblem`;
- ran 1 Adam step and 1 BFGS step;
- final smoke loss: about `0.225`.

This was an early sandbox-only conclusion. Formal Case 1 training later ran on GPU successfully; see the Case 1 section below.

### GPU/Sandbox Validation Checklist

Formal training should use an escalated Codex command and should confirm these pass in that escalated/host context:

```bash
nvidia-smi
ls -l /dev/nvidia*
env -u PYTHONPATH LD_LIBRARY_PATH=/home/fzt/miniconda3/envs/ldnets-py39/lib \
  MPLCONFIGDIR=/tmp/matplotlib-ldnets \
  /home/fzt/miniconda3/envs/ldnets-py39/bin/python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

For TensorFlow 2.8 GPU use, the conda environment provides CUDA 11.2/cuDNN 8.1 user-space libraries. The operational blocker is Codex sandbox GPU visibility, not the host driver. If a default Codex shell reports no GPU, re-check with `sandbox_permissions=require_escalated` before changing drivers or CUDA.

### Research Modification Map

- Change latent dimension and network widths in each `TestCase_*` file around `num_latent_states`, `NNdyn`, and `NNrec`.
- Change dataset construction and normalization in `src/utils.py`.
- Change optimizer schedule, callbacks, or BFGS behavior in `src/optimization.py`.
- Change loss definitions inside each case file; cases 1, 3, and 4 use MSE plus weight regularization, while case 2 adds velocity direction loss.

## Next Steps

- Case 1a/1b/1c have been reproduced on GPU; see the Case 1 section below.
- Next practical reproduction targets are Case 2/3/4, starting from the lighter AP1D Case 3 before retrying the heavier NS Case 2.
- Keep using explicit `env -u PYTHONPATH LD_LIBRARY_PATH=... MPLCONFIGDIR=...` for every TensorFlow run.

---环境调试部分
### 🔧 Environment Conflict Fix (ROS × Conda × CUDA)
Problem Summary
Initial TensorFlow GPU failure was not due to missing CUDA, but due to environment variable pollution from ROS.
Observed symptoms:
Could not load dynamic library 'libcudart.so.11.0'
LD_LIBRARY_PATH: /opt/ros/humble/...
gpus []
Key issues:
LD_LIBRARY_PATH contained only ROS paths → CUDA libs in conda not visible
PYTHONPATH pointed to ROS Python 3.10 → conflicted with conda Python 3.9
ROS environment was injected globally (not from ~/.bashrc)
Root Cause
System-level ROS initialization (likely via /etc/profile.d/) executed:
source /opt/ros/humble/setup.bash
This introduced:
LD_LIBRARY_PATH=/opt/ros/...
PYTHONPATH=/opt/ros/...
AMENT_PREFIX_PATH=/opt/ros/...
These overrides broke:
TensorFlow CUDA dynamic linking
Python environment isolation
Fix Strategy
1. Ensure Conda CUDA has priority
Added to end of ~/.bashrc:
# Force conda priority over ROS
if [ -n "$CONDA_PREFIX" ]; then
    export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
    unset PYTHONPATH
    unset AMENT_PREFIX_PATH
    unset ROS_PACKAGE_PATH
    unset ROS_ROOT
    unset ROS_DISTRO
    unset ROS_ETC_DIR
fi
2. Resulting Environment State
echo $LD_LIBRARY_PATH
/home/fzt/miniconda3/envs/ldnets-py39/lib:...:/opt/ros/humble/...
✔ Conda CUDA libs take precedence
✔ ROS remains but no longer interferes
echo $PYTHONPATH
(empty)
✔ Python environment fully isolated
Verification
TensorFlow GPU detection now succeeds:
import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))
Output:
[GPU:0, GPU:1]
✔ CUDA runtime resolved
✔ cuDNN / cuBLAS loaded
✔ GPU devices registered
---

### 2026-05-14 Follow-up Validation

The ROS/Conda diagnosis is partially correct: global ROS variables do pollute Python and CUDA lookup paths. However, the recorded fix is not sufficient in the current shell state.

Observed after `bash -ic 'conda activate ldnets-py39; ...'`:

```text
PYTHONPATH=/opt/ros/humble/lib/python3.10/site-packages:/opt/ros/humble/local/lib/python3.10/dist-packages
LD_LIBRARY_PATH=/opt/ros/humble/opt/rviz_ogre_vendor/lib:/opt/ros/humble/lib/x86_64-linux-gnu:/opt/ros/humble/lib
```

Reason: the `.bashrc` block checks `CONDA_PREFIX` during shell startup. If `conda activate ldnets-py39` happens later, the block does not automatically re-run.

GPU status is still blocked:

```bash
nvidia-smi
# NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver.

ls -l /dev/nvidia*
# no files
```

TensorFlow still sees CPU only:

```text
tf 2.8.0
gpus []
device /job:localhost/replica:0/task:0/device:CPU:0
```

Driver-side details:

- `/proc/driver/nvidia/gpus` lists two NVIDIA GeForce RTX 4090 cards.
- Device minors are `0` and `1`.
- Kernel driver/module version remains `595.71.05`.
- `/usr/bin/nvidia-modprobe` is setuid but owned by `nobody:nogroup`, not root, so it cannot create `/dev/nvidia*`.
- `/usr/bin/nvidia-smi` is also owned by `nobody:nogroup`.

Conclusion: do not start formal LDNets GPU training yet. The remaining blocker is system-level device-node/driver access, not only ROS/Conda environment variables.

### 2026-05-14 Sandbox vs Host GPU Access

Later validation showed the previous "GPU blocked" conclusion was caused by Codex default sandbox visibility, not by the host GPU stack itself.

Host/user terminal state:

```bash
nvidia-smi
# works; reports 2x NVIDIA GeForce RTX 4090, driver 595.71.05, CUDA 13.2

echo "$LD_LIBRARY_PATH"
# /home/fzt/miniconda3/envs/ldnets-py39/lib:...:/opt/ros/humble/...

echo "$PYTHONPATH"
# empty

python test.py
# [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU'),
#  PhysicalDevice(name='/physical_device:GPU:1', device_type='GPU')]
```

Codex default sandbox state:

```bash
nvidia-smi
# fails

ls -l /dev/nvidia*
# no files visible
```

Codex outside sandbox / escalated state:

```bash
nvidia-smi
# works; same 2x RTX 4090

env -u PYTHONPATH \
  LD_LIBRARY_PATH=/home/fzt/miniconda3/envs/ldnets-py39/lib:/opt/ros/humble/opt/rviz_ogre_vendor/lib:/opt/ros/humble/lib/x86_64-linux-gnu:/opt/ros/humble/lib \
  MPLCONFIGDIR=/tmp/matplotlib-ldnets \
  /home/fzt/miniconda3/envs/ldnets-py39/bin/python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
# [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU'),
#  PhysicalDevice(name='/physical_device:GPU:1', device_type='GPU')]
```

Operational rule for this project:

- Do not diagnose GPU availability from the default Codex sandbox alone; it can hide `/dev/nvidia*`.
- For GPU verification/training from Codex, run outside the sandbox with `sandbox_permissions=require_escalated`.
- Use explicit environment variables for reproducibility:

```bash
env -u PYTHONPATH \
  LD_LIBRARY_PATH=/home/fzt/miniconda3/envs/ldnets-py39/lib:/opt/ros/humble/opt/rviz_ogre_vendor/lib:/opt/ros/humble/lib/x86_64-linux-gnu:/opt/ros/humble/lib \
  MPLCONFIGDIR=/tmp/matplotlib-ldnets \
  /home/fzt/miniconda3/envs/ldnets-py39/bin/python <script.py>
```

Training attempt note:

- `TestCase_2.py` was started outside the sandbox and TensorFlow successfully created both GPU devices.
- GPU0 held about 22.7 GiB and GPU1 about 0.5 GiB, confirming GPU access.
- The process remained at model summary/tracing setup without progressing to loss output for about 9 minutes, so it was stopped to avoid wasting GPU time.
- Next practical step is to validate with a lighter case first, preferably TestCase 1a from the ADR notebooks converted/executed carefully, before launching the heavy NS case again.

### 2026-05-14 Case 1 GPU Reproduction

Case 1a/1b/1c were converted from the notebook-only entry points into a reusable script:

- Script: `src/TestCase_1.py`
- Outputs: `runs/case1/`
- Metrics summary: `runs/case1/summary_metrics.json`
- Figures: `runs/case1/TestCase_1{a,b,c}_loss.png` and `runs/case1/TestCase_1{a,b,c}_comparison.png`

The script preserves the notebook model definitions, dataset splits, random seed, Adam/BFGS schedules, and reference metrics. It adds CLI controls for reproducibility:

```bash
/home/fzt/miniconda3/envs/ldnets-py39/bin/python src/TestCase_1.py --help
# --case 1a|1b|1c|all
# --adam-epochs
# --bfgs-epochs
# --seed
# --batch-samples
```

Codex GPU command pattern used for formal runs:

```bash
env -u PYTHONPATH \
  CUDA_VISIBLE_DEVICES=0 \
  LD_LIBRARY_PATH=/home/fzt/miniconda3/envs/ldnets-py39/lib:/opt/ros/humble/opt/rviz_ogre_vendor/lib:/opt/ros/humble/lib/x86_64-linux-gnu:/opt/ros/humble/lib \
  MPLCONFIGDIR=/tmp/matplotlib-ldnets \
  /home/fzt/miniconda3/envs/ldnets-py39/bin/python src/TestCase_1.py \
  --case 1a --output-dir runs/case1
```

For 1b and 1c the same command was used with `--case 1b` / `--case 1c` and `--batch-samples 25`.

Important runtime finding:

- Case 1a runs as a full batch on GPU and matches the notebook reference.
- Case 1b initially failed on GPU with TensorFlow 2.8/CUDA 11.2/cuBLAS on RTX 4090:

```text
Blas xGEMV launch failed : a.shape=[1,2010000,7], b.shape=[1,7,1]
```

- The failure was avoided by computing the same full-data MSE in sample chunks via `--batch-samples 25`. This reduces the largest per-kernel Dense input while keeping the training/validation objective as the weighted mean over all samples.
- Case 1b and Case 1c then completed on GPU0. During chunked runs `nvidia-smi` showed the TensorFlow process on GPU0 using about 1.8 GiB and nonzero GPU utilization. GPU memory returned to idle after training.

Formal Case 1 metrics:

| Case | Adam + BFGS | Batch samples | NRMSE | Reference NRMSE | Pearson dissimilarity | Reference Pearson dissimilarity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1a | 50 + 150 | full | `8.577e-03` | `8.576e-03` | `6.836e-04` | `6.835e-04` |
| 1b | 50 + 150 | 25 | `2.439e-02` | `2.440e-02` | `1.051e-02` | `1.052e-02` |
| 1c | 200 + 1800 | 25 | `2.059e-02` | `2.039e-02` | `1.175e-02` | `1.152e-02` |

Conclusion: Case 1 reproduction is complete on the current server GPU setup. For later research changes, start from `src/TestCase_1.py` rather than the notebooks when running noninteractive experiments.

## 2026-05-14 Case 2/3 GPU Reproduction

Case 2 and Case 3 were converted from fixed author scripts into reusable command-line runners while preserving the author defaults. The scripts now write outputs under `runs/` instead of overwriting `src/TestCase2.png` or `src/TestCase3.png`.

Updated scripts:

- `src/TestCase_2.py`: Navier-Stokes 2D case.
- `src/TestCase_3.py`: AP1D electrophysiology case.

Both scripts configure TensorFlow `float64`, set GPU memory growth, use repository-root data paths, save loss/comparison figures, and write metrics JSON.

### Command Template

Use this pattern from the repository root:

```bash
env -u PYTHONPATH \
  CUDA_VISIBLE_DEVICES=0 \
  LD_LIBRARY_PATH=/home/fzt/miniconda3/envs/ldnets-py39/lib:/opt/ros/humble/opt/rviz_ogre_vendor/lib:/opt/ros/humble/lib/x86_64-linux-gnu:/opt/ros/humble/lib \
  MPLCONFIGDIR=/tmp/matplotlib-ldnets \
  /home/fzt/miniconda3/envs/ldnets-py39/bin/python src/TestCase_3.py \
  --bfgs-epochs 500 \
  --eval-batch-samples 10 \
  --output-dir runs/case3/bfgs500_chunked
```

Case 2 command used:

```bash
env -u PYTHONPATH \
  CUDA_VISIBLE_DEVICES=1 \
  LD_LIBRARY_PATH=/home/fzt/miniconda3/envs/ldnets-py39/lib:/opt/ros/humble/opt/rviz_ogre_vendor/lib:/opt/ros/humble/lib/x86_64-linux-gnu:/opt/ros/humble/lib \
  MPLCONFIGDIR=/tmp/matplotlib-ldnets \
  /home/fzt/miniconda3/envs/ldnets-py39/bin/python src/TestCase_2.py \
  --bfgs-epochs 500 \
  --eval-point-batch 500 \
  --output-dir runs/case2/bfgs500_chunked
```

### Case 3 Result

Configuration:

- Dataset split: train `0:100`, valid `100:200`, test `200:400`.
- Author default model/training settings: `dt=1`, `dt_base=205`, `num_latent_states=12`, dynamics width `8`, reconstruction width `17`, Adam `200`, learning rate `1e-2`, train/valid `20` spatial points per time step.
- Controlled run used `BFGS 500` instead of author default `5000` for this stage.
- Test set was evaluated without point downsampling, using `--eval-batch-samples 10` to avoid an RTX 4090 + TensorFlow 2.8/cuBLAS xGEMV launch failure on one huge Dense call.

Outputs:

- `runs/case3/bfgs500_chunked/TestCase_3_metrics.json`
- `runs/case3/bfgs500_chunked/TestCase_3_loss.png`
- `runs/case3/bfgs500_chunked/TestCase_3_comparison.png`

Metrics:

| Case | Adam + BFGS | Eval chunk | NRMSE | Pearson dissimilarity | Final train loss | Final valid loss | Time |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 3 | 200 + 500 | 10 samples | `2.096e-02` | `2.339e-03` | `2.382e-03` | `5.472e-04` | `479.8 s` |

Failed attempts before chunked evaluation:

- Full Case 3 with author-style test evaluation reached BFGS training but was manually stopped once the old runner was known to be unsuitable.
- `Adam 200 + BFGS 500` completed training but failed at full test Dense evaluation:

```text
Blas xGEMV launch failed : a.shape=[1,10120200,17]
inputs=tf.Tensor(shape=(200, 501, 101, 17), dtype=float64)
```

- Reducing `--test-points 20` in the old evaluation path still failed:

```text
Blas xGEMV launch failed : a.shape=[1,2004000,17]
inputs=tf.Tensor(shape=(200, 501, 20, 17), dtype=float64)
```

Conclusion: for Case 3 on this server, keep the training objective unchanged and use `--eval-batch-samples` for test prediction/metrics.

### Case 2 Result

Configuration:

- Dataset split: train `T20_80samples.npy` first 80 samples, valid `T20_20samples.npy` first 20, test `T40_10samples.npy` first 10.
- Author default model/training settings: `dt=0.2`, `dt_base=5.4`, `num_latent_states=1`, dynamics width `7`, reconstruction width `24`, Adam `200`, learning rate `1e-2`, train/valid `200` spatial points per time step.
- Controlled run used `BFGS 500` instead of author default `10000`.
- Test set was evaluated on the full grid, using `--eval-point-batch 500` to avoid very large Dense calls during post-training prediction.

Outputs:

- `runs/case2/bfgs500_chunked/TestCase_2_metrics.json`
- `runs/case2/bfgs500_chunked/TestCase_2_loss.png`
- `runs/case2/bfgs500_chunked/TestCase_2_comparison.png`

Metrics:

| Case | Adam + BFGS | Eval chunk | NRMSE | Pearson dissimilarity | Final train loss | Final valid loss | Time |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 200 + 500 | 500 points | `2.030e-02` | `9.335e-02` | `3.945e-03` | `4.618e-03` | `468.1 s` |

Case 2 smoke test also passed:

```bash
env -u PYTHONPATH CUDA_VISIBLE_DEVICES=1 LD_LIBRARY_PATH=... MPLCONFIGDIR=/tmp/matplotlib-ldnets \
  /home/fzt/miniconda3/envs/ldnets-py39/bin/python src/TestCase_2.py \
  --train-samples 2 --valid-samples 2 --test-samples 1 \
  --train-points 50 --valid-points 50 --test-points 100 \
  --adam-epochs 1 --bfgs-epochs 0 --skip-figures \
  --output-dir runs/case2_smoke
```

Smoke output: NRMSE `9.188e-02`, Pearson dissimilarity `1.135e+00`. This validates the NS data and GPU training path only; it is not a reproduction metric.

### Adjustable Parameters

Common training controls:

- `--adam-epochs`: number of Adam steps. Author defaults are `200` for Case 2 and Case 3. Use `1` to `5` for smoke tests.
- `--bfgs-epochs`: SciPy BFGS `maxiter`. Author defaults are Case 2 `10000`, Case 3 `5000`. Current controlled reproduction used `500` to keep runtime bounded.
- `--learning-rate`: Adam learning rate, default `1e-2`.
- `--seed`: NumPy and TensorFlow seed, default `0`. Keep this fixed when comparing model changes.
- `CUDA_VISIBLE_DEVICES`: select GPU. Use one GPU per process to avoid TensorFlow reserving both 4090s.

Case 2 controls:

- `--train-samples`, `--valid-samples`, `--test-samples`: sample counts, defaults `80/20/10`.
- `--train-points`, `--valid-points`: random spatial points per time step for training/validation, defaults `200/200`. Reducing these speeds training but changes the objective.
- `--test-points`: optional test downsampling. Leave unset for full-grid metrics.
- `--eval-point-batch`: point chunk size for test prediction only, default `1000`; current run used `500`.
- `--dt`, `--dt-base`: time resampling and normalization. Defaults `0.2/5.4`; changing these changes the dynamical discretization.
- `--num-latent-states`, `--dynamics-width`, `--reconstruction-width`: LDNet capacity controls.
- `--weight-direction`, `--epsilon`: direction-loss weight and normalization epsilon.

Case 3 controls:

- `--train-start`, `--train-samples`, `--valid-start`, `--valid-samples`, `--test-start`, `--test-samples`: AP1D sample ranges; defaults reproduce the author split.
- `--points-subsampling-rate`: AP loader grid stride, default `8`, giving 101 points before training subsampling.
- `--time-steps`: number of raw time steps loaded, default `501`.
- `--train-points`, `--valid-points`: random spatial points per time step for training/validation, defaults `20/20`.
- `--test-points`: optional test downsampling. Leave unset for full test metrics.
- `--eval-batch-samples`: sample chunk size for test prediction only, default `25`; current run used `10`.
- `--alpha-reg`: kernel regularization weight, default `4.7e-3`.
- `--num-latent-states`, `--dynamics-width`, `--reconstruction-width`: LDNet capacity controls.

### Interpretation

The current Case 2/3 runs are controlled, reproducible GPU runs, not full author-budget runs. They keep the model, data split, Adam stage, and train/valid sampling defaults, but reduce BFGS from the paper scripts (`10000`/`5000`) to `500`. For publication-grade reproduction, rerun the same commands with the author BFGS defaults and keep chunked evaluation enabled.

## 2026-05-16 Stage 1 JEPA-LDNet Implementation

Stage 1 has started from the `refer.md`/`AGENT.md` direction: convert LDNet from a fixed-zero-latent reconstruction surrogate into a sparse-observation, latent-rollout, JEPA-constrained scientific world model.

### Code Added

- `src/models.py`
  - `PointSetEncoder`: DeepSets-style encoder for sparse coordinate/value observations.
  - `LatentTransition`: Euler transition module matching the original `NNdyn` structure.
  - `ContinuousDecoder`: meshless coordinate-query decoder matching the original `NNrec` role.
  - `JEPAPredictor`: predicts target embeddings from rollout latent states.
  - `JEPALDNet`: wires `E_phi`, `T_theta`, `D_omega`, `E_bar`, and `P_psi`.
- `src/losses.py`
  - reconstruction MSE, JEPA feature MSE with stop-gradient target, latent smoothness, and kernel L2 helpers.
- `src/metrics.py`
  - denormalized NRMSE, Pearson dissimilarity, horizon-wise NRMSE, and parameter count.
- `src/TestCase_1_jepa.py`
  - first Stage 1 runner for ADR Case `1a/1b/1c`.
  - supports sparse sensors, context steps, target point sampling, JEPA warmup/ramp, EMA target encoder, metrics/config JSON, and figures.
- `docs/stage1_jepa_image2_prompts.md`
  - image-2 prompts and commands for the final architecture/training flow figures.
- `docs/stage1_jepa_image2_architecture_prompt.txt`
- `docs/stage1_jepa_image2_training_prompt.txt`
  - direct prompt files for the image-2 CLI `--prompt-file` commands.
- `docs/figures/make_stage1_diagrams.py`
  - deterministic local fallback generator for the architecture and training-flow figures.

### Algorithm Interfaces

Core tensors:

- `X_obs: [B,Tc,No,dx]`
- `Y_obs: [B,Tc,No,dy]`
- context features in the runner: `[x, t, y] -> [B,Tc*No,dx+1+dy]`
- `z0: [B,dz]`
- rollout states: `z_1:T: [B,T,dz]`
- full/query coordinates: `X_query: [B,T,Nq,dx]`
- predicted field: `Y_hat: [B,T,Nq,dy]`
- target patch features: `[B,K,Np,dx+1+dy]`
- target embedding: `h_target: [B,K,dh]`
- predicted embedding: `h_pred: [B,K,dh]`

Loss:

```text
L = lambda_rec * MSE(Y_hat, Y)
  + lambda_jepa * MSE(P_psi(z), stopgrad(E_bar(target)))
  + lambda_smooth * mean(||z_{t+1} - z_t||^2)
```

`lambda_dyn` is intentionally set to `0.0` in the first implementation because comparing a transition output against the same rollout transition would be tautological. A nonzero dynamics loss should only be added after implementing per-time sensor latent targets.

### Diagrams

Generated local fallback diagrams:

- `docs/figures/stage1_jepa_ldnet_architecture.png`
- `docs/figures/stage1_jepa_ldnet_architecture.svg`
- `docs/figures/stage1_jepa_ldnet_training_flow.png`
- `docs/figures/stage1_jepa_ldnet_training_flow.svg`

The actual image-2 generation path is documented in `docs/stage1_jepa_image2_prompts.md`, but was not executed because `OPENAI_API_KEY` is not set in the current Codex environment.

### 2026-05-17 [SMOKE] Stage 1 JEPA-LDNet Case 1a

- Commit: `e2ece79`
- Git status at run time:

```text
M .gitignore
M PROGRESS.md
M refer.md
?? AGENT.md
?? docs/
?? src/TestCase_1_jepa.py
?? src/losses.py
?? src/metrics.py
?? src/models.py
```

- Agent/role: Codex execution with worker/reviewer subagents.
- Start/end time: `2026-05-17 00:10:18 +0800` to `2026-05-17 00:10:30 +0800`
- GPU: `CUDA_VISIBLE_DEVICES=0`, TensorFlow physical GPU `/physical_device:GPU:0`
- Command:

```bash
env -u PYTHONPATH \
  CUDA_VISIBLE_DEVICES=0 \
  LD_LIBRARY_PATH=/home/fzt/miniconda3/envs/ldnets-py39/lib:/opt/ros/humble/opt/rviz_ogre_vendor/lib:/opt/ros/humble/lib/x86_64-linux-gnu:/opt/ros/humble/lib \
  MPLCONFIGDIR=/tmp/matplotlib-ldnets \
  /home/fzt/miniconda3/envs/ldnets-py39/bin/python src/TestCase_1_jepa.py \
  --case 1a \
  --adam-epochs 5 \
  --bfgs-epochs 0 \
  --seed 0 \
  --sensor-ratio 0.2 \
  --batch-samples 25 \
  --warmup-epochs 0 \
  --jepa-ramp-epochs 1 \
  --lambda-rec 1.0 \
  --lambda-jepa 0.1 \
  --lambda-smooth 1e-4 \
  --ema-decay 0.99 \
  --output-dir runs/jepa/case1a/smoke5_seed0_sr020
```

- Output dir: `runs/jepa/case1a/smoke5_seed0_sr020`
- Dataset split: ADR Case 1a original split, train `0:100`, valid `100:200`, test `200:300`.
- Model:
  - latent dimension `2`
  - embedding dimension `32`
  - condition dimension `3`
  - trainable parameters `8226`
- JEPA config:
  - sensor ratio `0.2`, fixed by seed `0`
  - context steps `1`
  - prediction horizon `1`
  - target points `32`
  - `lambda_rec=1.0`
  - `lambda_jepa=0.1`
  - `lambda_dyn=0.0`
  - `lambda_smooth=1e-4`
  - warmup `0`, JEPA ramp `1`
  - EMA decay `0.99`
- Training budget: Adam `5`, BFGS `0`. This validates the Stage 1 code path only; it is not a formal reproduction or improvement result.
- Evaluation mode:
  - denormalized full-field metrics
  - `--batch-samples 25` is equivalent full-data loss/prediction chunking by sample count, not sample subsampling.
- Metrics:
  - full-field NRMSE: `2.412e-01`
  - Pearson dissimilarity: `9.166e-01`
  - few-sensor NRMSE: `2.563e-01`
  - validation JEPA feature MSE: `1.465e-02`
  - validation latent smoothness: `4.367e-03`
  - validation reconstruction MSE: `2.589e-01`
  - final train loss: `2.372e-01`
  - final valid loss: `2.603e-01`
  - runtime: `11.9 s`
  - peak TensorFlow GPU memory: `492168448` bytes
  - parameter count: `8226`
- Artifacts:
  - `runs/jepa/case1a/smoke5_seed0_sr020/config.json`
  - `runs/jepa/case1a/smoke5_seed0_sr020/metrics.json`
  - `runs/jepa/case1a/smoke5_seed0_sr020/loss.png`
  - `runs/jepa/case1a/smoke5_seed0_sr020/comparison.png`
- Interpretation:
  - GPU execution, sparse sensor encoding, JEPA target encoding, EMA update, full-field decoding, metrics, and figure output all work.
  - Error is high because this is only 5 Adam epochs. Do not compare this smoke metric against original LDNet reproduction metrics.
- Next action:
  - Run Case 1a formal Stage 1 screening with `Adam 200`, `BFGS 0`, sensor ratios `1.0/0.5/0.2/0.1/0.05`, warmup `20`, JEPA ramp `80`.

### 2026-05-17 [FORMAL] Stage 1 JEPA-LDNet Case 1a Sensor Sweep

- Goal: formal sparse-sensor screening for ADR Case `1a`.
- Agent workflow:
  - Worker agent prepared/executed duplicate verification runs and then stopped after main sweep completion.
  - Reviewer agent checked that `src/TestCase_1_jepa.py` supports the requested controls and that Case 1a sensor ratios map to `100/50/20/10/5` sensors.
- GPU execution:
  - Physical GPUs were selected with `CUDA_VISIBLE_DEVICES=0/1`.
  - TensorFlow records the visible card as `/physical_device:GPU:0` inside each process, so `config.json` is not a physical GPU id.
  - `nvidia-smi` after completion showed no remaining training processes and both RTX 4090 cards idle.
- Commit recorded in run configs: `e2ece79`.
- Formal output root: `runs/jepa/case1a/formal_sensor_sweep`.
- Formal result directories use the suffix `_adam200`; duplicate verification directories without that suffix are not part of the main result table.

Command template:

```bash
env -u PYTHONPATH \
  CUDA_VISIBLE_DEVICES=<0-or-1> \
  LD_LIBRARY_PATH=/home/fzt/miniconda3/envs/ldnets-py39/lib:/opt/ros/humble/opt/rviz_ogre_vendor/lib:/opt/ros/humble/lib/x86_64-linux-gnu:/opt/ros/humble/lib \
  MPLCONFIGDIR=/tmp/matplotlib-ldnets \
  /home/fzt/miniconda3/envs/ldnets-py39/bin/python src/TestCase_1_jepa.py \
  --case 1a \
  --adam-epochs 200 \
  --bfgs-epochs 0 \
  --seed 0 \
  --sensor-ratio <ratio> \
  --batch-samples 25 \
  --warmup-epochs 20 \
  --jepa-ramp-epochs 80 \
  --lambda-rec 1.0 \
  --lambda-jepa 0.1 \
  --lambda-smooth 1e-4 \
  --ema-decay 0.99 \
  --output-dir runs/jepa/case1a/formal_sensor_sweep/<run-name>
```

Fixed settings:

- Dataset split: ADR Case 1a original split, train `0:100`, valid `100:200`, test `200:300`.
- Optimizer budget: Adam `200`, BFGS `0`.
- JEPA schedule: warmup `20`, ramp `80`.
- Loss weights: `lambda_rec=1.0`, `lambda_jepa=0.1`, `lambda_dyn=0.0`, `lambda_smooth=1e-4`.
- EMA decay: `0.99`.
- `--batch-samples 25` is chunking only; it does not subsample the dataset.

Formal results:

| Run | Ratio | Sensors | NRMSE | Pearson dissim. | Sensor NRMSE | Valid loss | Valid rec MSE | Valid JEPA MSE | Runtime |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `sr100_seed0_adam200` | `1.00` | `100` | `2.403e-02` | `5.385e-03` | `2.403e-02` | `2.892e-03` | `2.868e-03` | `2.454e-04` | `75.3 s` |
| `sr050_seed0_adam200` | `0.50` | `50` | `3.211e-02` | `9.630e-03` | `3.231e-02` | `4.967e-03` | `4.933e-03` | `3.375e-04` | `82.6 s` |
| `sr020_seed0_adam200` | `0.20` | `20` | `3.139e-02` | `9.202e-03` | `3.148e-02` | `4.645e-03` | `4.641e-03` | `4.454e-05` | `79.7 s` |
| `sr010_seed0_adam200` | `0.10` | `10` | `3.240e-02` | `9.791e-03` | `3.437e-02` | `4.872e-03` | `4.867e-03` | `4.859e-05` | `72.8 s` |
| `sr005_seed0_adam200` | `0.05` | `5` | `3.060e-02` | `8.748e-03` | `3.332e-02` | `4.363e-03` | `4.355e-03` | `7.787e-05` | `65.7 s` |

Per-run artifacts:

- `config.json`: command, arguments, start/end time, git commit/status, visible GPU devices, sensor indices, target indices.
- `metrics.json`: NRMSE, Pearson dissimilarity, sensor NRMSE, validation loss components, elapsed seconds, TensorFlow GPU memory info, parameter count.
- `loss.png`
- `comparison.png`

Reviewer final check:

- The five formal `_adam200` directories all contain `config.json` and `metrics.json`.
- For each formal run, `metrics.json.config` matches the same directory's `config.json`.
- Case, seed, ratio, sensor count, Adam epochs, BFGS epochs, warmup, and ramp all match the requested sweep.
- `git_status_short` is non-empty in the configs because this stage was run from a dirty worktree containing uncommitted Stage 1 implementation/docs. Keep this context when comparing future clean-commit reruns.
- The runner does not yet write an explicit `run_status` field. Completion is inferred from existing `metrics.json`, `end_time`, and complete metric fields.

Duplicate verification directories generated by the worker agent:

- `runs/jepa/case1a/formal_sensor_sweep/sr100_seed0`
- `runs/jepa/case1a/formal_sensor_sweep/sr050_seed0`
- `runs/jepa/case1a/formal_sensor_sweep/sr020_seed0`

These duplicate metrics match the same configuration family and are useful as sanity checks, but the main formal table above uses only the five `_adam200` directories for naming consistency.

Interpretation:

- Full sensors give the best Case 1a result in this sweep: NRMSE `2.403e-02`.
- Sparse sensors remain stable: `5%` sensors still reached NRMSE `3.060e-02`, only modestly worse than `50%/20%/10%`.
- The non-monotonic sparse trend (`5%` slightly better than `10%`) is plausible with a single seed and random sensor placement; do not over-interpret it without multi-seed repeats.
- Next experimental step should be a multi-seed sensor-placement repeat for `0.05/0.10/0.20`, or a direct comparison against original LDNet under identical sensor subsets.

### 2026-05-17 [FORMAL] Stage 1 JEPA-LDNet Remaining Case 1 Experiments

Goal: finish the Stage 1 JEPA evidence chain after the first Case `1a` sensor sweep:

- Case `1a` sparse sensor multi-seed repeat for ratios `0.20/0.10/0.05`.
- Case `1a` formal ablation at ratio `0.20`.
- Case `1b/1c` transfer sweeps at ratios `1.00/0.20/0.05`.

Agent workflow:

- Worker agent executed the training matrix on the two RTX 4090 GPUs.
- Reviewer agent first audited the command plan, then audited the completed run artifacts.
- Main process independently validated all 16 new formal runs with a JSON consistency script.

Runner and summary tooling updates:

- `src/TestCase_1_jepa.py` now writes:
  - `run_status=completed`
  - `command_hash`
  - `elapsed_seconds` in `config.json`
  - `cuda_visible_devices`
  - a note explaining TensorFlow visible GPU remapping.
- Added `scripts/summarize_jepa_runs.py` for CSV/Markdown summaries from `metrics.json` and `config.json`.
- The summary script supports `--include-run-regex` and `--exclude-run-regex`.
- `runs/jepa/case1a/formal_sensor_sweep/summary.md|csv` was regenerated with `--include-run-regex '_adam200$'`, so it now excludes the duplicate verification directories without `_adam200`.

Smoke checks before formal runs:

- `runs/jepa/case1b/smoke5_seed0_sr020`
  - `Adam 5`, `BFGS 0`, ratio `0.20`, full JEPA.
  - NRMSE `1.698e-01`, Pearson dissimilarity `1.028e+00`, sensor NRMSE `1.725e-01`.
  - Verified Case `1b` data path, GPU execution, new `run_status`, `command_hash`, and `cuda_visible_devices`.
- `runs/jepa/case1a/ablation_smoke/no_jepa_seed0_sr020`
  - `Adam 5`, `BFGS 0`, ratio `0.20`, `lambda_jepa=0.0`.
  - NRMSE `2.410e-01`, Pearson dissimilarity `9.216e-01`, sensor NRMSE `2.570e-01`.
  - TensorFlow emitted expected no-gradient warnings for predictor/JEPA-only variables because the JEPA loss branch was disabled.

Formal training settings:

```bash
env -u PYTHONPATH \
  CUDA_VISIBLE_DEVICES=<0-or-1> \
  LD_LIBRARY_PATH=/home/fzt/miniconda3/envs/ldnets-py39/lib:/opt/ros/humble/opt/rviz_ogre_vendor/lib:/opt/ros/humble/lib/x86_64-linux-gnu:/opt/ros/humble/lib \
  MPLCONFIGDIR=/tmp/matplotlib-ldnets \
  /home/fzt/miniconda3/envs/ldnets-py39/bin/python src/TestCase_1_jepa.py \
  --case <1a|1b|1c> \
  --adam-epochs 200 \
  --bfgs-epochs 0 \
  --batch-samples 25 \
  --warmup-epochs 20 \
  --jepa-ramp-epochs 80 \
  --lambda-rec 1.0 \
  --learning-rate 1e-2 \
  --seed <seed> \
  --sensor-ratio <ratio> \
  --lambda-jepa <value> \
  --lambda-smooth <value> \
  --ema-decay <value> \
  --output-dir <run-dir>
```

New formal run count: `16`.

- Case `1a` multi-seed: `6` new runs for ratios `0.20/0.10/0.05`, seeds `1/2`.
- Case `1a` ablation: `4` new runs at ratio `0.20`, seed `0`.
- Case `1b` transfer: `3` new runs, seed `0`.
- Case `1c` transfer: `3` new runs, seed `0`.
- Reused Case `1a` sparse seed `0` runs from the previous formal sweep:
  - `runs/jepa/case1a/formal_sensor_sweep/sr020_seed0_adam200`
  - `runs/jepa/case1a/formal_sensor_sweep/sr010_seed0_adam200`
  - `runs/jepa/case1a/formal_sensor_sweep/sr005_seed0_adam200`

Summary artifacts:

- `runs/jepa/case1a/multiseed_sensor_sweep/summary.md`
- `runs/jepa/case1a/multiseed_sensor_sweep/summary.csv`
- `runs/jepa/case1a/multiseed_sensor_sweep/summary_with_seed0.md`
- `runs/jepa/case1a/multiseed_sensor_sweep/summary_with_seed0.csv`
- `runs/jepa/case1a/ablation/summary.md`
- `runs/jepa/case1a/ablation/summary.csv`
- `runs/jepa/case1b/formal_sensor_sweep/summary.md`
- `runs/jepa/case1b/formal_sensor_sweep/summary.csv`
- `runs/jepa/case1c/formal_sensor_sweep/summary.md`
- `runs/jepa/case1c/formal_sensor_sweep/summary.csv`

Case `1a` multi-seed sparse aggregate, including reused seed `0`:

| Ratio | Sensors | Seeds | NRMSE mean | NRMSE std | Pearson dissim. mean | Pearson dissim. std | Sensor NRMSE mean | Sensor NRMSE std |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `0.20` | `20` | `0/1/2` | `2.403e-02` | `6.383e-03` | `5.639e-03` | `3.087e-03` | `2.401e-02` | `6.502e-03` |
| `0.10` | `10` | `0/1/2` | `2.992e-02` | `6.944e-03` | `7.944e-03` | `2.984e-03` | `3.080e-02` | `7.632e-03` |
| `0.05` | `5` | `0/1/2` | `2.430e-02` | `5.468e-03` | `5.686e-03` | `2.655e-03` | `2.617e-02` | `6.210e-03` |

Case `1a` formal ablation at ratio `0.20`, seed `0`:

| Run | `lambda_jepa` | `lambda_smooth` | `ema_decay` | NRMSE | Pearson dissim. | Sensor NRMSE | Runtime |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `full_jepa_sr020_seed0_adam200` | `0.1` | `1e-4` | `0.99` | `3.139e-02` | `9.202e-03` | `3.148e-02` | `68.0 s` |
| `no_jepa_sr020_seed0_adam200` | `0.0` | `1e-4` | `0.99` | `3.084e-02` | `8.880e-03` | `3.136e-02` | `68.6 s` |
| `no_smooth_sr020_seed0_adam200` | `0.1` | `0.0` | `0.99` | `3.110e-02` | `9.032e-03` | `3.157e-02` | `66.3 s` |
| `no_ema_lag_sr020_seed0_adam200` | `0.1` | `1e-4` | `0.0` | `3.086e-02` | `8.892e-03` | `3.147e-02` | `66.5 s` |

Case `1b/1c` transfer sweep:

| Case | Ratio | Sensors | NRMSE | Pearson dissim. | Sensor NRMSE | Runtime |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `1b` | `1.00` | `100` | `6.746e-02` | `8.345e-02` | `6.746e-02` | `121.2 s` |
| `1b` | `0.20` | `20` | `6.756e-02` | `8.370e-02` | `7.023e-02` | `119.3 s` |
| `1b` | `0.05` | `5` | `6.741e-02` | `8.331e-02` | `6.160e-02` | `128.1 s` |
| `1c` | `1.00` | `100` | `7.656e-02` | `1.760e-01` | `7.656e-02` | `126.4 s` |
| `1c` | `0.20` | `20` | `8.180e-02` | `2.053e-01` | `8.189e-02` | `124.6 s` |
| `1c` | `0.05` | `5` | `7.701e-02` | `1.801e-01` | `7.779e-02` | `133.9 s` |

Post-training audit:

- All 16 new formal runs contain `config.json` and `metrics.json`.
- All new formal runs have `run_status=completed`.
- For all reviewed runs, `metrics.json.config` matches the same directory's `config.json`.
- Case, seed, sensor ratio, sensor count, Adam/BFGS epochs, warmup/ramp, loss weights, and EMA decay match the intended experiment matrix.
- Key metrics are finite for all reviewed runs.
- `nvidia-smi` after completion showed both RTX 4090 cards idle with only display processes using memory.

Interpretation:

- Case `1a` sparse multi-seed is stable. Even `5%` sensors gives aggregate NRMSE `2.430e-02 +/- 5.468e-03`.
- The Case `1a` ablation at ratio `0.20`, seed `0`, does not show a clear full-JEPA advantage. `no_jepa` and `no_ema_lag` are slightly better on this single point. Treat this as a warning that the current JEPA loss may be regularizing weakly or redundantly for simple ADR Case `1a`.
- Case `1b/1c` transfer runs completed and are numerically stable, but their NRMSE is much higher than the recorded original LDNet Case `1b/1c` baselines. This is transfer-success evidence, not positive accuracy evidence.
- Next method-improvement step should focus on making the JEPA target less redundant with reconstruction, for example masked/patch targets, longer prediction horizon, or applying JEPA on harder temporal cases before claiming accuracy gains.

Provenance notes:

- New formal runs record git commit `4a4af28` with a dirty worktree because runner/summary documentation improvements were active during execution.
- Reused Case `1a` seed `0` sparse runs record git commit `e2ece79` with a dirty worktree.
- `refer.md` remains a pre-existing user/reference change and was not modified in this stage.

### 2026-05-19 [FORMAL] Stage 1 Fair Baseline and Masked Target Audit

Goal: check whether the JEPA world-model changes have a fair optimizer-budget advantage and whether masked JEPA targets improve over the current point-target JEPA and no-JEPA sparse encoder baselines.

Code updates:

- `src/TestCase_1_jepa.py`
  - added `--target-mode {points,masked-points,future-patches}`.
  - added `--mask-ratio`.
  - added `--target-time-strategy {next,random-future,horizon}`.
  - added `target_selection` to `config.json`.
  - default behavior remains backward-compatible: `points` + `next`.
- `scripts/summarize_jepa_runs.py`
  - now records `target_mode`, `mask_ratio`, and `target_time_strategy`.
  - default aggregate grouping includes the target fields.
  - legacy JEPA configs without `target_selection` are interpreted as `points / next`; `mask_ratio` is blank for `points` mode because it is ignored.
- `src/TestCase_1.py`
  - added provenance fields to result JSON: `run_status`, `algorithm`, `elapsed_seconds`, `command_hash`, `git_commit`, `git_status_short`, and `cuda_visible_devices`.
  - training objective, optimizer calls, data chunking, and evaluation path are unchanged.

GPU and agent workflow:

- Worker agent executed the training matrix.
- Reviewer agent audited the code changes, command matrix, result JSON, and summaries.
- GPU0 had other user processes under `/home/wat/miniconda3/envs/marl_cm/bin/python`, so formal runs used only `CUDA_VISIBLE_DEVICES=1` sequentially.
- Final `nvidia-smi` showed GPU1 idle; GPU0 still had the unrelated user workload.

Smoke validation:

- `runs/jepa/case1a/masked_smoke/sr020_seed0_adam5`
  - Case `1a`, ratio `0.20`, seed `0`, `Adam 5`, `BFGS 0`.
  - `target-mode=masked-points`, `mask-ratio=0.5`, `target-time-strategy=random-future`, `prediction-horizon=2`.
  - NRMSE `2.441e-01`, Pearson dissimilarity `9.157e-01`, sensor NRMSE `2.567e-01`.
  - This is a code-path smoke only, not a formal result.

Formal outputs:

- Original LDNet same-budget baseline:
  - `runs/baseline/case1/adam200_bfgs0_seed0`
- No-JEPA sparse encoder baseline:
  - `runs/jepa/case1a/no_jepa_sparse_fair`
- Masked-points JEPA sweep:
  - `runs/jepa/case1a/masked_points_sweep`
- Combined fair comparison:
  - `runs/jepa/case1a/fair_jepa_comparison/summary.md`
  - `runs/jepa/case1a/fair_jepa_comparison/summary.csv`

Original LDNet same-budget baseline:

| Case | Adam + BFGS | NRMSE | Pearson dissim. | Runtime |
| --- | ---: | ---: | ---: | ---: |
| `1a` | `200 + 0` | `7.267e-02` | `4.976e-02` | `43.5 s` |
| `1b` | `200 + 0` | `6.924e-02` | `8.822e-02` | `78.7 s` |
| `1c` | `200 + 0` | `8.883e-02` | `2.489e-01` | `77.1 s` |

No-JEPA sparse encoder baseline, Case `1a`:

| Ratio | Sensors | Seeds | NRMSE mean | NRMSE std | Pearson dissim. mean | Sensor NRMSE mean |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `0.20` | `20` | `0/1/2` | `2.386e-02` | `6.049e-03` | `5.536e-03` | `2.395e-02` |
| `0.10` | `10` | `0/1/2` | `2.634e-02` | `5.873e-03` | `6.552e-03` | `2.647e-02` |
| `0.05` | `5` | `0/1/2` | `2.455e-02` | `6.352e-03` | `5.868e-03` | `2.687e-02` |

Current point-target JEPA vs masked-points JEPA, Case `1a`:

| Family | Ratio | Seeds | NRMSE mean | NRMSE std | Pearson dissim. mean | Sensor NRMSE mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Current JEPA points | `0.20` | `0/1/2` | `2.403e-02` | `6.383e-03` | `5.639e-03` | `2.401e-02` |
| Masked-points JEPA | `0.20` | `0/1/2` | `2.489e-02` | `5.417e-03` | `5.729e-03` | `2.467e-02` |
| Current JEPA points | `0.05` | `0/1/2` | `2.430e-02` | `5.468e-03` | `5.686e-03` | `2.617e-02` |
| Masked-points JEPA | `0.05` | `0/1/2` | `2.409e-02` | `5.591e-03` | `5.611e-03` | `2.559e-02` |

Post-training audit:

- `runs/baseline/case1/adam200_bfgs0_seed0` contains `TestCase_1a/1b/1c_metrics.json` and `summary_metrics.json`.
- No-JEPA sparse baseline contains `9/9` completed runs.
- Masked-points sweep contains `6/6` completed runs.
- Combined fair summary contains `21` formal run rows and `7` aggregate rows.
- No `masked_smoke`, `ablation_smoke`, or other smoke directories are included in the combined fair summary.
- Metrics are finite for all reviewed runs.
- No-JEPA runs emit expected TensorFlow no-gradient warnings because `lambda_jepa=0.0` disables predictor/JEPA supervision.

Interpretation:

- Fair positive evidence exists only for the broader JEPA-LDNet sparse encoder family: under the same `Adam 200 / BFGS 0` optimizer budget, the sparse encoder JEPA runner is far better than Original LDNet Adam-only on Case `1a`.
- This is not evidence that masked-points JEPA is better. Masked-points JEPA does not clearly beat no-JEPA or current point-target JEPA:
  - at ratio `0.05`, masked-points is slightly better than current points and no-JEPA, but the difference is much smaller than seed variability.
  - at ratio `0.20`, masked-points is worse than no-JEPA and current points.
- These Adam-only results must not be presented as better than the author-budget Original LDNet `Adam+BFGS` baseline. The author-budget Case `1a` baseline remains much lower at NRMSE about `8.58e-03`.
- Current evidence suggests the main gain is from learning an observation encoder / nonzero inferred initial latent state under sparse context, not from the current JEPA feature target itself.

Next recommended method step:

- Do not expand masked-points JEPA to Case `2/3` yet.
- First improve the JEPA objective so it contributes beyond the sparse encoder:
  - predict longer-horizon targets with `prediction_horizon > 2`;
  - use target patches or time-separated targets where reconstruction is less redundant;
  - add a real dynamics consistency loss only if latent targets from multiple context windows are implemented;
  - evaluate on Case `3` long rollout after a small Case `1a` target-mode gate shows a clear benefit over no-JEPA.

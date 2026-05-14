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

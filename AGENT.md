# LDNets Project Agent Guide

本文档是 LDNets 项目后续复现、改进和多智能体协作的大纲。它基于当前服务器复现状态、`PROGRESS.md` 的实验记录，以及 `refer.md` 中提出的创新方向制定。后续 agent 执行任务时，应优先读取本文档，再读取 `PROGRESS.md`、`refer.md` 和对应 `src/TestCase_*.py`。

## 1. 当前状态

### 1.1 已验证环境

- 项目路径：`/home/fzt/projects/LDNets/LDNets-repo`
- Python 环境：`/home/fzt/miniconda3/envs/ldnets-py39`
- TensorFlow：`2.8.0`
- GPU：2 x NVIDIA GeForce RTX 4090
- 当前服务器 GPU 可用，但 Codex 默认 sandbox 可能看不到 `/dev/nvidia*`。
- 从 Codex 执行 GPU 检查或训练时，必须使用 escalated command；不要仅凭默认 sandbox 的 `nvidia-smi` 失败判断主机 GPU 不可用。

标准运行模板：

```bash
env -u PYTHONPATH \
  CUDA_VISIBLE_DEVICES=0 \
  LD_LIBRARY_PATH=/home/fzt/miniconda3/envs/ldnets-py39/lib:/opt/ros/humble/opt/rviz_ogre_vendor/lib:/opt/ros/humble/lib/x86_64-linux-gnu:/opt/ros/humble/lib \
  MPLCONFIGDIR=/tmp/matplotlib-ldnets \
  /home/fzt/miniconda3/envs/ldnets-py39/bin/python <script.py> <args>
```

运行原则：

- 一次训练进程只绑定一张 GPU：`CUDA_VISIBLE_DEVICES=0` 或 `1`。
- `PYTHONPATH` 必须 unset，避免 ROS Python 3.10 污染 conda Python 3.9。
- `LD_LIBRARY_PATH` 必须让 conda env 的 `lib` 排在最前。
- 测试阶段必须启用 chunked evaluation，避免 TensorFlow 2.8 + RTX 4090 在超大 Dense 调用上出现 cuBLAS launch failure。

### 1.2 已完成复现

Case 1 已完成 GPU 复现：

| Case | Adam + BFGS | NRMSE | Pearson dissimilarity | 说明 |
| --- | ---: | ---: | ---: | --- |
| 1a | 50 + 150 | `8.577e-03` | `6.836e-04` | 匹配作者 notebook |
| 1b | 50 + 150 | `2.439e-02` | `1.051e-02` | 使用 `--batch-samples 25` |
| 1c | 200 + 1800 | `2.059e-02` | `1.175e-02` | 使用 `--batch-samples 25` |

Case 2/3 已完成受控复现：

| Case | Adam + BFGS | NRMSE | Pearson dissimilarity | 说明 |
| --- | ---: | ---: | ---: | --- |
| 2 NS | 200 + 500 | `2.030e-02` | `9.335e-02` | 受控复现，非作者完整 BFGS |
| 3 AP1D | 200 + 500 | `2.096e-02` | `2.339e-03` | 受控复现，非作者完整 BFGS |

待补齐 baseline：

- Case 2 author budget：`BFGS 10000`
- Case 3 author budget：`BFGS 5000`
- Case 4 reentry：先不作为第一轮创新阻塞项，待 Case 1/2/3 改进稳定后接入。

## 2. 项目主线

`refer.md` 推荐的最小可发表路线是：

1. 严格复现原始 LDNet，建立可信 baseline anchor。
2. 实现 JEPA 约束的预测式 LDNet，形成主要创新点。
3. 实现 neural operator token 与 LDNet 的 hybrid，形成第二创新点。
4. 概率生成和控制规划作为扩展方向，不进入第一轮主线实现。

第一阶段论文叙事建议：

```text
World-LDNet:
Joint-Embedding Predictive Latent Dynamics
for Meshless Scientific World Models
```

核心 claim：

- 从原始 LDNet 的 supervised field reconstruction，升级为 observation-encoding、latent rollout、representation prediction、continuous decoding 的 scientific world model。
- 在 sparse observation、variable initial condition 和 long rollout 设置下，比原始 LDNet 更稳定。
- 在 NS 等复杂空间场中，通过 local operator tokens 改善局部结构和高频误差。

## 3. 分阶段执行计划

### Stage 0: Baseline Anchor

目标：

- 巩固原始 LDNet 复现结果。
- 建立所有后续创新都必须对齐的 benchmark。
- 规范实验输出和记录格式。

明确改动：

- 保留现有 CLI runner：
  - `src/TestCase_1.py`
  - `src/TestCase_2.py`
  - `src/TestCase_3.py`
- 不在 Stage 0 修改模型结构。
- 给后续实验统一输出目录：
  - `runs/baseline/case1/<run_name>/`
  - `runs/baseline/case2/<run_name>/`
  - `runs/baseline/case3/<run_name>/`
- 每次实验必须保存：
  - command
  - git commit
  - GPU id
  - seed
  - model hyperparameters
  - metrics JSON
  - loss figure
  - comparison figure

优先实验：

```bash
# Case 2 full-budget baseline
env -u PYTHONPATH \
  CUDA_VISIBLE_DEVICES=0 \
  LD_LIBRARY_PATH=/home/fzt/miniconda3/envs/ldnets-py39/lib:/opt/ros/humble/opt/rviz_ogre_vendor/lib:/opt/ros/humble/lib/x86_64-linux-gnu:/opt/ros/humble/lib \
  MPLCONFIGDIR=/tmp/matplotlib-ldnets \
  /home/fzt/miniconda3/envs/ldnets-py39/bin/python src/TestCase_2.py \
  --bfgs-epochs 10000 \
  --eval-point-batch 500 \
  --output-dir runs/baseline/case2/bfgs10000_seed0
```

```bash
# Case 3 full-budget baseline
env -u PYTHONPATH \
  CUDA_VISIBLE_DEVICES=1 \
  LD_LIBRARY_PATH=/home/fzt/miniconda3/envs/ldnets-py39/lib:/opt/ros/humble/opt/rviz_ogre_vendor/lib:/opt/ros/humble/lib/x86_64-linux-gnu:/opt/ros/humble/lib \
  MPLCONFIGDIR=/tmp/matplotlib-ldnets \
  /home/fzt/miniconda3/envs/ldnets-py39/bin/python src/TestCase_3.py \
  --bfgs-epochs 5000 \
  --eval-batch-samples 10 \
  --output-dir runs/baseline/case3/bfgs5000_seed0
```

实验指标：

- NRMSE
- Pearson dissimilarity
- final train loss
- final valid loss
- runtime
- max GPU memory
- parameter count

成功标准：

- Case 1 保持与作者 notebook 基本一致。
- Case 2/3 full-budget 结果优于或接近当前 BFGS 500 受控复现。
- 形成可复用的 baseline metrics 表，后续所有创新实验必须引用。

### Stage 1: JEPA Predictive LDNet

目标：

- 实现主要创新方向：JEPA 约束的预测式 LDNet。
- 支持 sparse sensors、variable initial conditions 和 long rollout。
- 将 LDNet 从“只监督输出场”的 surrogate，升级为“观测-表征-预测-重构”的 world model。

核心模块：

- `E_phi` observation encoder：
  - 输入 sparse sensor 坐标和值：`X_obs, Y_obs`
  - 输出初始 latent：`z0`
  - 后续可扩展为输出上下文 latent 序列：`z_{1:Tc}`
- `T_theta` LDNet latent transition：
  - 复用当前 `NNdyn`
  - 第一版仍使用 Euler rollout
- `E_bar` target encoder：
  - 输入未来时间片或 masked spatial patches
  - 输出 target embedding
  - 使用 EMA teacher，不直接反向传播更新
- `P_psi` predictor：
  - 从 current latent 或 rollout latent 预测 target embedding
- `D_omega` continuous decoder：
  - 复用当前 `NNrec`
  - 保持 meshless coordinate query 能力

建议代码结构：

- 新增 `src/models.py`：放共享模型组件。
- 新增 `src/losses.py`：放 reconstruction、JEPA、latent smoothness、regularization loss。
- 新增 `src/metrics.py`：统一 NRMSE、Pearson、horizon-wise NRMSE、sensor NRMSE。
- 新增 `src/TestCase_1_jepa.py`：先在 ADR 上实现最小闭环。
- 待 Case 1 稳定后，新增 `src/TestCase_3_jepa.py` 和 `src/TestCase_2_jepa.py`。

第一版损失：

```text
L = lambda_rec * MSE(y_hat, y)
  + lambda_jepa * MSE(stopgrad(h_target), h_pred)
  + lambda_dyn * MSE(z_next, T(z_t, u_t, p))
  + lambda_smooth * mean(||z_{t+1} - z_t||^2)
```

默认权重：

- `lambda_rec = 1.0`
- `lambda_jepa = 0.1`
- `lambda_dyn = 0.01`
- `lambda_smooth = 1e-4`

训练策略：

- warmup 阶段只训练 reconstruction 和 dynamics。
- JEPA 权重从 0 线性增加到目标值。
- target encoder 使用 EMA：
  - `ema_decay = 0.99` 起步
  - 后续 sweep `0.95 / 0.99 / 0.995`

实验设置：

- Case 1a/1b/1c 作为快速验证。
- Case 3 作为中等复杂时序验证。
- Case 2 作为复杂空间场验证。
- sensor ratio sweep：
  - `1.0`
  - `0.5`
  - `0.2`
  - `0.1`
  - `0.05`
- rollout horizon：
  - in-distribution horizon
  - full test horizon
  - extended horizon, if data supports it

实验指标：

- full-field NRMSE
- Pearson dissimilarity
- few-sensor NRMSE
- horizon-wise NRMSE
- JEPA feature MSE
- latent smoothness
- rollout stability
- runtime
- parameter count

消融：

- Original LDNet
- Encoder-LDNet without JEPA
- JEPA-LDNet without EMA teacher
- JEPA-LDNet without smoothness loss
- Full JEPA-LDNet

成功标准：

- 在 `sensor ratio <= 0.2` 时，full-field NRMSE 明显优于原始 LDNet 或 zero-latent baseline。
- long rollout 的 horizon-wise NRMSE 增长更慢。
- 完整观测下，性能不能明显劣于原始 LDNet baseline。

### Stage 2: Operator-token Hybrid LDNet

目标：

- 改善原始 `NNrec(z, x)` 对局部结构和非局部空间交互的表达能力。
- 重点提升 Case 2 Navier-Stokes 和后续 Case 4 reentry。

核心模块：

- `LocalOperatorEncoder`：
  - 输入局部观测场或 sampled points
  - 输出 local tokens：`L_t`
- `GlobalLatentState`：
  - 从 local tokens pooling 得到 global latent context
  - 与原始 LDNet latent `z_t` 协同工作
- `CoupledTransition`：
  - 同时推进 `z_t` 和 `L_t`
- `CoordinateQueryDecoder`：
  - 输入 `z_t, local_context(x), x`
  - 输出 `y_hat(x, t)`

建议代码结构：

- 在 `src/models.py` 中新增 operator token 模块。
- 新增 `src/TestCase_2_hybrid.py`，优先服务 NS。
- Case 4 等 reentry 数据验证后再接入 hybrid runner。

第一版实现限制：

- 不引入大型 FNO 依赖。
- 先用 MLP/attention pooling 实现轻量 local token encoder。
- 保持输入输出仍兼容当前 meshless data loader。

实验设置：

- Case 2 full grid test。
- irregular query points test。
- train points sweep：
  - `50`
  - `100`
  - `200`
  - `500`
- token number sweep：
  - `8`
  - `16`
  - `32`

实验指标：

- NRMSE
- Pearson dissimilarity
- Case 2 direction loss
- spectral error
- irregular query NRMSE
- high-gradient region NRMSE
- runtime
- GPU memory
- parameter count

消融：

- Original LDNet
- JEPA-LDNet
- Hybrid-LDNet without JEPA
- JEPA + Hybrid

成功标准：

- Case 2 full-grid NRMSE 低于原始 LDNet baseline。
- 高频结构或局部梯度区域误差下降。
- 参数量和训练成本仍处于轻量级范围，不能退化为大规模神经算子。

### Stage 3: Paper-facing Ablation Matrix

目标：

- 形成论文级证据链。
- 明确每个模块的贡献，而不是只展示最终最优结果。

实验矩阵：

| Block | Dataset | Baseline | Metrics |
| --- | --- | --- | --- |
| baseline reproduction | Case 1/2/3 | Original LDNet | NRMSE, Pearson, runtime |
| sparse observation | Case 1/3/2 | zero-latent LDNet, Encoder-LDNet | few-sensor NRMSE, full-field NRMSE |
| long rollout | Case 2/3 | Original LDNet, JEPA-LDNet | horizon-wise NRMSE |
| operator hybrid | Case 2, later Case 4 | LDNet, JEPA-LDNet | spectral error, direction loss |
| ablation | all stable cases | module removal | delta metrics |

报告规范：

- 关键实验必须报告 `mean ± std`。
- 初期使用 `seed=0` 快速筛选。
- 进入论文表格前使用 `seed=0/1/2`。
- 每个结果必须能追溯到：
  - command
  - commit
  - output directory
  - metrics JSON

主要图表：

- baseline reproduction table
- sparse sensor ratio curve
- horizon-wise rollout error curve
- ablation bar chart
- Case 2 field comparison
- Case 2 spectral/high-gradient error plot

### Stage 4: Deferred Extensions

以下方向暂不进入第一轮主线，避免工程复杂度失控：

- probabilistic latent rollout
- feature diffusion / flow matching
- uncertainty calibration
- control and planning
- TD-MPC style latent planning

进入条件：

- Stage 1 JEPA 在 sparse observation 和 long rollout 上有稳定收益。
- Stage 2 hybrid 在 NS 或 reentry 上有明确收益。
- baseline 和 ablation 已形成可复现实验矩阵。

## 4. Agent 协作规则

建议使用两个角色协作：

### Training Agent

职责：

- 执行 GPU 训练。
- 检查训练进程、显存、日志和输出。
- 汇总 metrics。
- 遇到失败时先判断是代码问题、数据问题、显存问题还是 sandbox/GPU 可见性问题。

必须记录：

- command
- start time / end time
- GPU id
- output dir
- metrics
- failure traceback

### Review and Documentation Agent

职责：

- 审查训练命令是否符合当前服务器约束。
- 审查结果是否可比。
- 更新 `PROGRESS.md`。
- 必要时更新本文档。
- 检查是否误改 baseline 或破坏既有 CLI 参数。

审查重点：

- 是否固定 seed。
- 是否改变训练 objective。
- evaluation chunk 是否只影响测试预测，不改变训练目标。
- 是否把受控复现和 full-budget 复现混为一谈。
- 是否把未提交的大文件、数据或 runs 加入 git。

## 5. 实验记录格式

每个正式实验在 `PROGRESS.md` 中使用如下格式：

```markdown
### YYYY-MM-DD <Experiment Name>

- Commit:
- GPU:
- Command:
- Output dir:
- Dataset split:
- Model:
- Training budget:
- Evaluation mode:
- Metrics:
  - NRMSE:
  - Pearson dissimilarity:
  - final train loss:
  - final valid loss:
  - runtime:
  - max GPU memory:
- Interpretation:
- Next action:
```

每个实验输出目录建议包含：

```text
metrics.json
command.txt
loss.png
comparison.png
config.json
```

## 6. 当前结论和下一步优先级

已完成的可靠结论：

- 原始 LDNets Case 1/2/3 复现路径可用，服务器 TensorFlow/GPU 环境可训练。
- sparse observation encoder / inferred latent initialization 是当前最稳定的改进方向，但结论需要按 case 分层陈述。
- Main `no_jepa_sparse` near-convergence matrix 已完成：Case `1a/1b/1c` x sensor ratio `1.0/0.5/0.2/0.1/0.05` x seed `0/1/2`，共 `45/45` runs，Adam `4000` + BFGS `150`。
- Case `1a` 和 Case `1b` 支持 sparse encoder 主线：
  - Case `1a` NRMSE 约 `0.0019-0.00235` across sensor ratios。
  - Case `1b` NRMSE 约 `0.00894-0.01043` across sensor ratios，明显优于原始参考约 `0.0244`。
- Case `1c` 是当前限制：
  - NRMSE 约 `0.0267-0.0283` across sensor ratios，弱于原始参考约 `0.0204`。
  - full-sensor 与 sparse ratios 同量级，问题更像 Case `1c` 的预算/架构/正则适配，而不是传感器比例本身。
- Stage 1B 的 points / long-horizon patch JEPA 没有稳定超过 `no_jepa_sparse`。
- Stage 1C 的 multi-context latent target 工程路径可用，但直接 latent MSE dynamics consistency 未带来正向精度收益。
- 不能把当前 JEPA objective 作为已验证的主创新 claim。

已实现的 Stage 1C 模块：

- `multi-context-latent` target mode。
- EMA teacher encoder 输出 teacher latent 和 teacher embedding。
- dynamics consistency:

```text
MSE(z_rollout, stopgrad(z_teacher))
```

- epoch-level dynamic target/time resampling。
- summary/provenance fields for `lambda_dyn_consistency`, teacher windows, resampling policy, and dynamics metrics。

下一步优先级：

1. 不要直接扩展 Stage 1C dynamics objective 到 Case `2/3`。
2. 不要直接宣称 sparse encoder 全 Case 改进；当前 verified claim 应限定为 Case `1a/1b` 的 sparse robustness。
3. 先做 Case `1c` 诊断：
   - original runner 与 sparse runner 使用相同 Adam/BFGS 预算对齐；
   - sparse runner Case `1c`, `sr=1.0/0.2`, seed `0`, BFGS `500/1000` 小矩阵；
   - 检查 Case `1c` 的 latent dimension、regularization `alpha_reg=2.7e-4`、signal conditioning 是否需要 sparse encoder 专门适配。
4. 如果继续 JEPA 方向，先做更小 gate：
   - normalized latent MSE 或 cosine latent consistency；
   - learned projection head before latent consistency；
   - `lambda_dyn_consistency=0.001/0.003/0.01`；
   - delayed dynamics schedule after reconstruction stabilizes；
   - teacher windows based on sensor-observable context to reduce hidden-patch train/test mismatch。
5. 如果追求短期论文稳定性，优先围绕 sparse observation encoder 展开：
   - sparse sensor ratio robustness；
   - seed stability；
   - Case `1a/1b` positive transfer；
   - Case `1c` limitation and diagnostic；
   - 与 original LDNet Adam-only 和 author-budget Adam+BFGS 分开对比。
6. Case `2/3` 后续实验应标注为 diagnostic，且必须保留 `no_jepa_sparse` 和 `current_points` 控制组。

短期目标：

- 将 sparse encoder 的 Case `1a/1b` 正向证据整理成清晰 baseline 表和曲线。
- 对 Case `1c` 做预算/runner 对齐诊断，判断是否只是 BFGS 不足。
- 对 JEPA objective 做一轮更弱/归一化目标的小规模 gate。
- 决定论文主创新是否从 JEPA 调整为 sparse-observation latent initialization + diagnostic self-supervised objectives，并明确 Case `1c` 是 limitation 还是可修复问题。

中期目标：

- 在小 gate 明确正向后，再迁移到 Case `3` long rollout。
- 如果 JEPA 仍无正向收益，转向 operator-token hybrid 或 uncertainty/probabilistic rollout。

长期目标：

- 形成 World-LDNet 论文级实验矩阵。
- 明确哪些模块是 verified contribution，哪些只是 negative ablation 或 future work。

## 7. GPU Scheduling Policy

Current server:

- GPU 0: NVIDIA GeForce RTX 4090, 24 GB.
- GPU 1: NVIDIA GeForce RTX 4090, 24 GB.
- Use host/external execution for TensorFlow GPU training. The default Codex sandbox can report `CUDA_ERROR_NO_DEVICE` even when the user's normal shell sees both GPUs.

Working command pattern:

```bash
env -u PYTHONPATH CUDA_VISIBLE_DEVICES=<0-or-1> \
  LD_LIBRARY_PATH=/home/fzt/miniconda3/envs/ldnets-py39/lib:/opt/ros/humble/opt/rviz_ogre_vendor/lib:/opt/ros/humble/lib/x86_64-linux-gnu:/opt/ros/humble/lib \
  MPLCONFIGDIR=/tmp/matplotlib-ldnets \
  /home/fzt/miniconda3/envs/ldnets-py39/bin/python ...
```

Observed one-GPU multi-process smoke:

- Date: `2026-05-29`
- While a formal Case `1c` sparse run was active on GPU 0, a second short `TestCase_1_jepa.py` process was started on the same GPU with `CUDA_VISIBLE_DEVICES=0`.
- TensorFlow successfully created a GPU device for the second process.
- `nvidia-smi` showed two compute Python processes on GPU 0:
  - formal process: about `1768 MiB`
  - smoke process: about `1000 MiB`
  - total GPU 0 memory: about `3226 MiB / 24564 MiB`
- The smoke run completed successfully in `17.3 s`.

Practical conclusion:

- Single-card multi-process training is feasible for the current Case `1` JEPA/sparse workloads.
- Start with at most `2` long training processes per RTX 4090. This keeps memory far below capacity while avoiding excessive CPU/BFGS contention.
- For light Adam-only probes, `3-4` processes per card may be possible, but test with a short smoke first and monitor power/utilization.
- Do not pack Case `2/3` full-budget BFGS jobs aggressively; they should be treated as heavier reproduction baselines.
- Always write each process to a unique output directory and record `CUDA_VISIBLE_DEVICES`, command hash, git status, seed, sensor ratio, budget, and model size.
- Monitor every `10-20` minutes during stable long runs with `nvidia-smi` and targeted `ps -o pid,etime,pcpu,pmem,rss,stat,cmd -p <pid>`.

## 8. Current Case 1c Alignment Finding

Stage A Case `1c` alignment diagnostic completed on `2026-05-29`.

Key results:

- Original author-style `TestCase_1.py`, Adam200+BFGS1800:
  - NRMSE `2.0632e-02`
  - Pearson dissimilarity `1.1794e-02`
  - This is close to the reference `2.039e-02 / 1.152e-02`.
- Original short BFGS, Adam4000+BFGS150:
  - NRMSE `2.4363e-02`
  - This confirms Case `1c` is sensitive to long BFGS.
- Sparse `TestCase_1_jepa.py`, Adam4000+BFGS150:
  - `sr=1.0`: NRMSE `2.6306e-02`
  - `sr=0.2`: NRMSE `2.7198e-02`
  - This reproduces the weak `~0.027` region.
- Sparse `TestCase_1_jepa.py`, Adam200+BFGS1800:
  - `sr=1.0`: NRMSE `1.8307e-02`
  - `sr=0.2`: NRMSE `2.1668e-02`

Conclusion:

- The earlier Case `1c` sparse weakness is mainly a convergence/budget issue, not a hard failure of sparse encoder.
- With author-style BFGS, full-sensor sparse Case `1c` beats the original author-style baseline.
- Sparse `sr=0.2` gets close to the Case `1c` reference but does not yet clearly beat it.
- Next experiments should search for a cheaper convergence point before expanding latent sweeps:
  - sparse Case `1c`, latent `4`, `sr=1.0/0.2`, BFGS `500/1000`;
  - then latent `5/7` only under the best reduced budget;
  - extend only winning candidates to BFGS1800 and seeds `1/2`.

# Stage 1 JEPA-LDNet Image-2 Prompts

The current Codex environment does not have `OPENAI_API_KEY` set, so the
image-2 CLI generation step cannot be executed here yet. Keep these prompts as
the source of truth for generating the final paper-style bitmap figures with
`gpt-image-2` once API credentials are available.

## Architecture Figure

Target path:

```text
docs/figures/stage1_jepa_ldnet_architecture.png
```

Prompt:

```text
Use case: scientific-educational
Asset type: research paper architecture diagram
Primary request: Create a clean technical architecture diagram for "JEPA-LDNet: Joint-Embedding Predictive Latent Dynamics Network".
Style/medium: polished scientific paper figure, vector-like raster diagram, white background, crisp lines.
Composition/framing: left-to-right architecture pipeline with grouped modules and arrows.
Text labels must be in English and exactly use these module names:
Sparse Observations, Observation Encoder E_phi, Initial Latent State z0, Latent Transition T_theta, Latent Rollout z_1:T, Continuous Decoder D_omega, Field Prediction Y_hat, Target Patches, EMA Target Encoder E_bar, Target Embedding h_target, Predictor P_psi, Predicted Embedding h_pred, Reconstruction Loss, JEPA Loss, EMA Update.
Visual structure:
- top branch: Sparse Observations -> Observation Encoder -> z0 -> Latent Transition -> Latent Rollout -> Continuous Decoder -> Field Prediction.
- bottom branch: Target Patches -> EMA Target Encoder -> Target Embedding.
- Predictor connects Latent Rollout to Predicted Embedding.
- JEPA Loss compares Predicted Embedding and Target Embedding with stop-gradient on target.
- Reconstruction Loss compares Field Prediction with Ground Truth Field.
- EMA Update arrow updates Target Encoder from online encoder.
Constraints: no decorative background, no 3D perspective, no icons unrelated to science, no watermark, no extra labels beyond the specified labels.
```

CLI command:

```bash
python "$HOME/.codex/skills/.system/imagegen/scripts/image_gen.py" generate \
  --model gpt-image-2 \
  --quality high \
  --size 2048x1152 \
  --prompt-file docs/stage1_jepa_image2_architecture_prompt.txt \
  --out docs/figures/stage1_jepa_ldnet_architecture.png
```

## Training Flow Figure

Target path:

```text
docs/figures/stage1_jepa_ldnet_training_flow.png
```

Prompt:

```text
Use case: scientific-educational
Asset type: machine learning training workflow diagram
Primary request: Create a complete training flow diagram for JEPA-LDNet.
Style/medium: clean research workflow figure, vector-like raster diagram, white background, clear numbered stages.
Composition/framing: vertical or circular workflow with five grouped phases.
Text labels must be in English and exactly use these stage names:
1. Load Normalized Field Data
2. Sample Context Sensors
3. Encode Latent State
4. Roll Out Latent Dynamics
5. Decode Query Field
6. Build JEPA Targets
7. Predict Target Embeddings
8. Compute Total Loss
9. Update Online Networks
10. EMA Update Target Encoder
11. Evaluate Metrics
Include metric labels:
NRMSE, Pearson Dissimilarity, Sensor NRMSE, Horizon-wise NRMSE.
Include loss labels:
L_rec, L_jepa, L_smooth.
Constraints: no decorative background, no unrelated icons, no watermark, no dense paragraphs, keep text large and readable.
```

CLI command:

```bash
python "$HOME/.codex/skills/.system/imagegen/scripts/image_gen.py" generate \
  --model gpt-image-2 \
  --quality high \
  --size 2048x1152 \
  --prompt-file docs/stage1_jepa_image2_training_prompt.txt \
  --out docs/figures/stage1_jepa_ldnet_training_flow.png
```

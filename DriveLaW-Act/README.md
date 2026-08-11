# DriveLaW-Act (ReWorld action stages)

Planning / NAVSIM side of **ReWorld**. Full setup, data, and three-stage curriculum are documented in the **[repository-root README](../README.md)**.

## Quick start

1. Install NAVSIM data and env: [docs/install.md](docs/install.md).  
2. Cache features/metrics:

```bash
sh scripts/evaluation/run_caching_videodrive_hidden_state.sh
sh scripts/evaluation/run_metric_caching.sh
```

3. Train (edit paths in each script and the matching YAML under `navsim/agents/videodrive/configs/ltx_model/`):

| Stage | Script | Config |
|-------|--------|--------|
| Base IL (optional) | `scripts/training/run_videodrive_train.sh` | `video_model_train_base.yaml` |
| ReWorld Stage 2 Align | `scripts/training/run_videodrive_train_stage2_align.sh` | `video_model_train_stage2_align.yaml` |
| ReWorld Stage 3 RDE | `scripts/training/run_videodrive_train_stage3_rde.sh` | `video_model_train_stage3_rde.yaml` |

4. Evaluate:

```bash
sh scripts/evaluation/run_videodrive_agent_pdm_score_evaluation.sh
```

You may skip base IL by loading [DriveLaW weights](https://huggingface.co/tz2026/DriveLaW) into `diffusion_model.model_path` before Stage 2.

Stage 3 requires BeyondDrive hard-negative `{token}.pkl` files — see the root README (download or generate via [BeyondDrive](https://github.com/wjl2244/BeyondDrive)).

> Note: PDMS improves with resolution; paper metrics used 1344×768 when hardware allows.

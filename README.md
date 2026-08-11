<div align="center">

# ReWorld: Representation Learning for World Action Models

**The first representation learning framework for autonomous-driving World Action Models —<br>
explicitly optimizing the latent world-to-action pathway.**

[![arXiv](https://img.shields.io/badge/arXiv-2606.27504-B31B1B?logo=arxiv)](https://arxiv.org/abs/2606.27504)
[![Project Page](https://img.shields.io/badge/Project-Page-1f6feb?logo=githubpages)](https://xiaomi-research.github.io/ReWorld/)
[![Models](https://img.shields.io/badge/%F0%9F%A4%97%20Models-ReWorld-ffb300)](https://huggingface.co/tz2026/ReWorld)
[![License](https://img.shields.io/badge/License-Apache%202.0-green)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](pyproject.toml)

[Tianze Xia](mailto:xiatianze@hust.edu.cn)<sup>1,2*</sup>, Lijun Zhou<sup>2*</sup>, Kaixin Xiong<sup>2</sup>, Jingfeng Yao<sup>1</sup>, Zhenxin Zhu<sup>2</sup>, Haiyang Sun<sup>2</sup>, Bing Wang<sup>2</sup>, Guang Chen<sup>2</sup>, Wenyu Liu<sup>1</sup>, Hangjun Ye<sup>2</sup>, Xinggang Wang<sup>1†</sup>

<sup>1</sup> Huazhong University of Science and Technology &nbsp;&nbsp; <sup>2</sup> Xiaomi EV
<br>
<sup>*</sup> Equal contribution &nbsp;&nbsp; <sup>†</sup> Corresponding author

<img src="assets/reworld_framework.png" width="100%" alt="ReWorld framework">

</div>

---

## :sparkles: Highlights

- **Better futures** — FVD **81.3 → 61.9** (−23.9%) on nuScenes video generation, with self-guided sampling enabled for free by intermediate supervision.
- **Safer plans** — closed-loop PDMS **89.1 → 90.4** on NAVSIM *Navtest*, with best-in-class NC / DAC / TTC among world-model planners — no RL, no test-time scoring.
- **Stronger representations** — frozen linear probe on UCF-101 action recognition **68.3% → 80.2%** (+11.9 pts over the DriveLaW baseline).
- **Nearly free** — no external encoders, no teacher models, only **+0.3%** per-step Video DiT training cost, and roughly **2× faster** convergence from scratch.

## :bulb: Why ReWorld?

World Action Models chain a **Video DiT** (imagines the future) with an **Action DiT** (plans trajectories on mid-denoising video features). But under standard output-level training, the intermediate states along this world-to-action pathway are mere byproducts — not future-predictive, not cross-modally grounded, and blind to closed-loop behavior quality. We call this the **representation bottleneck of WAMs**.

ReWorld removes the bottleneck with a three-stage curriculum whose supervision comes *entirely from the model's own* generation targets, attended features, and trajectory candidates. One line of math tells the whole story — formation, then transfer, then decision-oriented shaping:

$$
\underbrace{\mathcal{L}_{\mathrm{Gen}} + \lambda_{\mathrm{Mid}}\,\mathcal{L}_{\mathrm{Mid}}}_{\text{Stage 1 · future-predictive video states}}
\;\Longrightarrow\;
\underbrace{\mathcal{L}_{\mathrm{FM}} + \lambda_{\mathrm{align}}\,\mathcal{L}_{\mathrm{align}}}_{\text{Stage 2 · world-grounded action states}}
\;\Longrightarrow\;
\underbrace{\mathcal{L}_{\mathrm{FM}} + \lambda_{\mathrm{RDE}}\,\mathcal{L}_{\mathrm{RDE}}}_{\text{Stage 3 · behavior-aware action shaping}}
$$

| Stage | In plain terms |
|:---:|---|
| **1 · Video Mid / IG** | Middle Video DiT blocks learn to predict the future *directly*, instead of only through the final output. The induced gap between intermediate and final predictions becomes a **free guidance direction** at sampling: $v_w = v_i + \gamma\,(v_f - v_i)$ — and training converges ~2× faster. |
| **2 · Cross-modal Align** | Video DiT **frozen**. Each action state right after cross-attention is pulled (cosine, stop-grad) toward the video readout it just attended to — retrieved world knowledge is *retained*, not transiently used. |
| **3 · RDE** | Both branches fine-tuned jointly. The prediction is **repelled from the nearest low-scoring trajectory** in an offline-mined candidate pool, separating the expert from geometrically close but unsafe alternatives. |

<div align="center">
<img src="assets/reworld_self_guidance.png" width="100%" alt="Self-guided sampling and faster convergence">
</div>

> **Why sequential?** Stage 2 needs a *frozen* video space as a stable grounding target; Stage 3 *deliberately* lets behavior-oriented gradients reshape the planner-facing video states — so $\mathcal{L}_{\mathrm{align}}$ and $\mathcal{L}_{\mathrm{RDE}}$ are applied in separate stages, never combined.

## :trophy: Results at a Glance

<details open><summary><b>Video generation on nuScenes</b> (val)</summary>

| Method | FID ↓ | FVD ↓ |
|:---|:---:|:---:|
| DriveDreamer | 52.6 | 452.0 |
| Vista | 6.9 | 89.4 |
| Epona | 7.5 | 82.8 |
| DriveLaW | 4.6 | 81.3 |
| **ReWorld (Ours)** | **4.4** | **61.9** |

</details>

<details open><summary><b>Closed-loop planning on NAVSIM <i>Navtest</i></b></summary>

| Method | NC ↑ | DAC ↑ | TTC ↑ | Comf. ↑ | EP ↑ | PDMS ↑ |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| DiffusionDrive (cam+lidar) | 98.2 | 96.2 | 94.7 | 100 | 82.2 | 88.1 |
| Epona | 97.9 | 95.1 | 93.8 | 99.9 | 80.4 | 86.2 |
| PWM | 98.6 | 95.9 | 95.4 | 100 | 81.8 | 88.1 |
| WorldDrive | 98.4 | 96.8 | 95.2 | 100 | **83.3** | 89.0 |
| DriveLaW | 99.0 | 97.1 | 96.7 | 100 | 81.3 | 89.1 |
| **ReWorld (Ours)** | **99.1** | **98.2** | **97.7** | 99.8 | 82.0 | **90.4** |

</details>

**Do the representations themselves actually improve?** — the experiment we care most about:

<div align="center">
<img src="assets/representation_results.png" width="92%" alt="Representation quality: UCF-101 frozen probe and unified-protocol FVD">
</div>

Driving-domain pretraining alone barely moves the probe (+1.5 over LTX-Video); the representation curriculum adds **+11.9**. And under a unified from-scratch protocol (120k steps, nuPlan + nuScenes, no text encoder), ReWorld beats the strongest self-supervised baseline by **12.9 FVD** at **1.003×** per-step cost — teacher-based and extra-forward methods pay ~1.4–1.7× for less.

<details><summary><b>Full numbers</b> (UCF-101 probe & unified protocol)</summary>

| Frozen Video DiT | Top-1 Acc. (%) ↑ |
|:---|:---:|
| LTX-Video | 66.8 |
| DriveLaW | 68.3 |
| ReWorld (Stage 1) | 71.7 |
| **ReWorld (full)** | **80.2** |

| Method (120k steps, from scratch) | FVD ↓ | Training cost |
|:---|:---:|:---:|
| Vanilla Flow | 304.1 | 1.0× |
| SRA / Self-Flow | 296.9 / 283.3 | ~1.4× |
| REPA w/ DINOv2 | 295.9 | ~1.7× |
| **ReWorld (Ours)** | **270.4** | **1.003×** |

</details>

<div align="center">
<img src="assets/qualitative_contrast.png" width="100%" alt="Qualitative comparison: DriveLaW vs ReWorld">
<br><i>1s history → 3s future. ReWorld yields sharper lane markings, more stable roadside structure, and clearer distant agents.</i>
<br><br>
<img src="assets/navsim_planning.png" width="100%" alt="NAVSIM planning visualization">
<br><i>NAVSIM Navtest rollouts (red: ReWorld prediction, green: expert) — straight, left turn, right turn, intersection.</i>
</div>

## :rocket: Getting Started

### Installation

Python ≥ 3.10, CUDA GPU recommended.

```bash
# Stage 1 (Video DiT) — from repo root
pip install -U pip setuptools && pip install -e .

# Stages 2–3 (Action planner + NAVSIM)
cd DriveLaW-Act && pip install -e .
```

Then follow [`DriveLaW-Act/docs/install.md`](DriveLaW-Act/docs/install.md) to download OpenScene / nuPlan maps and set environment variables (`NAVSIM_DEVKIT_ROOT`, `NAVSIM_EXP_ROOT`, `OPENSCENE_DATA_ROOT`, `NUPLAN_MAPS_ROOT`). Place [LTX-Video 0.9.5](https://huggingface.co/Lightricks/LTX-Video) weights locally or point the YAMLs at the HF id.

<details><summary><b>Data preparation</b> (video clips · NAVSIM caches · hard negatives)</summary>

- **Stage 1:** point `video_root` in [`configs/dualflow/dit_train.online.example.yaml`](configs/dualflow/dit_train.online.example.yaml) at your driving clips. Paper geometry: 33 frames (9 condition + 24 future), 1024×512, LTX temporal 8 / spatial 32 compression. For IG inference, name conditioning clips `scene_XXXX_window_000_conditioning.mp4`.
- **Stages 2–3:** cache features and metrics —
  ```bash
  cd DriveLaW-Act
  sh scripts/evaluation/run_caching_videodrive_hidden_state.sh
  sh scripts/evaluation/run_metric_caching.sh
  ```
  Optional warm-start: pretrained weights on 🤗 [tz2026/ReWorld](https://huggingface.co/tz2026/ReWorld) (base DriveLaW weights: [tz2026/DriveLaW](https://huggingface.co/tz2026/DriveLaW)).
- **Stage 3 hard negatives:** an offline pool of `{scene_token}.pkl` (`pdm_score_matrix` (N,7) + `pred_trajectorys` (N,L,3)) under `negative_samples_path` — [download](https://drive.google.com/file/d/1M3U5VvhL58QmG91PMr6EH5M6EfPwmRF_/view?usp=drive_link) or generate with [BeyondDrive](https://github.com/wjl2244/BeyondDrive). Format: [`beyonddrive_negatives.py`](DriveLaW-Act/navsim/agents/videodrive/beyonddrive_negatives.py).

</details>

### The ReWorld Pipeline

```text
Stage 1  Video DiT + IG ──► (base Act IL or 🤗 DriveLaW ckpt)
                                │
Stage 2  Cross-modal Align ◄────┘   (Video DiT frozen)
                                │
Stage 3  RDE joint fine-tune ◄──┘   (+ hard negatives)
                                │
                          NAVSIM PDMS eval
```

**Stage 1 — future-predictive Video DiT**

```bash
torchrun --nproc_per_node=$RESOURCE_GPU \
  scripts/train_dualflow_dit.py configs/dualflow/dit_train.online.example.yaml
# key fields: dit_latent_mode: vae_only · lambda_internal_guidance: 1.0
#             internal_guidance_dit_block: 8 · sampling_scale: 1.4
```

**Self-guided video generation**

```bash
python scripts/generate_reworld_future_from_condition.py \
  --conditioning-dir /path/to/conditioning_videos \
  --dit-checkpoint  /path/to/dualflow_dit_best_loss.safetensors \
  --output-root outputs/reworld_ig_future \
  --transformer-config configs/dualflow/transformer_vae_dualflow_config.json \
  --vae-model-source /path/to/LTX-Video-0.9.5 \
  --ig-scale 1.4 --ig-block 8
```

**Stage 2 — cross-modal alignment** (Video DiT frozen)

```bash
cd DriveLaW-Act && sh scripts/training/run_videodrive_train_stage2_align.sh
# video_model_train_stage2_align.yaml: lambda_cross_modal_align: 0.05
#   cross_modal_align_blocks: [12] · use_beyonddrive: false
```

**Stage 3 — RDE with hard negatives** (joint fine-tune, align off)

```bash
cd DriveLaW-Act && sh scripts/training/run_videodrive_train_stage3_rde.sh
# video_model_train_stage3_rde.yaml: use_beyonddrive: true
#   rde_loss_weight: 0.04 · beyonddrive_submetric_threshold: 0.6
```

**Evaluate** (NAVSIM PDMS)

```bash
cd DriveLaW-Act && sh scripts/evaluation/run_videodrive_agent_pdm_score_evaluation.sh
```

> Optional base imitation before Stage 2: `scripts/training/run_videodrive_train.sh` with [`video_model_train_base.yaml`](DriveLaW-Act/navsim/agents/videodrive/configs/ltx_model/video_model_train_base.yaml), or start from the 🤗 DriveLaW checkpoint.

## :file_folder: Repository Map

| Path | Contents |
|---|---|
| `src/dualflow/` | Stage-1 Video DiT (`vae_only` latent mode + Internal Guidance) |
| `configs/dualflow/` | Stage-1 training & IG sampling configs |
| `scripts/` | Stage-1 training entry + self-guided future generation |
| `DriveLaW-Act/` | Action planner: base IL, Stage-2 align, Stage-3 RDE, NAVSIM eval |
| `assets/` / `docs/` | README figures & the [project page](https://xiaomi-research.github.io/ReWorld/) source |

This release contains the **mainline** training/inference code of the three stages. Experimental ablation hooks (REPA, ReDi PCA bridges, dual-latent concat, FID/FVD hooks, planning-side Action IG, LAF scorers) are not included.

## :pray: Acknowledgments

ReWorld builds on [DriveLaW](https://arxiv.org/abs/2512.23421), [NAVSIM](https://github.com/autonomousvision/navsim), [LTX-Video](https://github.com/Lightricks/LTX-Video), [Diffusers](https://github.com/huggingface/diffusers), and the hard-negative protocol of [BeyondDrive](https://github.com/wjl2244/BeyondDrive). Thanks to all of them for open-sourcing.

This work was in part supported by NSFC U25B2067.

## :pencil: Citation

```bibtex
@article{xia2026reworld,
  title   = {ReWorld: Learning Better Representations for World Action Models},
  author  = {Xia, Tianze and Zhou, Lijun and Xiong, Kaixin and Yao, Jingfeng and Zhu, Yu and Zhu, Zhenxin and Wang, Bing and Chen, Guang and Ye, Hangjun and Liu, Wenyu and others},
  journal = {arXiv preprint arXiv:2606.27504},
  year    = {2026}
}
```

## :mailbox: Contact

Tianze Xia — xiatianze@hust.edu.cn · Xinggang Wang — xgwang@hust.edu.cn

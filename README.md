<p align="center">
  <img src="https://github.com/user-attachments/assets/89797869-d06e-4a37-bb82-8163b33f7d1e" width="900" alt="LECDrive">
</p>

<h2 align="center">
  ✨Think Densely, Act Sparsely:✨<br>
  <sub>Latent Expert Cognitive Chains for Vision-Language-Action Autonomous Driving</sub>
</h2>

<p align="center">
  <a href="https://scholar.google.com/citations?user=WjdWNi8AAAAJ" target="_blank">Jie Wang</a><sup>1</sup>,
  Guang Li<sup>3</sup>,
  Zhijian Huang<sup>3</sup>,
  Jinlong Li<sup>3</sup>,
  Chenxu Dang<sup>3</sup>,
  Hangjun Ye<sup>3</sup>,
  Yahong Han<sup>2,†</sup>,
  Long Chen<sup>3</sup>
</p>

<p align="center">
<sup>1</sup> School of Artificial Intelligence, Shanxi University<br>
<sup>2</sup> School of Artificial Intelligence, Tianjin University<br>
<sup>3</sup> Xiaomi EV &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <sup>†</sup> Corresponding author
</p>

<p align="center">
<a href="https://wj-cv.github.io/LECDrive/"><strong>🌐 Project Page</strong></a>
&nbsp; | &nbsp;
<a href="https://huggingface.co/datasets/wang-jie825/LECDrive"><strong>🤗 Dataset</strong></a>
&nbsp; | &nbsp;
<a href="https://huggingface.co/wang-jie825/LECDrive"><strong>🤗 Checkpoints</strong></a>
&nbsp; | &nbsp;
<a href="#citation"><strong>📌 Citation</strong></a>
</p>


## 📢 News

- 🚧 Installation instructions and training and evaluation guides are coming soon.
- 🚀 The **LECDrive source code** is now publicly available!
- 🌐 Visit our [project page](https://wj-cv.github.io/LECDrive/) for the method overview, experimental results, and driving demonstrations.
- 🎉 **LECDrive has been accepted to NeurIPS 2026!**
  
## 🔬 Project Overview

Autonomous driving requires an understanding of **visual appearance, semantic structure, and spatial geometry**. However, conventional vision-language-action models map dense, multi-view observations to sparse language responses or trajectory points. This mismatch can leave important scene details insufficiently supervised.

**LECDrive** introduces **“Think Densely, Act Sparsely”**: build rich and structured world representations in latent space before generating sparse driving actions.

Inspired by hierarchical human perception, LECDrive embeds a **progressive Latent Expert Cognitive Chain** within the VLA, connecting visual representation, semantic understanding, and spatial geometry.

### Key Contributions

- **🧠 Progressive latent expert cognition.** Compact task-specific tokens carry knowledge from DINOv3, SAM3, and DepthAnything3. Successive stages build on earlier representations to develop increasingly structured scene understanding.

- **🎯 Dense supervision with efficient inference.** Expert tokens are inserted after each view’s visual tokens and aggregate information across VLM layers. Lightweight heads provide dense supervision during training, allowing the VLA to internalize expert knowledge without running the large teacher models at inference.

- **🚗 Structured motion with precise refinement.** **Motion Token + Offset** combines discrete motion primitive retrieval with continuous offset correction, capturing the overall motion pattern while refining trajectory precision.

Experiments show improvements across trajectory planning, language-driven perception, and prediction on NAVSIM, NuInstruct, and DriveLM.

<p align="center">
<img src="https://raw.githubusercontent.com/WJ-CV/WJ-CV.github.io/main/LECDrive/assets/figure-2.png" width="100%" alt="LECDrive benchmark comparison">
</p>

## 🏗️ Framework

LECDrive connects dense world modeling with sparse action generation through a compact latent interface:

1. **Expert token injection:** insert task-specific tokens after each view’s visual tokens.
2. **Progressive latent cognition:** develop complementary visual, semantic, and geometric representations with expert supervision.
3. **Cross-layer aggregation:** aggregate expert-token hidden states across VLM layers for dense prediction.
4. **Trajectory generation:** combine motion primitives and continuous offsets to produce future trajectories.

Frozen expert models provide supervision during training. Inference uses the knowledge internalized by the VLA.

<p align="center">
<img src="https://raw.githubusercontent.com/WJ-CV/WJ-CV.github.io/main/LECDrive/assets/framework.png" width="100%" alt="LECDrive framework">
</p>

## 📊 Results

LECDrive is evaluated on trajectory planning, risk object perception, and state prediction benchmarks.

| Benchmark | Evaluation Focus |
|:----------|:-----------------|
| NAVSIM v1 | Trajectory planning |
| NAVSIM v2 | Trajectory planning |
| NuInstruct | Risk object perception and state prediction |
| DriveLM | Risk object perception, action prediction, and planning |

<details>
<summary><strong>NAVSIM v1</strong></summary>

<p align="center">
<img src="https://raw.githubusercontent.com/WJ-CV/WJ-CV.github.io/main/LECDrive/assets/table-1.png" width="100%" alt="Results on NAVSIM v1">
</p>

</details>

<details>
<summary><strong>NAVSIM v2</strong></summary>

<p align="center">
<img src="https://raw.githubusercontent.com/WJ-CV/WJ-CV.github.io/main/LECDrive/assets/table-2.png" width="100%" alt="Results on NAVSIM v2">
</p>

</details>

<details>
<summary><strong>NuInstruct</strong></summary>

<p align="center">
<img src="https://raw.githubusercontent.com/WJ-CV/WJ-CV.github.io/main/LECDrive/assets/table-3.png" width="100%" alt="Results on NuInstruct">
</p>

</details>

<details>
<summary><strong>DriveLM</strong></summary>

<p align="center">
<img src="https://raw.githubusercontent.com/WJ-CV/WJ-CV.github.io/main/LECDrive/assets/table-4.png" width="100%" alt="Results on DriveLM">
</p>

</details>

## 🎬 Visualizations

LECDrive combines trajectory planning with dense world modeling, including semantic segmentation and depth prediction, as well as risk object perception and state prediction.

<p align="center">
<img src="https://raw.githubusercontent.com/WJ-CV/WJ-CV.github.io/main/LECDrive/assets/figure-4.png" width="100%" alt="LECDrive visualizations across driving tasks">
</p>

Visit our [project page](https://wj-cv.github.io/LECDrive/#demos) for animated driving demonstrations.

## 🏛️ Model Zoo and Data

The following Hugging Face repositories are linked from our project page. Please refer to their contents for available files and release updates.

| Resource | Repository |
|:---------|:-----------|
| Model checkpoints | [wang-jie825/LECDrive](https://huggingface.co/wang-jie825/LECDrive) |
| Dataset resources | [wang-jie825/LECDrive](https://huggingface.co/datasets/wang-jie825/LECDrive) |

The pretrained checkpoints for the three vision foundation models used in **LECDrive** are available for download on [Hugging Face](https://huggingface.co/wang-jie825/LECDrive/tree/main).

## 🏁 Getting Started

The training and inference scripts are now available. Follow the instructions below to train LECDrive, generate predictions, and evaluate its performance.

### 1. Environment

Dependency versions and environment installation instructions will be provided soon.

### 2. Training

Run the following command to launch training:

```bash
USE_DINOV3=True USE_SAM3=True USE_DEPTH=True CONFIG="folder_name" bash run_scripts/dist_sft_navsim.sh
```

Replace `folder_name` with the configuration folder name for your experiment.

### 3. Run Inference

Run the NAVSIM inference script to generate predictions:

```bash
bash run_scripts/inference_navsim.sh
```

### 4. Evaluation

Please follow the official evaluation protocol for your target NAVSIM version:

- [NAVSIM v1.1](https://github.com/autonomousvision/navsim/tree/v1.1)
- [NAVSIM v2.2](https://github.com/autonomousvision/navsim/tree/v2.2)

Refer to the corresponding official documentation for dataset preparation, evaluation setup, and metric computation.

## 🔗 Related Work

Check out our related project:

**[VGGDrive: Empowering Vision-Language Models with Cross-View Geometric Grounding for Autonomous Driving](https://github.com/WJ-CV/VGGDrive)**

[Project Page](https://wj-cv.github.io/VGGDrive/) | [Code](https://github.com/WJ-CV/VGGDrive)

<a id="citation"></a>

## 📌 Citation

If you find this work useful, please consider citing:

```bibtex
@inproceedings{wang2026lecdrive,
  title     = {Think Densely, Act Sparsely: Latent Expert Cognitive Chains for Vision-Language-Action Autonomous Driving},
  author    = {Wang, Jie and Li, Guang and Huang, Zhijian and Li, Jinlong and Dang, Chenxu and Ye, Hangjun and Han, Yahong and Chen, Long},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026}
}
```

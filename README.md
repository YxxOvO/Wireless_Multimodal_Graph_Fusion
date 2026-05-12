# Wireless Multimodal Graph Fusion for IoT-Enabled Scene Understanding

**[Paper]()** · **[中文介绍](#中文介绍)**

## Overview

This repository implements **NCMG (Node-Centric Multimodal Graph)**, a multimodal graph fusion framework for IRS-assisted THz MIMO communication systems. NCMG jointly optimizes beamforming, IRS phase shifts, and sub-carrier bandwidth allocation by unifying visual scene features and wireless channel state information through a heterogeneous graph structure.

![THz MIMO IRS](https://img.shields.io/badge/THz%20MIMO%20IRS-green?style=flat-square)
![Python](https://img.shields.io/badge/Python-3.8+-blue?style=flat-square)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red?style=flat-square)

## Method

### Heterogeneous Graph Construction

Three node types (User / BS / IRS) with 15 meta-path patterns:

| Node Pair | Meta-path | Description |
|-----------|-----------|-------------|
| U-B | UB, BU | User-Base Station direct link |
| U-R | UR, RU | User-IRS reflection link |
| R-B | RB, BR | IRS-Base Station link |
| U-R-B | URB, UBR, BUR, RUB, RBU | Two-hop composite paths |

### Multimodal Fusion

**Wireless Branch**: CSI/RSSI signals encoded via 1D CNN + GRU + Transformer Encoder into node embeddings.

**Visual Branch**: ResNet50 extracts visual features (2048-dim) from scene images. Cosine similarity constructs visual adjacency matrix.

**Cross-Stitch Unit**: Learnable matrix (`E ∈ R^(2d×2d)`) for adaptive cross-modal feature fusion.

### Resource Optimization Output

- **Beamforming**: Complex-domain MLP → power normalization (P_max constraint)
- **IRS Phase Shifts**: Sigmoid activation → `2π·σ(φ)` ∈ [0, 2π]
- **Sub-carrier Bandwidth**: Proportional allocation under total bandwidth constraint

## Baselines

| Model | Modality | Description |
|-------|---------|-------------|
| **NCMG** (Ours) | Wireless + Visual | Dual-branch GNN with Cross-Stitch fusion |
| MHG | Wireless only | Multi-modal heterogeneous graph baseline |
| Visual | Visual only | Vision-dominant baseline |

## Dataset

- Scene images: `car`, `trunk`, `bus` object crops + BS/IRS site images
- Synthetic THz channel parameters: distance, angle, path loss
- Train/Val/Test split: 400 / 20 / 177 samples

## Setup

```bash
pip install torch torchvision torch-scatter
```

## Training

```bash
cd NCMG
python train.py
```

Logs saved to `logs_train/` via TensorBoard.

## Results

- NCMG outperforms MHG and Visual baselines in sum-rate and Jain's fairness index
- Visual modality provides scene-aware priors for resource allocation

## Project Structure

```
Wireless_Multimodal_Graph_Fusion/
├── MHG/          # Wireless-only baseline
│   ├── dataload.py
│   ├── model.py
│   └── trainer.py
├── NCMG/         # Our method
│   ├── dataload.py
│   ├── model.py  # VisualGNN + WirelessGNN + CrossStitch
│   ├── trainer.py
│   └── train.py
├── visual/       # Visual-only baseline
└── dataset/      # Scene images
```

---

## 中文介绍

**NCMG（Node-Centric Multimodal Graph）** 是一种面向物联网场景的**多模态图融合**框架，用于 IRS 辅助的 THz MIMO 通信系统资源联合优化。

### 核心创新

1. **异构图建模**：以节点为中心，将 User、BS、IRS 三类节点统一建模，通过 15 条元路径捕捉多跳关系
2. **跨模态交互**：提出 Cross-Stitch Unit，通过可学习矩阵实现视觉特征与无线信道特征的自适应融合
3. **联合资源优化**：端到端输出 beamforming、IRS相位偏移、子载波带宽分配

### 系统配置

- BS天线数：20，IRS元件数：64（8×8），用户数：6
- 频段：380–400 GHz，子载波数：5
- 训练样本：597（Train 400 / Val 20 / Test 177）

### 引用

```bibtex
@article{ncmg2026,
  title={Wireless Multimodal Graph Fusion for IoT-Enabled Scene Understanding},
  author={},
  year={2026}
}
```

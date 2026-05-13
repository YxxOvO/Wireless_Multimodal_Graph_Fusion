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

**Wireless Branch**: CSI/RSSI signals encoded via 1D CNN + GRU + Transformer Encoder (8-head attention) into node embeddings.

**Visual Branch**: CLIP ViT-B/32 extracts 512-dim visual features from scene images, with ResNet50 fallback. Features projected to 128-dim via learnable linear layer before GNN. Cosine similarity constructs visual adjacency matrix.

**Cross-Stitch Unit with Gate Mechanism**: Dual learnable matrices (`A, B, C, D`) with dynamic gate weighting (`gate_v`, `gate_w`) for adaptive cross-modal feature fusion — replacing static matrix multiplication with content-aware modality mixing.

### Resource Optimization Output

- **Beamforming**: Complex-domain MLP with symmetric real/imag Xavier init → power normalization (P_max constraint)
- **IRS Phase Shifts**: Sigmoid activation → `2π·σ(φ)` ∈ [0, 2π]
- **Sub-carrier Bandwidth**: Proportional allocation under total bandwidth constraint

### Training & Optimization

- **Multi-GPU**: DistributedDataParallel (DDP) via `torchrun` (`--ddp` flag)
- **Memory Efficient**: `.expand()` lazy-copy broadcasting for 21 intermediate tensors (avoids `.repeat()` copy overhead)
- **Fairness Auxiliary Loss**: Jain's Fairness Index integrated as `fairness_weight * (1 - fairness_index)` term
- **Early Stopping**: Patience-based with best-state checkpointing
- **Quantization**: Dynamic int8 quantization available for deployment (`--quantize` flag)

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
# For CLIP visual encoder (recommended):
pip install git+https://github.com/openai/CLIP.git
# Otherwise ResNet50 is used as fallback automatically
```

## Training

```bash
cd NCMG
python train.py                          # single GPU

# Multi-GPU (DDP):
torchrun --nproc_per_node=N train.py --ddp

# With int8 quantization for deployment:
python train.py --quantize
```

Key arguments:
- `--visual_proj_dim`: Visual feature projection dimension (default 128)
- `--fairness_weight`: Fairness auxiliary loss weight (default 0.05)
- `--early_stop_patience`: Early stopping patience (default 50)
- `--use_clip`: Use CLIP ViT-B/32 (default True, auto-fallback to ResNet50)
- `--ddp`: Enable DistributedDataParallel multi-GPU
- `--quantize`: Apply dynamic int8 quantization for inference

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
│   ├── dataload.py   # CLIP ViT-B/32 + ResNet50 fallback, visual projection, 15 meta-paths
│   ├── model.py      # VisualGNN + WirelessGNN + CrossStitch (8-head Transformer, Gate)
│   ├── trainer.py    # DDP support, expand() memory optimization, fairness loss, early stop
│   └── train.py      # Multi-GPU (torchrun), int8 quantization, rank-aware logging
├── visual/       # Visual-only baseline
└── dataset/      # Scene images (car/trunk/bus crops, BS/IRS site)
```

---

## 中文介绍

**NCMG（Node-Centric Multimodal Graph）** 是一种面向物联网场景的**多模态图融合**框架，用于 IRS 辅助的 THz MIMO 通信系统资源联合优化。

### 核心创新

1. **异构图建模**：以节点为中心，将 User、BS、IRS 三类节点统一建模，通过 15 条元路径捕捉多跳关系
2. **跨模态交互**：提出 Cross-Stitch Unit with Gate，可学习矩阵 + 动态门控实现视觉特征与无线信道特征的自适应融合
3. **联合资源优化**：端到端输出 beamforming、IRS相位偏移、子载波带宽分配
4. **多模态视觉编码**：CLIP ViT-B/32 (512维) 提取视觉特征，兼容 ResNet50 fallback，经投影层降至128维入GNN
5. **内存优化**：21 个中间张量使用 `.expand()` 惰性复制，节省显存
6. **公平性约束**：Jain's Fairness Index 作为辅助损失项联合优化
7. **多卡训练**：DDP 分布式支持（torchrun 启动）+ int8 动态量化推理

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

---
id: intro
title: 项目简介
sidebar_label: 🏠 项目简介
slug: /
---

# DL-Demos 深度学习教程

本项目收录深度学习经典算法的 PyTorch 从零实现，配以系统的数学推导和逐行代码解析。

**核心价值**：不只是能跑代码，而是真正理解「论文公式」到「代码实现」的完整推导链条。

---

## 🎨 生成模型系列

| 章节 | 算法 | 核心思想 |
|------|------|---------|
| 1 | [VAE](./generative-models/vae) | 隐变量模型 + 重参数化技巧 |
| 2 | [DDPM](./generative-models/ddpm) | 马尔可夫链扩散 + 去噪分数匹配 |
| 3 | [DDIM](./generative-models/ddim) | 非马尔可夫加速采样（20步 ≈ 1000步）|
| 4 | [PixelCNN](./generative-models/pixelcnn) | 自回归像素生成 + 掩码卷积 |
| 5 | [VQVAE](./generative-models/vqvae) | 离散隐空间 + Codebook 向量量化 |

## ⚙️ 训练技巧系列

| 章节 | 算法 | 核心思想 |
|------|------|---------|
| 1 | [参数初始化](./training-tricks/initialization) | He 初始化 vs 随机初始化 vs 全零初始化 |
| 2 | [正则化](./training-tricks/regularization) | Weight Decay + Dropout 防过拟合 |
| 3 | [优化器](./training-tricks/optimizer) | SGD → Momentum → RMSProp → Adam |

## 🚀 工程实践系列

| 章节 | 算法 | 核心思想 |
|------|------|---------|
| 1 | [傅里叶特征](./engineering/fourier-feature) | 频谱偏置 + Fourier Feature 映射 |
| 2 | [分布式训练](./engineering/distributed) | PyTorch DDP 数据并行 |

---

## 快速开始

```bash
git clone https://github.com/luochang212/DL-Demos.git
cd DL-Demos
python setup.py develop
pip install -r requirements.txt
mkdir work_dirs
```

## 源码仓库

[https://github.com/luochang212/DL-Demos](https://github.com/luochang212/DL-Demos)

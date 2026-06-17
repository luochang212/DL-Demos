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

## 🧠 基础神经网络系列

| 章节 | 算法 | 核心思想 |
|------|------|---------|
| 1 | [Logistic Regression](./fundamentals/logistic-regression) | Sigmoid + 二元交叉熵 |
| 2 | [浅层神经网络](./fundamentals/shallow-network) | 隐藏层 + 链式法则 |
| 3 | [深层 MLP](./fundamentals/deep-network) | 通用前向与反向传播 |
| 4 | [多分类](./fundamentals/multiclass-classification) | Logits + Softmax 交叉熵 |

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

## 🖼️ CNN 与视觉系列

| 章节 | 算法 | 核心思想 |
|------|------|---------|
| 1 | [CNN 基础](./cv/basic-cnn) | 卷积运算 + NumPy 手写实现 |
| 2 | [ResNet](./cv/resnet) | 残差连接 + 深层网络训练 |
| 3 | [Style Transfer](./cv/style-transfer) | VGG 特征 + Gram Matrix |
| 4 | [IoU 与 NMS](./cv/iou-nms) | 目标框重叠度 + 重复预测抑制 |

## 🔗 序列模型系列

| 章节 | 算法 | 核心思想 |
|------|------|---------|
| 1 | [RNN](./sequence-models/rnn) | 循环结构 + 字符级语言模型 |
| 2 | [情感分析](./sequence-models/sentiment-analysis) | GloVe 词向量 + GRU 文本分类 |
| 3 | [Attention](./sequence-models/attention) | 对齐分数 + 上下文向量 |
| 4 | [Transformer](./sequence-models/transformer) | 多头注意力 + Mask |

---

## 快速开始

```bash
git clone https://github.com/luochang212/DL-Demos.git
cd DL-Demos
uv sync --group dev
mkdir work_dirs
```

完整运行与验证方式见[环境、运行与验证](./getting-started/environment)。

> 默认 Python 3.13 环境覆盖 PyTorch 教程。TensorFlow 与 `torchtext` 示例作为 legacy 材料保留。

## 源码仓库

[https://github.com/luochang212/DL-Demos](https://github.com/luochang212/DL-Demos)

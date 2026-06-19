# Fourier Feature — 突破 MLP 的频率瓶颈

## 运行命令

### 烟雾测试

```bash
uv run pytest chapters/engineering/fourier_feature/tests -q
```

### Jupyter 实验

```bash
uv run jupyter notebook chapters/engineering/fourier_feature/image_mlp.ipynb
uv run jupyter notebook chapters/engineering/fourier_feature/kernel_regression.ipynb
```

### 代码入口

```bash
uv run python -c "
from chapters.engineering.fourier_feature.code.model import MLP, FourierFeature
print('MLP and FourierFeature imported successfully')
"
```

## 数据与依赖

- 无需外部数据集。`image_mlp.ipynb` 使用内置的 `misuzu.png` 作为目标图像。
- 依赖 `einops`（已在 `pyproject.toml` 中声明）。

## 输出位置

- Jupyter notebook 内嵌显示图像输出，不产生持久文件。

## 推导

完整符号推导在 [`derivations/formulas.md`](derivations/formulas.md)（NTK → 频谱偏置 → Fourier 特征映射 → Random Features → scale 参数）。

## 参考资料

- Tancik, M., et al. (2020). [Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains](https://arxiv.org/abs/2006.10739).
- Rahimi, A. & Recht, B. (2007). [Random Features for Large-Scale Kernel Machines](https://proceedings.neurips.cc/paper/2007/file/013a006f03dbc5392effeb8f18fda755-Paper.pdf).
- Jacot, A., Gabriel, F., & Hongler, C. (2018). [Neural Tangent Kernel: Convergence and Generalization in Neural Networks](https://arxiv.org/abs/1806.07572).
- Mildenhall, B., et al. (2020). [NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis](https://arxiv.org/abs/2003.08934).

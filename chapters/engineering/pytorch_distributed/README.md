# 分布式训练 — PyTorch DDP 实战

## 运行命令

### 烟雾测试

```bash
uv run pytest chapters/engineering/pytorch_distributed/tests -q
```

### 完整 DDP 运行（需要 CUDA + 多 GPU 或单 GPU 多进程）

```bash
torchrun --nproc_per_node=2 chapters/engineering/pytorch_distributed/code/main.py
```

### 代码入口

```bash
uv run python -c "
from chapters.engineering.pytorch_distributed.main import ToyModel
model = ToyModel()
print(f'ToyModel: {model}')
"
```

## 数据与依赖

- 使用内嵌的 4 条合成数据（`[1, 2, 3, 4]`），无需外部数据集。
- DDP 全流程需要 CUDA 和 NCCL；烟雾测试可在 CPU 上运行。

## 输出位置

- 默认 checkpoint：`tmp.pth`（当前目录）。
- 测试 checkpoint：`work_dirs/pytorch_distributed/test_toy_model.pth`。

## 参考资料

- Goyal, P., et al. (2017). [Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour](https://arxiv.org/abs/1706.02677).
- Li, S., et al. (2020). [PyTorch Distributed: Experiences on Multi-Node Training](https://arxiv.org/abs/2006.15704).
- PyTorch 官方：[Distributed Data Parallel Tutorial](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html).
- 周弈帆（2022-12-19）。[PyTorch 并行训练极简 Demo](https://zhouyifan.net/2022/12/19/20221029-torch-parallel-training/)。

from pathlib import Path

import torch

from chapters.engineering.pytorch_distributed.code.main import ToyModel


def test_toy_model_forward_save_on_cpu():
    # DDP itself requires CUDA/NCCL and multiple ranks; this smoke only
    # exercises the chapter's model code on CPU to verify imports, the forward
    # path, and the checkpoint output location.
    model = ToyModel()
    x = torch.arange(1, 5, dtype=torch.float32).reshape(-1, 1)

    y = model(x)
    assert y.shape == x.shape

    loss = torch.nn.MSELoss()(y, torch.zeros_like(y))
    loss.backward()
    assert torch.isfinite(loss)

    ckpt = Path('work_dirs/pytorch_distributed/test_toy_model.pth')
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), ckpt)
    assert ckpt.exists()

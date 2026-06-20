"""Smoke tests for ResNet PyTorch implementation.

Tests all block types, factory functions, forward passes, shortcut logic,
and device movement.  No TensorFlow dependency.
"""

from pathlib import Path

import torch

from chapters.cv.resnet.code.model import (
    BasicBlock,
    Bottleneck,
    resnet18,
    resnet34,
    resnet50,
    resnet101,
    resnet152,
)


def test_basic_block_forward():
    """BasicBlock with identity shortcut (stride=1, same channels)."""
    block = BasicBlock(64, 64, stride=1)
    x = torch.randn(2, 64, 56, 56)
    out = block(x)
    assert out.shape == x.shape, f'{out.shape} != {x.shape}'


def test_basic_block_downsample():
    """BasicBlock with downsampling shortcut."""
    block = BasicBlock(64, 128, stride=2)
    x = torch.randn(2, 64, 56, 56)
    out = block(x)
    assert out.shape == (2, 128, 28, 28), f'{out.shape}'


def test_bottleneck_forward():
    """Bottleneck with identity shortcut."""
    block = Bottleneck(256, 64)
    x = torch.randn(2, 256, 56, 56)
    out = block(x)
    # Bottleneck expansion = 4 → 64 * 4 = 256 output channels
    assert out.shape == (2, 256, 56, 56), f'{out.shape}'


def test_bottleneck_downsample():
    """Bottleneck with downsampling shortcut."""
    block = Bottleneck(64, 128, stride=2)
    x = torch.randn(2, 64, 56, 56)
    out = block(x)
    # 128 * 4 = 512 output channels, spatial / 2
    assert out.shape == (2, 512, 28, 28), f'{out.shape}'


def test_resnet18_forward():
    """ResNet-18 forward pass on realistic input."""
    model = resnet18(num_classes=10)
    model.eval()
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (2, 10), f'{out.shape}'


def test_resnet50_forward():
    """ResNet-50 forward pass on realistic input."""
    model = resnet50(num_classes=10)
    model.eval()
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (2, 10), f'{out.shape}'


def test_resnet18_small_input():
    """ResNet-18 handles smaller inputs (64x64)."""
    model = resnet18(num_classes=2)
    model.eval()
    x = torch.randn(4, 3, 64, 64)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (4, 2), f'{out.shape}'


def test_all_factory_functions():
    """All five factory functions create valid models."""
    x = torch.randn(1, 3, 224, 224)
    for factory in [resnet18, resnet34, resnet50, resnet101, resnet152]:
        model = factory(num_classes=5)
        model.eval()
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, 5), f'{factory.__name__}: {out.shape}'


def test_train_step_and_backward():
    """Training step: loss, backward, optimizer step all succeed."""
    model = resnet18(num_classes=1)
    x = torch.randn(4, 3, 224, 224)
    y = torch.rand(4, 1)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    pred = model(x)
    loss = loss_fn(pred, y)
    assert torch.isfinite(loss), f'Loss not finite: {loss}'

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def test_checkpoint_save_load():
    """Model checkpoint round-trip."""
    model = resnet18(num_classes=1)
    model.eval()
    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        out_before = model(x)

    ckpt_dir = Path('work_dirs/resnet/tests')
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / 'test_ckpt.pth'
    torch.save(model.state_dict(), ckpt_path)

    model2 = resnet18(num_classes=1)
    model2.load_state_dict(torch.load(ckpt_path, map_location='cpu', weights_only=True))
    model2.eval()
    with torch.no_grad():
        out_after = model2(x)

    assert torch.allclose(out_before, out_after), 'Checkpoint round-trip mismatch'


def test_device_movement():
    """Model and tensors move to available device successfully."""
    model = resnet18(num_classes=1)
    device = torch.device('cpu')
    model.to(device)
    x = torch.randn(2, 3, 224, 224).to(device)
    with torch.no_grad():
        out = model(x)
    assert out.device == device
    # Also test CUDA when available
    if torch.cuda.is_available():
        device = torch.device('cuda')
        model.to(device)
        x = torch.randn(2, 3, 224, 224).to(device)
        with torch.no_grad():
            out = model(x)
        assert out.device.type == 'cuda'

"""Training and evaluation entrypoint for ResNet.

Usage:
  Smoke test (synthetic data, 2 epochs):
    uv run python -m chapters.cv.resnet.code.main --smoke

  Full training (requires cat/dog dataset at data/archive/dataset):
    uv run python -m chapters.cv.resnet.code.main --model resnet18 --epochs 20
"""

import argparse
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from chapters.common.utils.device import resolve_device
from chapters.cv.basic_cnn.code.dataset import get_cat_set
from chapters.cv.resnet.code.model import resnet18, resnet50

WORK_DIR = Path('work_dirs/resnet')


def _make_synthetic_data():
    """Return 32 synthetic cat/dog samples for smoke testing."""
    n = 32
    X = np.random.randn(n, 3, 224, 224).astype(np.float32)
    Y = np.random.randint(0, 2, (n, 1)).astype(np.float32)
    n_test = 8
    test_X = np.random.randn(n_test, 3, 224, 224).astype(np.float32)
    test_Y = np.random.randint(0, 2, (n_test, 1)).astype(np.float32)
    return X, Y, test_X, test_Y


def train_epoch(
    model,
    loader_X,
    loader_Y,
    optimizer,
    loss_fn,
    device,
    batch_size: int = 16,
):
    """Train one epoch over mini-batches."""
    m = loader_X.shape[0]
    indices = np.random.permutation(m)
    X = loader_X[indices]
    Y = loader_Y[indices]
    n_batches = math.ceil(m / batch_size)

    total_loss = 0.0
    for i in range(n_batches):
        start = i * batch_size
        end = min(start + batch_size, m)
        bx = torch.from_numpy(X[start:end]).to(device)
        by = torch.from_numpy(Y[start:end]).float().to(device)

        pred = model(bx)
        loss = loss_fn(pred, by)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / n_batches


@torch.no_grad()
def evaluate(model, X, Y, device):
    """Binary classification accuracy."""
    X_t = torch.from_numpy(X).to(device)
    Y_t = torch.from_numpy(Y).to(device)
    logits = model(X_t)
    preds = (torch.sigmoid(logits) > 0.5).float()
    acc = (preds == Y_t).float().mean().item()
    return acc


def smoke(device: torch.device):
    """Quick smoke run with synthetic data."""
    print('--- ResNet smoke test ---')
    model = resnet18(num_classes=1).to(device)
    train_X, train_Y, test_X, test_Y = _make_synthetic_data()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()

    for epoch in range(2):
        loss = train_epoch(
            model,
            train_X,
            train_Y,
            optimizer,
            loss_fn,
            device,
            batch_size=16,
        )
        acc = evaluate(model, test_X, test_Y, device)
        print(f'  Epoch {epoch}: loss={loss:.4f}, test_acc={acc:.3f}')

    WORK_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path = WORK_DIR / 'smoke_checkpoint.pth'
    torch.save(model.state_dict(), ckpt_path)
    print(f'  Checkpoint saved to {ckpt_path}')
    print('--- Smoke test passed ---')


def main():
    parser = argparse.ArgumentParser(description='ResNet training')
    parser.add_argument('--model', default='resnet18', choices=['resnet18', 'resnet50'])
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--data-root', default='data/archive/dataset')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--pretrained', action='store_true')
    args = parser.parse_args()

    device = resolve_device(args.device)
    print(f'Using device: {device}')

    if args.smoke:
        smoke(device)
        return

    # Full data training
    train_X, train_Y, test_X, test_Y = get_cat_set(
        args.data_root,
        train_size=2000,
        test_size=400,
        format='nchw',
    )
    print(f'Train: {train_X.shape}, {train_Y.shape}')
    print(f'Test:  {test_X.shape}, {test_Y.shape}')

    if args.model == 'resnet18':
        model = resnet18(num_classes=1, pretrained=args.pretrained)
    else:
        model = resnet50(num_classes=1, pretrained=args.pretrained)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.BCEWithLogitsLoss()

    for epoch in range(args.epochs):
        loss = train_epoch(
            model,
            train_X,
            train_Y,
            optimizer,
            loss_fn,
            device,
            batch_size=args.batch_size,
        )
        acc = evaluate(model, test_X, test_Y, device)
        print(f'Epoch {epoch:3d}: loss={loss:.4f}, test_acc={acc:.3f}')

    WORK_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path = WORK_DIR / f'{args.model}.pth'
    torch.save(model.state_dict(), ckpt_path)
    print(f'Saved checkpoint to {ckpt_path}')


if __name__ == '__main__':
    main()

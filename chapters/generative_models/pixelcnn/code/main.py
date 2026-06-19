import argparse
import os
import time

import cv2
import einops
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from chapters.common.utils.device import resolve_device
from chapters.generative_models.pixelcnn.code.dataset import (
    get_dataloader,
    get_img_shape,
)
from chapters.generative_models.pixelcnn.code.model import GatedPixelCNN, PixelCNN

DEFAULT_BATCH_SIZE = 128
DEFAULT_COLOR_LEVEL = 8
DEFAULT_CHECKPOINT = 'work_dirs/pixelcnn/model_1_8.pth'
DEFAULT_OUTPUT = 'work_dirs/pixelcnn/pixelcnn_1_8.jpg'
DEFAULT_SMOKE_CHECKPOINT = 'work_dirs/pixelcnn/smoke_model.pth'
DEFAULT_SMOKE_OUTPUT = 'work_dirs/pixelcnn/smoke_sample.jpg'


def ensure_parent_dir(path):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def train(
    model,
    device,
    model_path,
    dataloader,
    color_level=DEFAULT_COLOR_LEVEL,
    epochs=40,
):
    ensure_parent_dir(model_path)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), 1e-3)
    loss_fn = nn.CrossEntropyLoss()
    tic = time.time()
    for e in range(epochs):
        total_loss = 0
        for x, _ in dataloader:
            current_batch_size = x.shape[0]
            x = x.to(device)
            y = torch.ceil(x * (color_level - 1)).long()
            y = y.squeeze(1)
            predict_y = model(x)
            loss = loss_fn(predict_y, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * current_batch_size
        total_loss /= len(dataloader.dataset)
        toc = time.time()
        torch.save(model.state_dict(), model_path)
        print(f'epoch {e} loss: {total_loss} elapsed {(toc - tic):.2f}s')
    print('Done')


def sample(
    model,
    device,
    model_path,
    output_path,
    n_sample=81,
    color_level=DEFAULT_COLOR_LEVEL,
):
    ensure_parent_dir(output_path)
    model.eval()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    C, H, W = get_img_shape()  # (1, 28, 28)
    x = torch.zeros((n_sample, C, H, W)).to(device)
    with torch.no_grad():
        for i in range(H):
            for j in range(W):
                output = model(x)
                prob_dist = F.softmax(output[:, :, i, j], -1)
                pixel = torch.multinomial(prob_dist, 1).float() / (color_level - 1)
                x[:, :, i, j] = pixel

    imgs = x * 255
    imgs = imgs.clamp(0, 255)
    imgs = einops.rearrange(
        imgs, '(b1 b2) c h w -> (b1 h) (b2 w) c', b1=int(n_sample**0.5)
    )

    imgs = imgs.detach().cpu().numpy().astype(np.uint8)

    cv2.imwrite(output_path, imgs)


def build_model(model_id, color_level=DEFAULT_COLOR_LEVEL, small=False):
    if small:
        return GatedPixelCNN(1, 2, 2, False, color_level)
    models = [
        PixelCNN(15, 128, 32, False, color_level),
        GatedPixelCNN(15, 128, 32, False, color_level),
    ]
    return models[model_id]


def smoke(
    device, checkpoint=DEFAULT_SMOKE_CHECKPOINT, output_path=DEFAULT_SMOKE_OUTPUT
):
    torch.manual_seed(0)
    color_level = 2
    model = build_model(1, color_level=color_level, small=True)
    x = torch.rand(4, *get_img_shape())
    y = torch.zeros(4, dtype=torch.long)
    dataloader = DataLoader(TensorDataset(x, y), batch_size=2, shuffle=False)
    train(model, device, checkpoint, dataloader, color_level=color_level, epochs=1)
    reloaded = build_model(1, color_level=color_level, small=True)
    sample(
        reloaded,
        device,
        checkpoint,
        output_path,
        n_sample=1,
        color_level=color_level,
    )
    print(f'smoke ok: checkpoint={checkpoint} output={output_path}')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['smoke', 'train', 'sample'], required=True)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--checkpoint', default=DEFAULT_CHECKPOINT)
    parser.add_argument('--output', default=DEFAULT_OUTPUT)
    parser.add_argument('--data-root', default='data/mnist')
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--batch-size', type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument('--model-id', type=int, default=1, choices=[0, 1])
    parser.add_argument('--color-level', type=int, default=DEFAULT_COLOR_LEVEL)
    parser.add_argument('--n-sample', type=int, default=81)
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs('work_dirs/pixelcnn', exist_ok=True)
    device = resolve_device(args.device)
    if args.mode == 'smoke':
        checkpoint = (
            args.checkpoint
            if args.checkpoint != DEFAULT_CHECKPOINT
            else DEFAULT_SMOKE_CHECKPOINT
        )
        output = args.output if args.output != DEFAULT_OUTPUT else DEFAULT_SMOKE_OUTPUT
        smoke(device, checkpoint, output)
        return

    model = build_model(args.model_id, args.color_level)
    if args.mode == 'train':
        dataloader = get_dataloader(args.batch_size, root=args.data_root)
        train(
            model,
            device,
            args.checkpoint,
            dataloader,
            color_level=args.color_level,
            epochs=args.epochs,
        )
        return

    sample(
        model,
        device,
        args.checkpoint,
        args.output,
        n_sample=args.n_sample,
        color_level=args.color_level,
    )


if __name__ == '__main__':
    main()

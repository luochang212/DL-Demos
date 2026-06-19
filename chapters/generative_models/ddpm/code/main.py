import argparse
import os
import time

import cv2
import einops
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from chapters.common.utils.device import resolve_device
from chapters.generative_models.ddpm.code.dataset import get_dataloader, get_img_shape
from chapters.generative_models.ddpm.code.ddpm_simple import DDPM
from chapters.generative_models.ddpm.code.network import (
    build_network,
    convnet_big_cfg,
    convnet_medium_cfg,
    convnet_small_cfg,
    unet_1_cfg,
    unet_res_cfg,
)

DEFAULT_BATCH_SIZE = 512
DEFAULT_EPOCHS = 100
DEFAULT_CHECKPOINT = 'work_dirs/ddpm/model_unet_res.pth'
DEFAULT_OUTPUT = 'work_dirs/ddpm/diffusion.jpg'
DEFAULT_SMOKE_CHECKPOINT = 'work_dirs/ddpm/smoke_model.pth'
DEFAULT_SMOKE_OUTPUT = 'work_dirs/ddpm/smoke_sample.jpg'


def ensure_parent_dir(path):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def train(
    ddpm: DDPM,
    net,
    dataloader,
    device='cuda',
    ckpt_path=DEFAULT_CHECKPOINT,
    epochs=DEFAULT_EPOCHS,
):
    ensure_parent_dir(ckpt_path)
    print('batch size:', dataloader.batch_size)
    n_steps = ddpm.n_steps
    net = net.to(device)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), 1e-3)

    tic = time.time()
    for e in range(epochs):
        total_loss = 0

        for x, _ in dataloader:
            current_batch_size = x.shape[0]
            x = x.to(device)
            t = torch.randint(0, n_steps, (current_batch_size,)).to(device)
            eps = torch.randn_like(x).to(device)
            x_t = ddpm.sample_forward(x, t, eps)
            eps_theta = net(x_t, t.reshape(current_batch_size, 1))
            loss = loss_fn(eps_theta, eps)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * current_batch_size
        total_loss /= len(dataloader.dataset)
        toc = time.time()
        torch.save(net.state_dict(), ckpt_path)
        print(f'epoch {e} loss: {total_loss} elapsed {(toc - tic):.2f}s')
    print('Done')


def sample_imgs(ddpm, net, output_path, n_sample=81, device='cuda', simple_var=True):
    ensure_parent_dir(output_path)
    net = net.to(device)
    net = net.eval()
    with torch.no_grad():
        shape = (n_sample, *get_img_shape())  # 1, 3, 28, 28
        imgs = (
            ddpm.sample_backward(shape, net, device=device, simple_var=simple_var)
            .detach()
            .cpu()
        )
        imgs = (imgs + 1) / 2 * 255
        imgs = imgs.clamp(0, 255)
        imgs = einops.rearrange(
            imgs, '(b1 b2) c h w -> (b1 h) (b2 w) c', b1=int(n_sample**0.5)
        )

        imgs = imgs.numpy().astype(np.uint8)

        cv2.imwrite(output_path, imgs)


def smoke(
    device, checkpoint=DEFAULT_SMOKE_CHECKPOINT, output_path=DEFAULT_SMOKE_OUTPUT
):
    torch.manual_seed(0)
    n_steps = 4
    ddpm = DDPM(device, n_steps)
    net = build_network(convnet_small_cfg, n_steps)
    x = torch.rand(4, *get_img_shape()) * 2 - 1
    y = torch.zeros(4, dtype=torch.long)
    dataloader = DataLoader(TensorDataset(x, y), batch_size=2, shuffle=False)
    train(ddpm, net, dataloader, device=device, ckpt_path=checkpoint, epochs=1)
    reloaded = build_network(convnet_small_cfg, n_steps).to(device)
    reloaded.load_state_dict(torch.load(checkpoint, map_location=device))
    sample_imgs(ddpm, reloaded, output_path, n_sample=1, device=device)
    print(f'smoke ok: checkpoint={checkpoint} output={output_path}')


configs = [
    convnet_small_cfg,
    convnet_medium_cfg,
    convnet_big_cfg,
    unet_1_cfg,
    unet_res_cfg,
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['smoke', 'train', 'sample'], required=True)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--checkpoint', default=DEFAULT_CHECKPOINT)
    parser.add_argument('--output', default=DEFAULT_OUTPUT)
    parser.add_argument('--data-root', default='data/mnist')
    parser.add_argument('--epochs', type=int, default=DEFAULT_EPOCHS)
    parser.add_argument('--batch-size', type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument('--n-steps', type=int, default=1000)
    parser.add_argument('--config-id', type=int, default=4, choices=range(len(configs)))
    parser.add_argument('--n-sample', type=int, default=81)
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs('work_dirs/ddpm', exist_ok=True)
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

    config = configs[args.config_id]
    net = build_network(config, args.n_steps)
    ddpm = DDPM(device, args.n_steps)

    if args.mode == 'train':
        dataloader = get_dataloader(args.batch_size, root=args.data_root)
        train(
            ddpm,
            net,
            dataloader,
            device=device,
            ckpt_path=args.checkpoint,
            epochs=args.epochs,
        )
        return

    net.load_state_dict(torch.load(args.checkpoint, map_location=device))
    sample_imgs(
        ddpm,
        net,
        args.output,
        n_sample=args.n_sample,
        device=device,
    )


if __name__ == '__main__':
    main()

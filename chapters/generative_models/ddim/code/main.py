import argparse
import os
import time

import cv2
import einops
import torch
import torch.nn as nn

from chapters.common.utils.device import resolve_device
from chapters.generative_models.ddim.code.configs import configs
from chapters.generative_models.ddim.code.dataset import get_dataloader
from chapters.generative_models.ddim.code.ddim import DDIM
from chapters.generative_models.ddim.code.ddpm import DDPM
from chapters.generative_models.ddim.code.network import UNet

DEFAULT_CHECKPOINT = 'work_dirs/ddim/mnist.pth'
DEFAULT_OUTPUT = 'work_dirs/ddim/sample.jpg'


def ensure_parent_dir(path):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def train(
    ddpm: DDPM,
    net,
    dataset_type,
    resolution=None,
    batch_size=512,
    n_epochs=50,
    device='cuda',
    ckpt_path=DEFAULT_CHECKPOINT,
    data_root='data/mnist',
):
    ensure_parent_dir(ckpt_path)
    print('batch size:', batch_size)
    dataloader = get_dataloader(
        dataset_type, batch_size, resolution=resolution, data_root=data_root
    )
    train_on_dataloader(ddpm, net, dataloader, n_epochs, device, ckpt_path)


def train_on_dataloader(ddpm, net, dataloader, n_epochs, device, ckpt_path):
    net = net.to(device)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), 2e-4)
    n_steps = ddpm.n_steps

    tic = time.time()
    for e in range(n_epochs):
        total_loss = 0
        total_count = 0

        for x in dataloader:
            if isinstance(x, list | tuple):
                x = x[0]
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
            total_count += current_batch_size
        total_loss /= total_count
        toc = time.time()
        torch.save(net.state_dict(), ckpt_path)
        print(f'epoch {e} loss: {total_loss} elapsed {(toc - tic):.2f}s')
    print('Done')


def sample_imgs(
    ddpm,
    net,
    output_path,
    img_shape,
    n_sample=64,
    device='cuda',
    simple_var=True,
    to_bgr=False,
    **kwargs,
):
    ensure_parent_dir(output_path)
    if img_shape[1] >= 256:
        max_batch_size = 16
    elif img_shape[1] >= 128:
        max_batch_size = 64
    else:
        max_batch_size = 256

    net = net.to(device)
    net = net.eval()

    index = 0
    with torch.no_grad():
        while n_sample > 0:
            if n_sample >= max_batch_size:
                batch_size = max_batch_size
            else:
                batch_size = n_sample
            n_sample -= batch_size
            shape = (batch_size, *img_shape)
            imgs = (
                ddpm.sample_backward(
                    shape, net, device=device, simple_var=simple_var, **kwargs
                )
                .detach()
                .cpu()
            )
            imgs = (imgs + 1) / 2 * 255
            imgs = imgs.clamp(0, 255).to(torch.uint8)

            img_list = einops.rearrange(imgs, 'n c h w -> n h w c').numpy()
            output_dir = os.path.splitext(output_path)[0]
            os.makedirs(output_dir, exist_ok=True)
            for i, img in enumerate(img_list):
                if to_bgr:
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                cv2.imwrite(f'{output_dir}/{i + index}.jpg', img)

            # First iteration
            if index == 0:
                imgs = einops.rearrange(
                    imgs, '(b1 b2) c h w -> (b1 h) (b2 w) c', b1=int(batch_size**0.5)
                )
                imgs = imgs.numpy()
                if to_bgr:
                    imgs = cv2.cvtColor(imgs, cv2.COLOR_RGB2BGR)
                cv2.imwrite(output_path, imgs)

            index += batch_size


def build_model(cfg, n_steps):
    return UNet(
        n_steps,
        cfg['img_shape'],
        cfg['channels'],
        cfg['pe_dim'],
        cfg.get('with_attn', False),
        cfg.get('norm_type', 'ln'),
    )


def smoke(
    device='cpu',
    checkpoint='work_dirs/ddim/smoke_model.pth',
    output='work_dirs/ddim/smoke_sample.jpg',
):
    device = resolve_device(device)
    n_steps = 4
    img_shape = [1, 8, 8]
    cfg = {'img_shape': img_shape, 'channels': [4, 8], 'pe_dim': 8}
    net = build_model(cfg, n_steps)
    ddpm = DDPM(device, n_steps)
    x = torch.randn(2, *img_shape)
    dataloader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x), batch_size=2
    )
    train_on_dataloader(ddpm, net, dataloader, 1, device, checkpoint)

    net.load_state_dict(torch.load(checkpoint, map_location=device))
    ddim = DDIM(device, n_steps)
    sample_imgs(
        ddim,
        net,
        output,
        img_shape,
        n_sample=1,
        device=device,
        simple_var=False,
        ddim_step=2,
        eta=0,
    )
    print(f'smoke checkpoint: {checkpoint}')
    print(f'smoke sample: {output}')


def parse_args():
    parser = argparse.ArgumentParser(description='DDIM chapter runner')
    parser.add_argument('--mode', choices=['smoke', 'train', 'sample'], default='smoke')
    parser.add_argument('--device', default='auto')
    parser.add_argument('--config-id', type=int, default=0)
    parser.add_argument('--n-steps', type=int, default=1000)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument('--data-root', default='data/mnist')
    parser.add_argument('--checkpoint', default=None)
    parser.add_argument('--output', default=DEFAULT_OUTPUT)
    parser.add_argument('--n-sample', type=int, default=16)
    parser.add_argument('--ddim-step', type=int, default=20)
    parser.add_argument('--eta', type=float, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    device = resolve_device(args.device)

    if args.mode == 'smoke':
        smoke(device=device)
        return

    os.makedirs('work_dirs/ddim', exist_ok=True)
    cfg = configs[args.config_id]
    n_steps = args.n_steps
    model_path = args.checkpoint or cfg['model_path']
    img_shape = cfg['img_shape']
    to_bgr = cfg['dataset_type'] == 'CelebAHQ'

    net = build_model(cfg, n_steps)
    ddpm = DDPM(device, n_steps)

    if args.mode == 'train':
        train(
            ddpm,
            net,
            cfg['dataset_type'],
            resolution=(img_shape[1], img_shape[2]),
            batch_size=args.batch_size or cfg['batch_size'],
            n_epochs=args.epochs if args.epochs is not None else cfg['n_epochs'],
            device=device,
            ckpt_path=model_path,
            data_root=args.data_root,
        )
        return

    net.load_state_dict(torch.load(model_path, map_location=device))
    ddim = DDIM(device, n_steps)
    sample_imgs(
        ddim,
        net,
        args.output,
        img_shape,
        n_sample=args.n_sample,
        device=device,
        simple_var=False,
        ddim_step=args.ddim_step,
        eta=args.eta,
        to_bgr=to_bgr,
    )


if __name__ == '__main__':
    main()

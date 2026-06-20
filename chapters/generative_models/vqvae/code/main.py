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
from chapters.generative_models.vqvae.code.configs import get_cfg
from chapters.generative_models.vqvae.code.dataset import get_dataloader
from chapters.generative_models.vqvae.code.model import VQVAE
from chapters.generative_models.vqvae.code.pixelcnn_model import PixelCNNWithEmbedding

USE_LMDB = False
DEFAULT_VQVAE_CHECKPOINT = 'work_dirs/vqvae/model.pth'
DEFAULT_GEN_CHECKPOINT = 'work_dirs/vqvae/gen_model.pth'
DEFAULT_SMOKE_VQVAE_CHECKPOINT = 'work_dirs/vqvae/smoke_model.pth'
DEFAULT_SMOKE_GEN_CHECKPOINT = 'work_dirs/vqvae/smoke_gen_model.pth'


def ensure_parent_dir(path):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def train_vqvae(
    model: VQVAE,
    img_shape=None,
    device='cuda',
    ckpt_path='work_dirs/vqvae/model.pth',
    batch_size=64,
    dataset_type='MNIST',
    lr=1e-3,
    n_epochs=100,
    l_w_embedding=1,
    l_w_commitment=0.25,
):
    ensure_parent_dir(ckpt_path)
    print('batch size:', batch_size)
    dataloader = get_dataloader(
        dataset_type, batch_size, img_shape=img_shape, use_lmdb=USE_LMDB
    )
    model.to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr)
    mse_loss = nn.MSELoss()
    tic = time.time()
    for e in range(n_epochs):
        total_loss = 0

        for x in dataloader:
            if isinstance(x, (list, tuple)):
                x = x[0]
            current_batch_size = x.shape[0]
            x = x.to(device)

            x_hat, ze, zq = model(x)
            l_reconstruct = mse_loss(x, x_hat)
            l_embedding = mse_loss(ze.detach(), zq)
            l_commitment = mse_loss(ze, zq.detach())
            loss = (
                l_reconstruct
                + l_w_embedding * l_embedding
                + l_w_commitment * l_commitment
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * current_batch_size
        total_loss /= len(dataloader.dataset)
        toc = time.time()
        torch.save(model.state_dict(), ckpt_path)
        print(f'epoch {e} loss: {total_loss} elapsed {(toc - tic):.2f}s')
    print('Done')


def train_vqvae_on_dataloader(
    model: VQVAE,
    dataloader,
    device='cuda',
    ckpt_path=DEFAULT_VQVAE_CHECKPOINT,
    lr=1e-3,
    n_epochs=1,
    l_w_embedding=1,
    l_w_commitment=0.25,
):
    ensure_parent_dir(ckpt_path)
    model.to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr)
    mse_loss = nn.MSELoss()
    for e in range(n_epochs):
        total_loss = 0
        for x in dataloader:
            if isinstance(x, (list, tuple)):
                x = x[0]
            current_batch_size = x.shape[0]
            x = x.to(device)
            x_hat, ze, zq = model(x)
            loss = (
                mse_loss(x, x_hat)
                + l_w_embedding * mse_loss(ze.detach(), zq)
                + l_w_commitment * mse_loss(ze, zq.detach())
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * current_batch_size
        total_loss /= len(dataloader.dataset)
        torch.save(model.state_dict(), ckpt_path)
        print(f'epoch {e} loss: {total_loss}')


def train_generative_model(
    vqvae: VQVAE,
    model,
    img_shape=None,
    device='cuda',
    ckpt_path='work_dirs/vqvae/gen_model.pth',
    dataset_type='MNIST',
    batch_size=64,
    n_epochs=50,
):
    ensure_parent_dir(ckpt_path)
    print('batch size:', batch_size)
    dataloader = get_dataloader(
        dataset_type, batch_size, img_shape=img_shape, use_lmdb=USE_LMDB
    )
    vqvae.to(device)
    vqvae.eval()
    model.to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), 1e-3)
    loss_fn = nn.CrossEntropyLoss()
    tic = time.time()
    for e in range(n_epochs):
        total_loss = 0
        for x in dataloader:
            current_batch_size = x.shape[0]
            with torch.no_grad():
                x = x.to(device)
                x = vqvae.encode(x)

            predict_x = model(x)
            loss = loss_fn(predict_x, x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * current_batch_size
        total_loss /= len(dataloader.dataset)
        toc = time.time()
        torch.save(model.state_dict(), ckpt_path)
        print(f'epoch {e} loss: {total_loss} elapsed {(toc - tic):.2f}s')
    print('Done')


def reconstruct(model, x, device, dataset_type='MNIST'):
    os.makedirs('work_dirs/vqvae', exist_ok=True)
    model.to(device)
    model.eval()
    with torch.no_grad():
        x_hat, _, _ = model(x)
    n = x.shape[0]
    n1 = int(n**0.5)
    x_cat = torch.concat((x, x_hat), 3)
    x_cat = einops.rearrange(x_cat, '(n1 n2) c h w -> (n1 h) (n2 w) c', n1=n1)
    x_cat = (x_cat.clip(0, 1) * 255).cpu().numpy().astype(np.uint8)
    if dataset_type == 'CelebA' or dataset_type == 'CelebAHQ':
        x_cat = cv2.cvtColor(x_cat, cv2.COLOR_RGB2BGR)
    cv2.imwrite(f'work_dirs/vqvae/reconstruct_{dataset_type}.jpg', x_cat)


def sample_imgs(
    vqvae: VQVAE, gen_model, img_shape, n_sample=81, device='cuda', dataset_type='MNIST'
):
    os.makedirs('work_dirs/vqvae', exist_ok=True)
    vqvae = vqvae.to(device)
    vqvae.eval()
    gen_model = gen_model.to(device)
    gen_model.eval()

    C, H, W = img_shape
    H, W = vqvae.get_latent_HW((C, H, W))
    input_shape = (n_sample, H, W)
    x = torch.zeros(input_shape).to(device).to(torch.long)
    with torch.no_grad():
        for i in range(H):
            for j in range(W):
                output = gen_model(x)
                prob_dist = F.softmax(output[:, :, i, j], -1)
                pixel = torch.multinomial(prob_dist, 1)
                x[:, i, j] = pixel[:, 0]

    imgs = vqvae.decode(x)

    imgs = imgs * 255
    imgs = imgs.clip(0, 255)
    imgs = einops.rearrange(
        imgs, '(n1 n2) c h w -> (n1 h) (n2 w) c', n1=int(n_sample**0.5)
    )

    imgs = imgs.detach().cpu().numpy().astype(np.uint8)
    if dataset_type == 'CelebA' or dataset_type == 'CelebAHQ':
        imgs = cv2.cvtColor(imgs, cv2.COLOR_RGB2BGR)

    cv2.imwrite(f'work_dirs/vqvae/sample_{dataset_type}.jpg', imgs)


def smoke(
    device,
    vqvae_ckpt=DEFAULT_SMOKE_VQVAE_CHECKPOINT,
    gen_ckpt=DEFAULT_SMOKE_GEN_CHECKPOINT,
):
    torch.manual_seed(0)
    img_shape = (1, 28, 28)
    vqvae = VQVAE(input_dim=1, dim=4, n_embedding=8)
    gen_model = PixelCNNWithEmbedding(1, 4, 4, False, 8)
    x = torch.rand(4, *img_shape)
    dataloader = DataLoader(TensorDataset(x), batch_size=2, shuffle=False)
    train_vqvae_on_dataloader(
        vqvae,
        dataloader,
        device=device,
        ckpt_path=vqvae_ckpt,
        n_epochs=1,
    )
    vqvae.load_state_dict(torch.load(vqvae_ckpt, map_location=device))
    reconstruct(vqvae, x[0:1].to(device), device)
    torch.save(gen_model.state_dict(), gen_ckpt)
    sample_imgs(vqvae, gen_model, img_shape, n_sample=1, device=device)
    print(f'smoke ok: vqvae={vqvae_ckpt} gen={gen_ckpt}')


def main():
    os.makedirs('work_dirs/vqvae', exist_ok=True)

    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['smoke', 'full'], default='full')
    parser.add_argument('--device', default='auto')
    parser.add_argument('-c', type=int, default=0)
    parser.add_argument('-d', type=int, default=0)
    args = parser.parse_args()
    device = resolve_device(args.device)
    if args.mode == 'smoke':
        smoke(device)
        return

    cfg = get_cfg(args.c)

    # Use the device resolved from --device flag (args.d is kept for backward
    # compatibility but no longer overrides --device)
    if args.d != 0:
        import warnings

        warnings.warn(
            'The -d argument is deprecated. Use --device instead.',
            DeprecationWarning,
            stacklevel=2,
        )
        device = resolve_device(f'cuda:{args.d}')
    # else: device is already resolved from --device above

    img_shape = cfg['img_shape']

    vqvae = VQVAE(img_shape[0], cfg['dim'], cfg['n_embedding'])
    gen_model = PixelCNNWithEmbedding(
        cfg['pixelcnn_n_blocks'],
        cfg['pixelcnn_dim'],
        cfg['pixelcnn_linear_dim'],
        False,
        cfg['n_embedding'],
    )
    # 1. Train VQVAE
    train_vqvae(
        vqvae,
        img_shape=(img_shape[1], img_shape[2]),
        device=device,
        ckpt_path=cfg['vqvae_path'],
        batch_size=cfg['batch_size'],
        dataset_type=cfg['dataset_type'],
        lr=cfg['lr'],
        n_epochs=cfg['n_epochs'],
        l_w_embedding=cfg['l_w_embedding'],
        l_w_commitment=cfg['l_w_commitment'],
    )

    # 2. Test VQVAE by visualizaing reconstruction result
    vqvae.load_state_dict(torch.load(cfg['vqvae_path'], map_location=device))
    dataloader = get_dataloader(
        cfg['dataset_type'], 16, img_shape=(img_shape[1], img_shape[2])
    )
    img = next(iter(dataloader)).to(device)
    reconstruct(vqvae, img, device, cfg['dataset_type'])

    # 3. Train Generative model (Gated PixelCNN in our project)
    vqvae.load_state_dict(torch.load(cfg['vqvae_path'], map_location=device))

    train_generative_model(
        vqvae,
        gen_model,
        img_shape=(img_shape[1], img_shape[2]),
        device=device,
        ckpt_path=cfg['gen_model_path'],
        dataset_type=cfg['dataset_type'],
        batch_size=cfg['batch_size_2'],
        n_epochs=cfg['n_epochs_2'],
    )

    # 4. Sample VQVAE
    vqvae.load_state_dict(torch.load(cfg['vqvae_path'], map_location=device))
    gen_model.load_state_dict(torch.load(cfg['gen_model_path'], map_location=device))
    sample_imgs(
        vqvae,
        gen_model,
        cfg['img_shape'],
        device=device,
        dataset_type=cfg['dataset_type'],
    )


if __name__ == '__main__':
    main()

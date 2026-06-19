import argparse
import os
from time import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision.transforms import ToPILImage

from chapters.common.utils.device import resolve_device
from chapters.generative_models.vae.code.dataset import get_dataloader
from chapters.generative_models.vae.code.model import VAE

# Hyperparameters
n_epochs = 10
kl_weight = 0.00025
lr = 0.005
DEFAULT_CHECKPOINT = 'work_dirs/vae/model.pth'
DEFAULT_OUTPUT = 'work_dirs/vae/tmp.jpg'
DEFAULT_SMOKE_CHECKPOINT = 'work_dirs/vae/smoke_model.pth'
DEFAULT_SMOKE_OUTPUT = 'work_dirs/vae/smoke_sample.jpg'


def ensure_parent_dir(path):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def loss_fn(y, y_hat, mean, logvar):
    recons_loss = F.mse_loss(y_hat, y)
    kl_loss = torch.mean(
        -0.5 * torch.sum(1 + logvar - mean**2 - torch.exp(logvar), 1), 0
    )
    loss = recons_loss + kl_loss * kl_weight
    return loss


def train(device, dataloader, model, checkpoint=DEFAULT_CHECKPOINT, epochs=n_epochs):
    optimizer = torch.optim.Adam(model.parameters(), lr)
    dataset_len = len(dataloader.dataset)
    ensure_parent_dir(checkpoint)

    begin_time = time()
    # train
    for i in range(epochs):
        loss_sum = 0
        for x in dataloader:
            x = x.to(device)
            y_hat, mean, logvar = model(x)
            loss = loss_fn(x, y_hat, mean, logvar)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * x.shape[0]
        loss_sum /= dataset_len
        training_time = time() - begin_time
        minute = int(training_time // 60)
        second = int(training_time % 60)
        print(f'epoch {i}: loss {loss_sum} {minute}:{second}')
        torch.save(model.state_dict(), checkpoint)


def smoke(
    device, checkpoint=DEFAULT_SMOKE_CHECKPOINT, output_path=DEFAULT_SMOKE_OUTPUT
):
    """Run a real CLI smoke path without requiring CelebA."""
    torch.manual_seed(0)
    x = torch.rand(4, 3, 64, 64)
    dataloader = DataLoader(TensorDataset(x), batch_size=2, shuffle=False)
    model = VAE(hiddens=[2, 4], latent_dim=3).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr)
    ensure_parent_dir(checkpoint)

    model.train()
    for (batch,) in dataloader:
        batch = batch.to(device)
        y_hat, mean, logvar = model(batch)
        loss = loss_fn(batch, y_hat, mean, logvar)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    torch.save(model.state_dict(), checkpoint)
    reloaded = VAE(hiddens=[2, 4], latent_dim=3).to(device)
    reloaded.load_state_dict(torch.load(checkpoint, map_location=device))
    generate(device, reloaded, output_path)
    print(f'smoke ok: checkpoint={checkpoint} output={output_path}')


def reconstruct(device, dataloader, model, output_path=DEFAULT_OUTPUT):
    model.eval()
    ensure_parent_dir(output_path)
    batch = next(iter(dataloader))
    x = batch[0:1, ...].to(device)
    output = model(x)[0]
    output = output[0].detach().cpu()
    input = batch[0].detach().cpu()
    combined = torch.cat((output, input), 1)
    img = ToPILImage()(combined)
    img.save(output_path)


def generate(device, model, output_path=DEFAULT_OUTPUT):
    model.eval()
    ensure_parent_dir(output_path)
    output = model.sample(device)
    output = output[0].detach().cpu()
    img = ToPILImage()(output)
    img.save(output_path)


def load_model(checkpoint, device):
    model = VAE().to(device)
    if not os.path.exists(checkpoint):
        raise FileNotFoundError(f'Checkpoint not found: {checkpoint}')
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--mode', choices=['train', 'reconstruct', 'generate', 'smoke'], required=True
    )
    parser.add_argument('--device', default='auto')
    parser.add_argument('--checkpoint', default=DEFAULT_CHECKPOINT)
    parser.add_argument('--output', default=DEFAULT_OUTPUT)
    parser.add_argument('--dataset', choices=['celeba', 'mnist'], default='celeba')
    parser.add_argument('--data-root')
    parser.add_argument('--epochs', type=int, default=n_epochs)
    args = parser.parse_args()

    device = resolve_device(args.device)
    os.makedirs('work_dirs/vae', exist_ok=True)

    if args.mode == 'train':
        model = VAE().to(device)
        train(
            device,
            get_dataloader(args.dataset, args.data_root),
            model,
            args.checkpoint,
            args.epochs,
        )
    elif args.mode == 'reconstruct':
        reconstruct(
            device,
            get_dataloader(args.dataset, args.data_root),
            load_model(args.checkpoint, device),
            args.output,
        )
    elif args.mode == 'generate':
        generate(device, load_model(args.checkpoint, device), args.output)
    else:
        checkpoint = (
            args.checkpoint
            if args.checkpoint != DEFAULT_CHECKPOINT
            else DEFAULT_SMOKE_CHECKPOINT
        )
        output = args.output if args.output != DEFAULT_OUTPUT else DEFAULT_SMOKE_OUTPUT
        smoke(device, checkpoint, output)


if __name__ == '__main__':
    main()

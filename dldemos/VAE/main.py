import argparse
import os
from time import time

import torch
import torch.nn.functional as F
from torchvision.transforms import ToPILImage

from dldemos.utils.device import resolve_device
from dldemos.VAE.load_celebA import get_dataloader
from dldemos.VAE.model import VAE

# Hyperparameters
n_epochs = 10
kl_weight = 0.00025
lr = 0.005


def loss_fn(y, y_hat, mean, logvar):
    recons_loss = F.mse_loss(y_hat, y)
    kl_loss = torch.mean(
        -0.5 * torch.sum(1 + logvar - mean**2 - torch.exp(logvar), 1), 0
    )
    loss = recons_loss + kl_loss * kl_weight
    return loss


def train(device, dataloader, model, checkpoint='dldemos/VAE/model.pth'):
    optimizer = torch.optim.Adam(model.parameters(), lr)
    dataset_len = len(dataloader.dataset)

    begin_time = time()
    # train
    for i in range(n_epochs):
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


def reconstruct(device, dataloader, model):
    model.eval()
    batch = next(iter(dataloader))
    x = batch[0:1, ...].to(device)
    output = model(x)[0]
    output = output[0].detach().cpu()
    input = batch[0].detach().cpu()
    combined = torch.cat((output, input), 1)
    img = ToPILImage()(combined)
    img.save('work_dirs/tmp.jpg')


def generate(device, model):
    model.eval()
    output = model.sample(device)
    output = output[0].detach().cpu()
    img = ToPILImage()(output)
    img.save('work_dirs/tmp.jpg')


def load_model(checkpoint, device):
    model = VAE().to(device)
    if not os.path.exists(checkpoint):
        raise FileNotFoundError(f'Checkpoint not found: {checkpoint}')
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--mode', choices=['train', 'reconstruct', 'generate'], required=True
    )
    parser.add_argument('--device', default='auto')
    parser.add_argument('--checkpoint', default='dldemos/VAE/model.pth')
    args = parser.parse_args()

    device = resolve_device(args.device)
    os.makedirs('work_dirs', exist_ok=True)

    if args.mode == 'train':
        model = VAE().to(device)
        train(device, get_dataloader(), model, args.checkpoint)
    elif args.mode == 'reconstruct':
        reconstruct(device, get_dataloader(), load_model(args.checkpoint, device))
    else:
        generate(device, load_model(args.checkpoint, device))


if __name__ == '__main__':
    main()

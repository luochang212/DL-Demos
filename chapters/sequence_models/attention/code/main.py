import argparse
import os

import torch

from chapters.common.utils.device import resolve_device
from chapters.sequence_models.attention.code.dataset import (
    generate_date,
    generate_date_data,
    get_dataloader,
    itos,
    stoi,
)
from chapters.sequence_models.attention.code.model import AttentionModel


def sequence_accuracy(prediction, target):
    return torch.all(prediction == target, dim=-1).float().mean()


def train(device, epochs=30, checkpoint='work_dirs/attention/model.pth'):
    train_path = 'chapters/sequence_models/attention/train.txt'
    if not os.path.exists(train_path):
        generate_date_data(50000, train_path)
    train_dataloader = get_dataloader(train_path)

    model = AttentionModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    citerion = torch.nn.CrossEntropyLoss()

    for epoch in range(epochs):
        loss_sum = 0
        dataset_len = len(train_dataloader.dataset)

        for x, lengths, y in train_dataloader:
            x = x.to(device)
            y = y.to(device)
            hat_y = model(x, lengths)
            n, Tx, _ = hat_y.shape
            hat_y = torch.reshape(hat_y, (n * Tx, -1))
            label_y = torch.reshape(y, (n * Tx,))
            loss = citerion(hat_y, label_y)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

            loss_sum += loss * n

        print(f'Epoch {epoch}. loss: {loss_sum / dataset_len}')

    os.makedirs(os.path.dirname(checkpoint), exist_ok=True)
    torch.save(model.state_dict(), checkpoint)
    return model


def test(model, device):
    test_path = 'chapters/sequence_models/attention/test.txt'
    if not os.path.exists(test_path):
        generate_date_data(10000, test_path)
    test_dataloader = get_dataloader(test_path)

    accuracy = 0
    dataset_len = len(test_dataloader.dataset)

    model.eval()
    for x, lengths, y in test_dataloader:
        x = x.to(device)
        y = y.to(device)
        with torch.no_grad():
            hat_y = model(x, lengths)
        prediction = torch.argmax(hat_y, 2)
        accuracy += sequence_accuracy(prediction, y) * prediction.shape[0]

    print(f'Accuracy: {accuracy / dataset_len}')


def infer(model, device):
    model.eval()
    for _ in range(5):
        x, y = generate_date()
        origin_x = x
        x = stoi(x).unsqueeze(0).to(device)
        lengths = torch.tensor([x.shape[1]])
        with torch.no_grad():
            hat_y = model(x, lengths)
        hat_y = hat_y.squeeze(0).argmax(1)
        hat_y = itos(hat_y)
        print(f'input: {origin_x}, prediction: {hat_y}, gt: {y}')


def smoke(device='cpu'):
    """Run a minimal training + test + inference smoke test on CPU."""
    print('[smoke] generating synthetic data and training for 2 epochs ...')
    train_path = 'chapters/sequence_models/attention/train.txt'
    generate_date_data(5000, train_path)
    # Train with fewer epochs for smoke
    model = train(device, epochs=2, checkpoint='work_dirs/attention/model.pth')
    print('[smoke] testing ...')
    test_path = 'chapters/sequence_models/attention/test.txt'
    generate_date_data(1000, test_path)
    test(model, device)
    print('[smoke] inference samples ...')
    infer(model, device)
    print('[smoke] done.')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--mode', choices=['train', 'test', 'infer', 'smoke'], default='train'
    )
    parser.add_argument('--device', default='auto')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--checkpoint', default='work_dirs/attention/model.pth')
    args = parser.parse_args()

    device = resolve_device(args.device)

    if args.mode == 'smoke':
        smoke(device)
    elif args.mode == 'train':
        train(device, epochs=args.epochs, checkpoint=args.checkpoint)
    elif args.mode == 'test':
        model = AttentionModel().to(device)
        model.load_state_dict(
            torch.load(args.checkpoint, map_location=device, weights_only=True)
        )
        test(model, device)
    elif args.mode == 'infer':
        model = AttentionModel().to(device)
        model.load_state_dict(
            torch.load(args.checkpoint, map_location=device, weights_only=True)
        )
        infer(model, device)


if __name__ == '__main__':
    main()

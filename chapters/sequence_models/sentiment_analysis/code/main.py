import argparse
import os

import torch

from chapters.common.utils.device import resolve_device
from chapters.sequence_models.sentiment_analysis.code.dataset import (
    _get_glove,
    get_dataloader,
    get_tokenizer,
)
from chapters.sequence_models.sentiment_analysis.code.model import RNN


def train(device, checkpoint='work_dirs/sentiment_analysis/rnn.pth', epochs=100):
    train_dataloader, _ = get_dataloader()
    model = RNN().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    citerion = torch.nn.BCELoss()
    for epoch in range(epochs):
        loss_sum = 0
        dataset_len = len(train_dataloader.dataset)

        for x, lengths, y in train_dataloader:
            batchsize = y.shape[0]
            x = x.to(device)
            y = y.to(device)
            hat_y = model(x, lengths)
            hat_y = hat_y.squeeze(-1)
            loss = citerion(hat_y, y)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

            loss_sum += loss * batchsize

        print(f'Epoch {epoch}. loss: {loss_sum / dataset_len}')

    os.makedirs(os.path.dirname(checkpoint), exist_ok=True)
    torch.save(model.state_dict(), checkpoint)
    return model


def test(model, device):
    _, test_dataloader = get_dataloader()

    accuracy = 0
    dataset_len = len(test_dataloader.dataset)
    model.eval()
    for x, lengths, y in test_dataloader:
        x = x.to(device)
        y = y.to(device)
        with torch.no_grad():
            hat_y = model(x, lengths)
        hat_y.squeeze_(1)
        predictions = torch.where(hat_y > 0.5, 1, 0)
        score = torch.sum(torch.where(predictions == y, 1, 0))
        accuracy += score.item()
    accuracy /= dataset_len

    print(f'Accuracy: {accuracy}')


def infer(model, device):
    glove = _get_glove()
    tokenizer = get_tokenizer()
    article = (
        'U.S. stock indexes fell Tuesday, driven by expectations for '
        'tighter Federal Reserve policy and an energy crisis in Europe. '
        'Stocks around the globe have come under pressure in recent weeks '
        'as worries about tighter monetary policy in the U.S. and a '
        'darkening economic outlook in Europe have led investors to '
        'sell riskier assets.'
    )

    model.eval()
    x = glove.get_vecs_by_tokens(tokenizer(article)).unsqueeze(0).to(device)
    lengths = torch.tensor([x.shape[1]])
    with torch.no_grad():
        hat_y = model(x, lengths)
    hat_y = hat_y.squeeze_().item()
    result = 'positive' if hat_y > 0.5 else 'negative'
    print(f'Inference result: {result} ({hat_y:.4f})')


def smoke(device='cpu'):
    """Run minimal training + test + inference smoke test on CPU."""
    print('[smoke] training for 2 epochs ...')
    model = train(device, checkpoint='work_dirs/sentiment_analysis/rnn.pth', epochs=2)
    print('[smoke] testing ...')
    test(model, device)
    print('[smoke] inference ...')
    infer(model, device)
    print('[smoke] done.')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--mode', choices=['train', 'test', 'infer', 'smoke'], default='train'
    )
    parser.add_argument('--device', default='auto')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--checkpoint', default='work_dirs/sentiment_analysis/rnn.pth')
    args = parser.parse_args()

    device = resolve_device(args.device)

    if args.mode == 'smoke':
        smoke(device)
    elif args.mode == 'train':
        train(device, epochs=args.epochs, checkpoint=args.checkpoint)
    elif args.mode == 'test':
        model = RNN().to(device)
        model.load_state_dict(
            torch.load(args.checkpoint, map_location=device, weights_only=True)
        )
        test(model, device)
    elif args.mode == 'infer':
        model = RNN().to(device)
        model.load_state_dict(
            torch.load(args.checkpoint, map_location=device, weights_only=True)
        )
        infer(model, device)


if __name__ == '__main__':
    main()

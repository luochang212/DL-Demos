import argparse
import os

import torch

from chapters.common.utils.device import resolve_device
from chapters.sequence_models.basic_rnn.code.dataset import (
    get_dataloader_and_max_length,
    words_to_label_array,
    words_to_onehot,
)
from chapters.sequence_models.basic_rnn.code.model import RNN1, RNN2

test_words = [
    'apple',
    'appll',
    'appla',
    'apply',
    'bear',
    'beer',
    'berr',
    'beee',
    'car',
    'cae',
    'cat',
    'cac',
    'caq',
    'query',
    'queee',
    'queue',
    'queen',
    'quest',
    'quess',
    'quees',
]


def train_rnn1(device, checkpoint='work_dirs/basic_rnn/rnn1.pth'):
    dataloader, max_length = get_dataloader_and_max_length(19)

    model = RNN1().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    citerion = torch.nn.CrossEntropyLoss()
    for epoch in range(5):
        loss_sum = 0
        dataset_len = len(dataloader.dataset)

        for y in dataloader:
            y = y.to(device)
            hat_y = model(y)
            n, Tx, _ = hat_y.shape
            hat_y = torch.reshape(hat_y, (n * Tx, -1))
            y = torch.reshape(y, (n * Tx, -1))
            label_y = torch.argmax(y, 1)
            loss = citerion(hat_y, label_y)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

            loss_sum += loss.item() * n

        print(f'Epoch {epoch}. loss: {loss_sum / dataset_len}')

    os.makedirs(os.path.dirname(checkpoint), exist_ok=True)
    torch.save(model.state_dict(), checkpoint)
    return model


def train_rnn2(device, checkpoint='work_dirs/basic_rnn/rnn2.pth'):
    dataloader, max_length = get_dataloader_and_max_length(19, is_onehot=False)

    model = RNN2().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    citerion = torch.nn.CrossEntropyLoss()
    for epoch in range(5):
        loss_sum = 0
        dataset_len = len(dataloader.dataset)

        for y in dataloader:
            y = y.to(device)
            hat_y = model(y)
            n, Tx, _ = hat_y.shape
            hat_y = torch.reshape(hat_y, (n * Tx, -1))
            label_y = torch.reshape(y, (n * Tx,))
            loss = citerion(hat_y, label_y)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

            loss_sum += loss.item() * n

        print(f'Epoch {epoch}. loss: {loss_sum / dataset_len}')

    os.makedirs(os.path.dirname(checkpoint), exist_ok=True)
    torch.save(model.state_dict(), checkpoint)
    return model


def test_language_model(model, is_onehot=True):
    _, max_length = get_dataloader_and_max_length(19)
    if is_onehot:
        test_word = words_to_onehot(test_words, max_length)
    else:
        test_word = words_to_label_array(test_words, max_length)
    test_word = test_word.to(next(model.parameters()).device)
    probs = model.language_model(test_word)
    for word, prob in zip(test_words, probs):
        print(f'{word}: {prob}')


def sample(model):
    words = []
    for _ in range(20):
        word = model.sample_word()
        words.append(word)
    print(*words)


def load_model(model_name, checkpoint, device):
    if not os.path.exists(checkpoint):
        raise FileNotFoundError(f'Checkpoint not found: {checkpoint}')
    model = RNN1() if model_name == 'rnn1' else RNN2()
    model.load_state_dict(
        torch.load(checkpoint, map_location=device, weights_only=True)
    )
    return model.to(device).eval()


def smoke(device='cpu'):
    """Run a minimal training + eval + sample smoke test on CPU."""
    print('[smoke] training rnn1 for 1 epoch on synthetic-like data ...')
    dataloader, max_length = get_dataloader_and_max_length(19)
    model = RNN1(hidden_units=8).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    citerion = torch.nn.CrossEntropyLoss()

    for y in dataloader:
        y = y.to(device)
        hat_y = model(y)
        n, Tx, _ = hat_y.shape
        hat_y = torch.reshape(hat_y, (n * Tx, -1))
        y = torch.reshape(y, (n * Tx, -1))
        label_y = torch.argmax(y, 1)
        loss = citerion(hat_y, label_y)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        break

    print(f'[smoke] loss: {loss.item():.4f}')
    print('[smoke] evaluating language model ...')
    test_language_model(model, is_onehot=True)
    print('[smoke] sampling words ...')
    sample(model)
    print('[smoke] done.')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['rnn1', 'rnn2'])
    parser.add_argument('--mode', choices=['train', 'evaluate', 'sample', 'smoke'])
    parser.add_argument('--device', default='auto')
    parser.add_argument('--checkpoint')
    args = parser.parse_args()

    device = resolve_device(args.device)

    if args.mode == 'smoke':
        smoke(device)
        return

    if args.model is None:
        parser.error('--model is required for train/evaluate/sample modes')

    checkpoint = args.checkpoint or f'work_dirs/basic_rnn/{args.model}.pth'
    if args.mode == 'train':
        train = train_rnn1 if args.model == 'rnn1' else train_rnn2
        train(device, checkpoint)
        return

    model = load_model(args.model, checkpoint, device)
    if args.mode == 'evaluate':
        test_language_model(model, args.model == 'rnn1')
    else:
        sample(model)


if __name__ == '__main__':
    main()

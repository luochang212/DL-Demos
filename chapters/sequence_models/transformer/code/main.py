import argparse
import os
import time

import torch
import torch.nn as nn

from chapters.common.utils.device import resolve_device
from chapters.sequence_models.transformer.code.data_load import (
    encode_source,
    get_batch_indices,
    idx_to_sentence,
    load_cn_vocab,
    load_en_vocab,
    load_train_data,
    maxlen,
)
from chapters.sequence_models.transformer.code.model import Transformer

# Default config
batch_size = 64
lr = 0.0001
d_model = 512
d_ff = 2048
n_layers = 6
heads = 8
dropout_rate = 0.2
PAD_ID = 0


def greedy_decode(model, x, start_id, end_id, pad_id=PAD_ID, max_len=maxlen):
    y = torch.full((x.shape[0], max_len), pad_id, dtype=torch.long, device=x.device)
    y[:, 0] = start_id
    finished = torch.zeros(x.shape[0], dtype=torch.bool, device=x.device)
    for i in range(1, max_len):
        logits = model(x, y[:, :i])
        next_token = torch.argmax(logits[:, -1], dim=-1)
        next_token = torch.where(finished, pad_id, next_token)
        y[:, i] = next_token
        finished |= next_token == end_id
        if torch.all(finished):
            return y[:, : i + 1]
    return y


def train(device, checkpoint='work_dirs/transformer/model.pth', epochs=60):
    cn2idx, idx2cn = load_cn_vocab()
    en2idx, idx2en = load_en_vocab()
    X, Y = load_train_data()

    print_interval = 100

    model = Transformer(
        len(en2idx),
        len(cn2idx),
        PAD_ID,
        d_model,
        d_ff,
        n_layers,
        heads,
        dropout_rate,
        maxlen,
    )
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr)
    citerion = nn.CrossEntropyLoss(ignore_index=PAD_ID)
    tic = time.time()
    cnter = 0

    for epoch in range(epochs):
        for index, _ in get_batch_indices(len(X), batch_size):
            x_batch = torch.LongTensor(X[index]).to(device)
            y_batch = torch.LongTensor(Y[index]).to(device)
            y_input = y_batch[:, :-1]
            y_label = y_batch[:, 1:]
            y_hat = model(x_batch, y_input)

            y_label_mask = y_label != PAD_ID
            preds = torch.argmax(y_hat, -1)
            correct = preds == y_label
            acc = torch.sum(y_label_mask * correct) / torch.sum(y_label_mask)

            n, seq_len = y_label.shape
            y_hat = torch.reshape(y_hat, (n * seq_len, -1))
            y_label = torch.reshape(y_label, (n * seq_len,))
            loss = citerion(y_hat, y_label)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()

            if cnter % print_interval == 0:
                toc = time.time()
                interval = toc - tic
                minutes = int(interval // 60)
                seconds = int(interval % 60)
                print(
                    f'{cnter:08d} {minutes:02d}:{seconds:02d}'
                    f' loss: {loss.item()} acc: {acc.item()}'
                )
            cnter += 1

    os.makedirs(os.path.dirname(checkpoint), exist_ok=True)
    torch.save(model.state_dict(), checkpoint)
    print(f'Model saved to {checkpoint}')
    return model


def translate(model, device, input_sentence=None):
    cn2idx, idx2cn = load_cn_vocab()
    en2idx, idx2en = load_en_vocab()

    if input_sentence is None:
        input_sentence = ['we', 'should', 'protect', 'environment']

    x_batch = torch.LongTensor([encode_source(input_sentence, en2idx)]).to(device)

    input_str = idx_to_sentence(x_batch[0], idx2en, True)
    print(f'Input : {input_str}')

    with torch.no_grad():
        y_output = greedy_decode(
            model, x_batch, cn2idx['<S>'], cn2idx['</S>'], PAD_ID, maxlen
        )
    output_str = idx_to_sentence(y_output[0], idx2cn, True)
    print(f'Output: {output_str}')


def smoke(device='cpu'):
    """Run minimal training + translation smoke test on CPU."""
    print('[smoke] downloading data and training for 1 epoch ...')
    model = train(device, checkpoint='work_dirs/transformer/model.pth', epochs=1)
    print('[smoke] translating ...')
    translate(model, device)
    print('[smoke] done.')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--mode', choices=['train', 'translate', 'smoke'], default='train'
    )
    parser.add_argument('--device', default='auto')
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--checkpoint', default='work_dirs/transformer/model.pth')
    args = parser.parse_args()

    device = resolve_device(args.device)

    if args.mode == 'smoke':
        smoke(device)
    elif args.mode == 'train':
        train(device, epochs=args.epochs, checkpoint=args.checkpoint)
    elif args.mode == 'translate':
        cn2idx, idx2cn = load_cn_vocab()
        en2idx, idx2en = load_en_vocab()
        model = Transformer(
            len(en2idx),
            len(cn2idx),
            PAD_ID,
            d_model,
            d_ff,
            n_layers,
            heads,
            dropout_rate,
            maxlen,
        )
        model.to(device)
        model.eval()
        model.load_state_dict(
            torch.load(args.checkpoint, map_location=device, weights_only=True)
        )
        translate(model, device)


if __name__ == '__main__':
    main()

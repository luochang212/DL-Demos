import torch

from chapters.sequence_models.transformer.data_load import (
    encode_source,
    idx_to_sentence,
    load_cn_vocab,
    load_en_vocab,
    maxlen,
)
from chapters.sequence_models.transformer.model import Transformer

# Config
batch_size = 1
lr = 0.0001
d_model = 512
d_ff = 2048
n_layers = 6
heads = 8
dropout_rate = 0.2
n_epochs = 60

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


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cn2idx, idx2cn = load_cn_vocab()
    en2idx, idx2en = load_en_vocab()

    model = Transformer(
        len(en2idx),
        len(cn2idx),
        0,
        d_model,
        d_ff,
        n_layers,
        heads,
        dropout_rate,
        maxlen,
    )
    model.to(device)
    model.eval()

    model_path = 'chapters/sequence_models/transformer/model.pth'
    model.load_state_dict(torch.load(model_path, map_location=device))

    my_input = ['we', 'should', 'protect', 'environment']
    x_batch = torch.LongTensor([encode_source(my_input, en2idx)]).to(device)

    input_sentence = idx_to_sentence(x_batch[0], idx2en, True)
    print(input_sentence)

    with torch.no_grad():
        y_output = greedy_decode(
            model, x_batch, cn2idx['<S>'], cn2idx['</S>'], PAD_ID, maxlen
        )
    output_sentence = idx_to_sentence(y_output[0], idx2cn, True)
    print(output_sentence)


if __name__ == '__main__':
    main()

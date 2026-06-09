# Modify from
# https://github.com/P3n9W31/transformer-pytorch/master/data_load.py

import codecs
import os
import random

import numpy as np
import regex
import requests

# Words whose occurred less than min_cnt are encoded as <UNK>.
min_cnt = 0
# Maximum number of words in a sentence.
maxlen = 50

source_train = 'dldemos/Transformer/data/cn.txt'
target_train = 'dldemos/Transformer/data/en.txt'
source_test = 'dldemos/Transformer/data/cn.test.txt'
target_test = 'dldemos/Transformer/data/en.test.txt'


def load_vocab(language):
    assert language in ['cn', 'en']
    vocab = [
        line.split()[0]
        for line in codecs.open(
            'dldemos/Transformer/data/{}.txt.vocab.tsv'.format(language), 'r', 'utf-8'
        )
        .read()
        .splitlines()
        if int(line.split()[1]) >= min_cnt
    ]
    word2idx = {word: idx for idx, word in enumerate(vocab)}
    idx2word = {idx: word for idx, word in enumerate(vocab)}
    return word2idx, idx2word


def load_cn_vocab():
    word2idx, idx2word = load_vocab('cn')
    return word2idx, idx2word


def load_en_vocab():
    word2idx, idx2word = load_vocab('en')
    return word2idx, idx2word


def create_data(english_sents, chinese_sents):
    cn2idx, _ = load_cn_vocab()
    en2idx, _ = load_en_vocab()

    # Index
    x_list, y_list, Sources, Targets = [], [], [], []
    for source_sent, target_sent in zip(english_sents, chinese_sents):
        x = [
            en2idx.get(word, 1) for word in ('<S> ' + source_sent + ' </S>').split()
        ]  # 1: OOV, </S>: End of Text
        y = [cn2idx.get(word, 1) for word in ('<S> ' + target_sent + ' </S>').split()]
        if max(len(x), len(y)) <= maxlen:
            x_list.append(np.array(x))
            y_list.append(np.array(y))
            Sources.append(source_sent)
            Targets.append(target_sent)

    # Pad
    X = np.zeros([len(x_list), maxlen], np.int32)
    Y = np.zeros([len(y_list), maxlen], np.int32)
    for i, (x, y) in enumerate(zip(x_list, y_list)):
        X[i] = np.pad(x, [0, maxlen - len(x)], 'constant', constant_values=(0, 0))
        Y[i] = np.pad(y, [0, maxlen - len(y)], 'constant', constant_values=(0, 0))

    return X, Y, Sources, Targets


def load_data(data_type):
    if data_type == 'train':
        source, target = source_train, target_train
    elif data_type == 'test':
        source, target = source_test, target_test
    assert data_type in ['train', 'test']
    cn_sents = [
        regex.sub(r"[^\s\p{L}']", '', line)
        for line in codecs.open(source, 'r', 'utf-8').read().split('\n')
        if line and line[0] != '<'
    ]
    en_sents = [
        regex.sub(r"[^\s\p{L}']", '', line)
        for line in codecs.open(target, 'r', 'utf-8').read().split('\n')
        if line and line[0] != '<'
    ]

    X, Y, Sources, Targets = create_data(en_sents, cn_sents)
    return X, Y, Sources, Targets


def load_train_data():
    X, Y, _, _ = load_data('train')
    return X, Y


def load_test_data():
    X, Y, _, _ = load_data('test')
    return X, Y


def get_batch_indices(total_length, batch_size):
    if total_length <= 0 or batch_size <= 0:
        raise ValueError('total_length and batch_size must be positive')
    indices = list(range(total_length))
    random.shuffle(indices)
    for start in range(0, total_length, batch_size):
        batch = indices[start : start + batch_size]
        yield batch, start + len(batch)


def idx_to_sentence(arr, vocab, insert_space=False):
    words = []
    for id in arr:
        word = vocab[id.item()]
        if word == '</S>':
            break
        if word not in {'<S>', '<PAD>'}:
            words.append(word)
    return (' ' if insert_space else '').join(words)


def download(url, dir, name=None):
    os.makedirs(dir, exist_ok=True)
    if name is None:
        name = url.split('/')[-1]
    path = os.path.join(dir, name)
    if not os.path.exists(path):
        print(f'Install {name} ...')
        open(path, 'wb').write(requests.get(url).content)
        print('Install successfully.')


def download_data():
    data_dir = 'dldemos/Transformer/data'
    urls = [
        (
            'https://raw.githubusercontent.com/P3n9W31/transformer-pytorch/'
            'master/corpora/cn.txt'
        ),
        (
            'https://raw.githubusercontent.com/P3n9W31/transformer-pytorch/'
            'master/corpora/en.txt'
        ),
        (
            'https://raw.githubusercontent.com/P3n9W31/transformer-pytorch/'
            'master/preprocessed/cn.txt.vocab.tsv'
        ),
        (
            'https://raw.githubusercontent.com/P3n9W31/transformer-pytorch/'
            'master/preprocessed/en.txt.vocab.tsv'
        ),
    ]
    for url in urls:
        download(url, data_dir)


if __name__ == '__main__':
    download_data()

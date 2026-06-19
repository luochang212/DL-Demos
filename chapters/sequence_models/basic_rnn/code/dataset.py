import torch
from torch.utils.data import DataLoader, Dataset

from chapters.sequence_models.basic_rnn.code.constant import (
    EMBEDDING_LENGTH,
    LETTER_MAP,
)
from chapters.sequence_models.basic_rnn.read_imdb import (
    read_imdb_vocab,
    read_imdb_words,
)


def words_to_label_array(words, max_length):
    if isinstance(words, str):
        words = [words]
    words = [word + ' ' for word in words]
    batch = len(words)
    tensor = torch.zeros(batch, max_length, dtype=torch.long)
    for i in range(batch):
        for j, letter in enumerate(words[i]):
            tensor[i][j] = LETTER_MAP[letter]

    return tensor


def words_to_onehot(words, max_length):
    if isinstance(words, str):
        words = [words]
    words = [word + ' ' for word in words]
    batch = len(words)
    tensor = torch.zeros(batch, max_length, EMBEDDING_LENGTH)
    for i in range(batch):
        word_length = len(words[i])
        for j in range(max_length):
            if j < word_length:
                tensor[i][j][LETTER_MAP[words[i][j]]] = 1
            else:
                tensor[i][j][0] = 1

    return tensor


class WordDataset(Dataset):
    def __init__(self, words, max_length, is_onehot=True):
        super().__init__()
        n_words = len(words)
        self.words = words
        self.n_words = n_words
        self.max_length = max_length
        self.is_onehot = is_onehot

    def __len__(self):
        return self.n_words

    def __getitem__(self, index):
        """return the (one-hot) encoding vector of a word."""
        word = self.words[index] + ' '
        word_length = len(word)
        if self.is_onehot:
            tensor = torch.zeros(self.max_length, EMBEDDING_LENGTH)
            for i in range(self.max_length):
                if i < word_length:
                    tensor[i][LETTER_MAP[word[i]]] = 1
                else:
                    tensor[i][0] = 1
        else:
            tensor = torch.zeros(self.max_length, dtype=torch.long)
            for i in range(word_length):
                tensor[i] = LETTER_MAP[word[i]]

        return tensor


def get_dataloader_and_max_length(limit_length=None, is_onehot=True, is_vocab=True):

    if is_vocab:
        words = read_imdb_vocab()
    else:
        words = read_imdb_words(n_files=200)

    max_length = 0
    for word in words:
        max_length = max(max_length, len(word))

    if limit_length is not None and max_length > limit_length:
        words = [w for w in words if len(w) <= limit_length]
        max_length = limit_length

    # for <EOS> (space)
    max_length += 1

    dataset = WordDataset(words, max_length, is_onehot)
    return DataLoader(dataset, batch_size=256), max_length

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from chapters.sequence_models.sentiment_analysis.code.model import GLOVE_DIM
from chapters.sequence_models.sentiment_analysis.read_imdb import read_imdb

_glove = None


def _get_glove():
    global _glove
    if _glove is None:
        from torchtext.vocab import GloVe

        _glove = GloVe(name='6B', dim=GLOVE_DIM)
    return _glove


def get_tokenizer():
    from torchtext.data import get_tokenizer

    return get_tokenizer('basic_english')


class IMDBDataset(Dataset):
    def __init__(self, is_train=True, dir='data/aclImdb'):
        super().__init__()
        self.tokenizer = get_tokenizer()
        pos_lines = read_imdb(dir, 'pos', is_train)
        neg_lines = read_imdb(dir, 'neg', is_train)
        self.lines = pos_lines + neg_lines
        self.pos_length = len(pos_lines)
        self.neg_length = len(neg_lines)

    def __len__(self):
        return self.pos_length + self.neg_length

    def __getitem__(self, index):
        glove = _get_glove()
        sentence = self.tokenizer(self.lines[index])
        x = glove.get_vecs_by_tokens(sentence)
        label = 1 if index < self.pos_length else 0
        return x, label


def get_dataloader(dir='data/aclImdb'):

    def collate_fn(batch):
        x, y = zip(*batch)
        lengths = torch.tensor([len(sentence) for sentence in x])
        x_pad = pad_sequence(x, batch_first=True)
        y = torch.Tensor(y)
        return x_pad, lengths, y

    train_dataloader = DataLoader(
        IMDBDataset(True, dir), batch_size=32, shuffle=True, collate_fn=collate_fn
    )
    test_dataloader = DataLoader(
        IMDBDataset(False, dir), batch_size=32, shuffle=True, collate_fn=collate_fn
    )
    return train_dataloader, test_dataloader

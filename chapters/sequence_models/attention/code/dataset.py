import os
import random

import torch
from babel.dates import format_date
from faker import Faker
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

faker = Faker()
format_list = [
    'short',
    'medium',
    'long',
    'full',
    'd MMM YYY',
    'd MMMM YYY',
    'dd/MM/YYY',
    'dd-MM-YYY',
    'EE d, MMM YYY',
    'EEEE d, MMMM YYY',
]


def generate_date():
    format = random.choice(format_list)
    date_obj = faker.date_object()
    formated_date = format_date(date_obj, format=format, locale='en')
    return formated_date, date_obj


def generate_date_data(count, filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w') as fp:
        for _ in range(count):
            formated_date, date_obj = generate_date()
            fp.write(f'{formated_date}\t{date_obj}\n')


def load_date_data(filename):
    with open(filename, 'r') as fp:
        lines = fp.readlines()
        return [line.strip('\n').split('\t') for line in lines]


def stoi(str):
    return torch.LongTensor([ord(char) for char in str])


def itos(arr):
    return ''.join([chr(x) for x in arr])


class DateDataset(Dataset):
    def __init__(self, lines):
        self.lines = lines

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, index):
        line = self.lines[index]
        return stoi(line[0]), stoi(line[1])


def get_dataloader(filename, batch_size=32):

    def collate_fn(batch):
        x, y = zip(*batch)
        lengths = torch.tensor([len(sequence) for sequence in x])
        x_pad = pad_sequence(x, batch_first=True)
        y_pad = pad_sequence(y, batch_first=True)
        return x_pad, lengths, y_pad

    lines = load_date_data(filename)
    dataset = DateDataset(lines)
    return DataLoader(dataset, batch_size, collate_fn=collate_fn)

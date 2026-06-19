import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence

GLOVE_DIM = 100


class RNN(torch.nn.Module):
    def __init__(self, hidden_units=64, dropout_rate=0.5):
        super().__init__()
        self.drop = nn.Dropout(dropout_rate)
        self.rnn = nn.GRU(GLOVE_DIM, hidden_units, 1, batch_first=True)
        self.linear = nn.Linear(hidden_units, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor, lengths: torch.Tensor):
        # x shape: [batch, max_word_length, embedding_length]
        emb = self.drop(x)
        packed = pack_padded_sequence(
            emb, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, hidden = self.rnn(packed)
        output = self.linear(hidden[-1])
        output = self.sigmoid(output)

        return output

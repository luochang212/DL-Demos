import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

EMBEDDING_LENGTH = 128
OUTPUT_LENGTH = 10


class AttentionModel(nn.Module):
    def __init__(
        self, embeding_dim=32, encoder_dim=32, decoder_dim=32, dropout_rate=0.5
    ):
        super().__init__()
        self.drop = nn.Dropout(dropout_rate)
        self.embedding = nn.Embedding(EMBEDDING_LENGTH, embeding_dim)
        self.attention_linear = nn.Linear(2 * encoder_dim + decoder_dim, 1)
        self.softmax = nn.Softmax(-1)
        self.encoder = nn.LSTM(
            embeding_dim, encoder_dim, 1, batch_first=True, bidirectional=True
        )
        self.decoder = nn.LSTM(
            EMBEDDING_LENGTH + 2 * encoder_dim, decoder_dim, 1, batch_first=True
        )
        self.output_linear = nn.Linear(decoder_dim, EMBEDDING_LENGTH)
        self.decoder_dim = decoder_dim

    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor,
        n_output: int = OUTPUT_LENGTH,
    ):
        # x: [batch, n_sequence]
        batch, n_squence = x.shape[0:2]

        # x: [batch, n_sequence, embeding_dim]
        x = self.drop(self.embedding(x))

        # a: [batch, n_sequence, hidden]
        packed = pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_a, _ = self.encoder(packed)
        a, _ = pad_packed_sequence(packed_a, batch_first=True, total_length=n_squence)
        padding_mask = torch.arange(n_squence, device=x.device).unsqueeze(
            0
        ) >= lengths.to(x.device).unsqueeze(1)

        # prev_s: [batch, n_squence=1, hidden]
        # prev_y: [batch, n_squence=1, EMBEDDING_LENGTH]
        # y: [batch, n_output, EMBEDDING_LENGTH]
        prev_s = x.new_zeros(batch, 1, self.decoder_dim)
        prev_y = x.new_zeros(batch, 1, EMBEDDING_LENGTH)
        y = x.new_empty(batch, n_output, EMBEDDING_LENGTH)
        tmp_states = None
        for i_output in range(n_output):
            # repeat_s: [batch, n_squence, hidden]
            repeat_s = prev_s.repeat(1, n_squence, 1)
            # attention_input: [batch * n_sequence, hidden_s + hidden_a]
            attention_input = torch.cat((repeat_s, a), 2).reshape(batch * n_squence, -1)
            # x: [batch * n_sequence, 1]
            scores = self.attention_linear(attention_input)
            # scores: [batch, n_sequence]
            scores = scores.reshape(batch, n_squence)
            scores = scores.masked_fill(padding_mask, float('-inf'))
            alpha = self.softmax(scores)
            c = torch.sum(a * alpha.reshape(batch, n_squence, 1), 1)
            c = c.unsqueeze(1)
            decoder_input = torch.cat((prev_y, c), 2)

            if tmp_states is None:
                prev_s, tmp_states = self.decoder(decoder_input)
            else:
                prev_s, tmp_states = self.decoder(decoder_input, tmp_states)

            prev_y = self.output_linear(prev_s)
            y[:, i_output] = prev_y.squeeze(1)
        return y

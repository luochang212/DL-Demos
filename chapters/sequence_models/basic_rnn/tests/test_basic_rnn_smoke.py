import torch
import torch.nn.functional as F

from chapters.sequence_models.basic_rnn.constant import EMBEDDING_LENGTH
from chapters.sequence_models.basic_rnn.models import RNN1, RNN2


def test_rnn1_forward_loss_backward_on_cpu():
    model = RNN1(hidden_units=8)
    # one-hot input: [batch, max_word_length, embedding_length]
    word = torch.rand(2, 5, EMBEDDING_LENGTH)

    out = model(word)
    assert out.shape == (2, 5, EMBEDDING_LENGTH)

    target = torch.rand(2, 5, EMBEDDING_LENGTH)
    loss = ((out - target) ** 2).mean()
    loss.backward()
    assert torch.isfinite(loss)


def test_rnn2_forward_loss_backward_on_cpu():
    torch.manual_seed(0)
    model = RNN2(hidden_units=8, embeding_dim=8)
    # token-index input: [batch, max_word_length]
    word = torch.randint(0, EMBEDDING_LENGTH, (2, 5))

    out = model(word)
    assert out.shape == (2, 5, EMBEDDING_LENGTH)

    target = torch.randint(0, EMBEDDING_LENGTH, (2, 5))
    loss = F.cross_entropy(out.reshape(-1, EMBEDDING_LENGTH), target.reshape(-1))
    loss.backward()
    assert torch.isfinite(loss)

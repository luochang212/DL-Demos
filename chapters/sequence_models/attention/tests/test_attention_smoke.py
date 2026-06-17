import torch
import torch.nn.functional as F

from chapters.sequence_models.attention.main import (
    EMBEDDING_LENGTH,
    AttentionModel,
)


def test_attention_model_forward_loss_backward_on_cpu():
    torch.manual_seed(0)
    model = AttentionModel(embeding_dim=8, encoder_dim=8, decoder_dim=8)

    batch, seq_len, n_output = 2, 6, 4
    x = torch.randint(0, EMBEDDING_LENGTH, (batch, seq_len))
    lengths = torch.tensor([seq_len, seq_len - 1])

    y = model(x, lengths, n_output=n_output)
    assert y.shape == (batch, n_output, EMBEDDING_LENGTH)

    target = torch.randint(0, EMBEDDING_LENGTH, (batch, n_output))
    loss = F.cross_entropy(y.reshape(-1, EMBEDDING_LENGTH), target.reshape(-1))
    loss.backward()
    assert torch.isfinite(loss)

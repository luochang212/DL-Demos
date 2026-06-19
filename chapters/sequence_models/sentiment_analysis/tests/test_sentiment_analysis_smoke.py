import sys
from unittest.mock import MagicMock

import torch


def test_rnn_forward_loss_backward_on_cpu():
    """Smoke test: RNN model forward, loss, backward with synthetic embeddings.

    sentiment_analysis requires torchtext for GloVe embeddings, which is not in
    the default dependency set.  The smoke test mocks torchtext so it can import
    and exercise the canonical RNN model without downloading GloVe.
    """
    # Prevent the real torchtext from being imported (it may not be installed).
    mock_torchtext = MagicMock()
    sys.modules['torchtext'] = mock_torchtext
    sys.modules['torchtext.data'] = MagicMock()
    sys.modules['torchtext.vocab'] = MagicMock()
    sys.modules['torchtext.data.utils'] = MagicMock()

    from chapters.sequence_models.sentiment_analysis.code.model import GLOVE_DIM, RNN

    torch.manual_seed(0)
    model = RNN(hidden_units=16, dropout_rate=0.0)
    model.eval()

    # Synthetic batch: 2 sentences, max 5 tokens, embedding_dim = GLOVE_DIM.
    batch_size, max_len = 2, 5
    x = torch.randn(batch_size, max_len, GLOVE_DIM)
    lengths = torch.tensor([4, 3])

    with torch.no_grad():
        out = model(x, lengths)
    assert out.shape == (batch_size, 1)

    # Loss + backward in train mode.
    model.train()
    out = model(x, lengths).squeeze(-1)
    target = torch.tensor([1.0, 0.0])
    loss = torch.nn.BCELoss()(out, target)
    loss.backward()
    assert torch.isfinite(loss)

    # Verify all parameters received gradients.
    for name, param in model.named_parameters():
        assert param.grad is not None, f'{name} has no gradient'


def test_rnn_output_range_on_cpu():
    """Output should be in [0, 1] with sigmoid."""
    # Reuse the same mock setup.
    for mod in [
        'torchtext',
        'torchtext.data',
        'torchtext.vocab',
        'torchtext.data.utils',
    ]:
        if mod not in sys.modules:
            sys.modules[mod] = MagicMock()

    from chapters.sequence_models.sentiment_analysis.code.model import GLOVE_DIM, RNN

    torch.manual_seed(1)
    model = RNN(hidden_units=16, dropout_rate=0.0)
    model.eval()

    x = torch.randn(3, 6, GLOVE_DIM)
    lengths = torch.tensor([6, 4, 2])

    with torch.no_grad():
        out = model(x, lengths)
    assert (out >= 0).all() and (out <= 1).all()

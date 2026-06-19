import torch

from chapters.sequence_models.transformer.code.model import (
    MultiHeadAttention,
    PositionalEncoding,
    Transformer,
    attention,
)


def test_scaled_dot_product_attention_on_cpu():
    q = torch.rand(2, 4, 3, 8)  # [n, heads, q_len, d_k]
    k = torch.rand(2, 4, 3, 8)
    v = torch.rand(2, 4, 3, 8)
    out = attention(q, k, v)
    assert out.shape == (2, 4, 3, 8)


def test_multi_head_attention_on_cpu():
    mha = MultiHeadAttention(heads=2, d_model=16)
    x = torch.rand(2, 5, 16)
    out = mha(x, x, x)
    assert out.shape == (2, 5, 16)


def test_positional_encoding_on_cpu():
    pe = PositionalEncoding(d_model=16, max_seq_len=10)
    x = torch.rand(2, 5, 16)
    assert pe(x).shape == (2, 5, 16)


def test_transformer_forward_loss_backward_on_cpu():
    torch.manual_seed(0)
    model = Transformer(
        src_vocab_size=20,
        dst_vocab_size=20,
        pad_idx=0,
        d_model=16,
        d_ff=32,
        n_layers=2,
        heads=2,
    )
    src = torch.randint(1, 20, (2, 5))
    dst = torch.randint(1, 20, (2, 5))

    logits = model(src, dst)
    assert logits.shape == (2, 5, 20)

    target = torch.randint(0, 20, (2, 5))
    loss = torch.nn.functional.cross_entropy(logits.reshape(-1, 20), target.reshape(-1))
    loss.backward()
    assert torch.isfinite(loss)

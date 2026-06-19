import torch

from chapters.cv.style_transfer.code.style_transfer import (
    ContentLoss,
    Normalization,
    StyleLoss,
    gram,
)


def test_style_transfer_gram_and_losses_on_cpu():
    target = torch.rand(1, 3, 8, 8)
    x = torch.rand(1, 3, 8, 8, requires_grad=True)

    # gram of a (1, c, h, w) tensor -> (c, c)
    assert gram(x).shape == (3, 3)

    norm = Normalization([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    assert norm(x).shape == x.shape

    content_loss = ContentLoss(target)
    style_loss = StyleLoss(target)
    assert content_loss(x).shape == x.shape
    assert style_loss(x).shape == x.shape

    assert torch.isfinite(content_loss.loss)
    assert torch.isfinite(style_loss.loss)

    # losses are differentiable end to end (no VGG weights needed here)
    (content_loss.loss + style_loss.loss).backward()
    assert x.grad is not None

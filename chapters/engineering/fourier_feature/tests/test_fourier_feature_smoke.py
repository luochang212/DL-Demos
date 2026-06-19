import torch

from chapters.engineering.fourier_feature.code.model import MLP, FourierFeature


def test_mlp_forward_on_cpu():
    """MLP forward pass on a small coordinate grid."""
    torch.manual_seed(0)
    model = MLP(in_c=2, out_c=3, hidden_states=32)
    model.eval()

    # Synthetic coordinate grid: (1, 2, H, W)
    x = torch.rand(1, 2, 16, 16)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (1, 3, 16, 16)
    assert (out >= 0).all() and (out <= 1).all()  # sigmoid output


def test_fourier_feature_forward_on_cpu():
    """FourierFeature mapping on a coordinate grid."""
    torch.manual_seed(0)
    ff = FourierFeature(in_c=2, out_c=16, scale=10.0)
    ff.eval()

    x = torch.rand(1, 2, 8, 8)
    with torch.no_grad():
        out = ff(x)
    assert out.shape == (1, 16, 8, 8)


def test_mlp_loss_backward_on_cpu():
    """MLP + FourierFeature joint forward and backward pass."""
    torch.manual_seed(0)
    ff = FourierFeature(in_c=2, out_c=16, scale=10.0)
    model = MLP(in_c=16, out_c=3, hidden_states=32)

    x = torch.rand(1, 2, 8, 8)
    target = torch.rand(1, 3, 8, 8)

    features = ff(x)
    out = model(features)
    loss = torch.nn.L1Loss()(out, target)
    loss.backward()
    assert torch.isfinite(loss)

    # Verify gradients flow through both modules.
    for name, param in ff.named_parameters():
        assert param.grad is not None, f'FourierFeature.{name} has no gradient'
    for name, param in model.named_parameters():
        assert param.grad is not None, f'MLP.{name} has no gradient'

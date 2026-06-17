import torch

from chapters.generative_models.vae.code.main import loss_fn
from chapters.generative_models.vae.code.model import VAE


def test_vae_forward_loss_backward_and_sample_on_cpu():
    model = VAE(hiddens=[2, 4], latent_dim=3)
    x = torch.rand(1, 3, 64, 64)

    output, mean, logvar = model(x)
    loss = loss_fn(x, output, mean, logvar)
    loss.backward()
    sample = model.sample()

    assert output.shape == x.shape
    assert sample.shape == x.shape
    assert torch.isfinite(loss)

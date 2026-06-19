import torch
import torch.nn.functional as F

from chapters.generative_models.vqvae.code.main import reconstruct, sample_imgs, smoke
from chapters.generative_models.vqvae.code.model import VQVAE
from chapters.generative_models.vqvae.code.pixelcnn_model import PixelCNNWithEmbedding


def test_vqvae_forward_backward_reconstruct_and_sample_paths(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    device = torch.device('cpu')
    vqvae = VQVAE(input_dim=1, dim=4, n_embedding=8)
    gen_model = PixelCNNWithEmbedding(1, 4, 4, color_level=8)
    x = torch.rand(1, 1, 28, 28)

    x_hat, ze, zq = vqvae(x)
    loss = F.mse_loss(x_hat, x) + F.mse_loss(ze, zq.detach())
    loss.backward()

    reconstruct(vqvae, x, device)
    sample_imgs(vqvae, gen_model, (1, 28, 28), n_sample=1, device=device)

    assert x_hat.shape == x.shape
    assert (tmp_path / 'work_dirs/vqvae/reconstruct_MNIST.jpg').exists()
    assert (tmp_path / 'work_dirs/vqvae/sample_MNIST.jpg').exists()


def test_vqvae_cli_smoke_path_writes_outputs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    smoke(torch.device('cpu'))

    assert (tmp_path / 'work_dirs/vqvae/smoke_model.pth').exists()
    assert (tmp_path / 'work_dirs/vqvae/smoke_gen_model.pth').exists()
    assert (tmp_path / 'work_dirs/vqvae/reconstruct_MNIST.jpg').exists()
    assert (tmp_path / 'work_dirs/vqvae/sample_MNIST.jpg').exists()

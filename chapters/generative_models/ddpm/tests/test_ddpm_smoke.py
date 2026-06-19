import torch
import torch.nn as nn

from chapters.generative_models.ddpm.code.ddpm_simple import DDPM
from chapters.generative_models.ddpm.code.main import sample_imgs, smoke


class ZeroNoisePredictor(nn.Module):
    def forward(self, x, t):
        return torch.zeros_like(x)


def test_ddpm_forward_backward_and_output_path(tmp_path):
    device = torch.device('cpu')
    ddpm = DDPM(device, n_steps=4)
    x = torch.zeros(1, 1, 28, 28)
    t = torch.tensor([1])

    x_t = ddpm.sample_forward(x, t, eps=torch.ones_like(x))
    sample = ddpm.sample_backward(x.shape, ZeroNoisePredictor(), device)
    output_path = tmp_path / 'test_sample.jpg'
    sample_imgs(ddpm, ZeroNoisePredictor(), str(output_path), n_sample=1, device=device)

    assert x_t.shape == x.shape
    assert sample.shape == x.shape
    assert output_path.exists()


def test_ddpm_cli_smoke_path_writes_outputs(tmp_path):
    checkpoint = tmp_path / 'smoke_model.pth'
    output = tmp_path / 'smoke_sample.jpg'

    smoke(torch.device('cpu'), str(checkpoint), str(output))

    assert checkpoint.exists()
    assert output.exists()

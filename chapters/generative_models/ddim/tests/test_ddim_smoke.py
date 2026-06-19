import torch
import torch.nn as nn

from chapters.generative_models.ddim.code.ddim import DDIM
from chapters.generative_models.ddim.code.main import sample_imgs, smoke


class ZeroNoisePredictor(nn.Module):
    def forward(self, x, t):
        return torch.zeros_like(x)


def test_ddim_sampling_and_output_path(tmp_path):
    device = torch.device('cpu')
    ddim = DDIM(device, n_steps=4)
    x = torch.zeros(1, 1, 8, 8)

    sample = ddim.sample_backward(
        x, ZeroNoisePredictor(), device, simple_var=False, ddim_step=2, eta=0.5
    )
    output_path = tmp_path / 'test_sample.jpg'
    sample_imgs(
        ddim,
        ZeroNoisePredictor(),
        str(output_path),
        (1, 8, 8),
        n_sample=1,
        device=device,
        simple_var=False,
        ddim_step=2,
        eta=0.5,
    )

    assert sample.shape == x.shape
    assert output_path.exists()


def test_ddim_smoke_runner(tmp_path):
    checkpoint = tmp_path / 'smoke_model.pth'
    output = tmp_path / 'smoke_sample.jpg'

    smoke(device='cpu', checkpoint=str(checkpoint), output=str(output))

    assert checkpoint.exists()
    assert output.exists()

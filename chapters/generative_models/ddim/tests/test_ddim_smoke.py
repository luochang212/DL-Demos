from pathlib import Path

import torch
import torch.nn as nn

from chapters.generative_models.ddim.ddim import DDIM
from chapters.generative_models.ddim.main import sample_imgs


class ZeroNoisePredictor(nn.Module):
    def forward(self, x, t):
        return torch.zeros_like(x)


def test_ddim_sampling_and_output_path():
    device = torch.device('cpu')
    ddim = DDIM(device, n_steps=4)
    x = torch.zeros(1, 1, 8, 8)

    sample = ddim.sample_backward(
        x, ZeroNoisePredictor(), device, simple_var=False, ddim_step=2, eta=0.5
    )
    output_path = Path('work_dirs/ddim/test_sample.jpg')
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

import torch

from chapters.generative_models.pixelcnn.code.main import sample, smoke
from chapters.generative_models.pixelcnn.code.model import GatedPixelCNN


def test_pixelcnn_forward_backward_and_output_path(tmp_path):
    model = GatedPixelCNN(1, 2, 2, color_level=2)
    x = torch.rand(1, 1, 5, 5, requires_grad=True)
    output = model(x)
    loss = output.mean()
    loss.backward()

    model_path = tmp_path / 'test_model.pth'
    output_path = tmp_path / 'test_sample.jpg'
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_path)
    sample(model, torch.device('cpu'), str(model_path), str(output_path), n_sample=1)

    assert output.shape == (1, 2, 5, 5)
    assert output_path.exists()


def test_pixelcnn_cli_smoke_path_writes_outputs(tmp_path):
    checkpoint = tmp_path / 'smoke_model.pth'
    output = tmp_path / 'smoke_sample.jpg'

    smoke(torch.device('cpu'), str(checkpoint), str(output))

    assert checkpoint.exists()
    assert output.exists()

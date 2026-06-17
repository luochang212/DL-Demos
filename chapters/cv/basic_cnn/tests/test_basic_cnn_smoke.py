import torch

from chapters.cv.basic_cnn.pt_main import init_model


def test_basic_cnn_forward_loss_backward_on_cpu():
    model = init_model('cpu')

    x = torch.rand(2, 3, 224, 224)
    target = torch.randint(0, 2, (2, 1)).float()

    y_hat = model(x)
    loss = torch.nn.BCELoss()(y_hat, target)
    loss.backward()

    assert y_hat.shape == (2, 1)
    assert torch.isfinite(loss)

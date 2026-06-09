import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

IMG_SIZE = (256, 256)
DEFAULT_CONTENT_LAYERS = ['conv_4']
DEFAULT_STYLE_LAYERS = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']


def read_image(image_path, device, img_size=IMG_SIZE):
    pipeline = transforms.Compose([transforms.Resize(img_size), transforms.ToTensor()])
    img = Image.open(image_path).convert('RGB')
    return pipeline(img).unsqueeze(0).to(device)


def save_image(tensor, image_path):
    output_path = Path(image_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    img = transforms.ToPILImage()(tensor.detach().cpu().squeeze(0))
    img.save(output_path)


class ContentLoss(torch.nn.Module):
    def __init__(self, target: torch.Tensor):
        super().__init__()
        self.target = target.detach()
        self.loss = torch.tensor(0.0, device=target.device)

    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)
        return input


def gram(x: torch.Tensor):
    n, c, h, w = x.shape
    features = x.reshape(n * c, h * w)
    return torch.mm(features, features.T) / (n * c * h * w)


class StyleLoss(torch.nn.Module):
    def __init__(self, target: torch.Tensor):
        super().__init__()
        self.target = gram(target.detach()).detach()
        self.loss = torch.tensor(0.0, device=target.device)

    def forward(self, input):
        self.loss = F.mse_loss(gram(input), self.target)
        return input


class Normalization(torch.nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.register_buffer('mean', torch.tensor(mean).reshape(-1, 1, 1))
        self.register_buffer('std', torch.tensor(std).reshape(-1, 1, 1))

    def forward(self, img):
        return (img - self.mean) / self.std


def get_model_and_losses(
    content_img,
    style_img,
    content_layers=DEFAULT_CONTENT_LAYERS,
    style_layers=DEFAULT_STYLE_LAYERS,
    cnn=None,
):
    content_losses = []
    style_losses = []
    model = torch.nn.Sequential(
        Normalization([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ).to(content_img.device)
    if cnn is None:
        weights = models.VGG19_Weights.DEFAULT
        cnn = models.vgg19(weights=weights).features
    cnn = cnn.to(content_img.device).eval()

    conv_index = 0
    for layer in cnn.children():
        if isinstance(layer, torch.nn.Conv2d):
            conv_index += 1
            name = f'conv_{conv_index}'
        elif isinstance(layer, torch.nn.ReLU):
            name = f'relu_{conv_index}'
            layer = torch.nn.ReLU(inplace=False)
        elif isinstance(layer, torch.nn.MaxPool2d):
            name = f'pool_{conv_index}'
        elif isinstance(layer, torch.nn.BatchNorm2d):
            name = f'bn_{conv_index}'
        else:
            raise RuntimeError(f'Unrecognized layer: {layer.__class__.__name__}')

        model.add_module(name, layer)
        if name in content_layers:
            content_loss = ContentLoss(model(content_img))
            model.add_module(f'content_loss_{conv_index}', content_loss)
            content_losses.append(content_loss)
        if name in style_layers:
            style_loss = StyleLoss(model(style_img))
            model.add_module(f'style_loss_{conv_index}', style_loss)
            style_losses.append(style_loss)
        if len(content_losses) + len(style_losses) == len(content_layers) + len(
            style_layers
        ):
            break

    model.requires_grad_(False)
    return model, content_losses, style_losses


def run_style_transfer(
    content_path,
    style_path,
    output_path,
    num_steps=300,
    device=None,
    style_weight=1e4,
    content_weight=1,
):
    device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    content_img = read_image(content_path, device)
    style_img = read_image(style_path, device)
    input_img = content_img.clone().requires_grad_(True)
    model, content_losses, style_losses = get_model_and_losses(content_img, style_img)
    optimizer = optim.LBFGS([input_img])
    step = 0

    while step < num_steps:

        def closure():
            nonlocal step
            with torch.no_grad():
                input_img.clamp_(0, 1)
            optimizer.zero_grad()
            model(input_img)
            content_loss = sum(loss.loss for loss in content_losses)
            style_loss = sum(loss.loss for loss in style_losses)
            total_loss = content_weight * content_loss + style_weight * style_loss
            total_loss.backward()
            step += 1
            return total_loss

        optimizer.step(closure)

    with torch.no_grad():
        input_img.clamp_(0, 1)
    save_image(input_img, output_path)
    return input_img.detach()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--content', default='dldemos/StyleTransfer/dancing.jpg')
    parser.add_argument('--style', default='dldemos/StyleTransfer/picasso.jpg')
    parser.add_argument('--output', default='work_dirs/style-transfer.jpg')
    parser.add_argument('--steps', type=int, default=300)
    parser.add_argument('--device')
    args = parser.parse_args()
    run_style_transfer(
        args.content, args.style, args.output, args.steps, device=args.device
    )


if __name__ == '__main__':
    main()

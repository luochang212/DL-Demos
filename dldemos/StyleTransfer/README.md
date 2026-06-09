Run neural style transfer from the repository root:

```shell
uv run python dldemos/StyleTransfer/style_transfer.py \
  --content dldemos/StyleTransfer/dancing.jpg \
  --style dldemos/StyleTransfer/picasso.jpg \
  --output work_dirs/style-transfer.jpg
```

The first run downloads torchvision's default VGG19 weights.

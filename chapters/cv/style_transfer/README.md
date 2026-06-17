Run neural style transfer from the repository root:

```shell
uv run python chapters/cv/style_transfer/style_transfer.py \
  --content chapters/cv/style_transfer/dancing.jpg \
  --style chapters/cv/style_transfer/picasso.jpg \
  --output work_dirs/style-transfer.jpg
```

The first run downloads torchvision's default VGG19 weights.

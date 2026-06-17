# VAE

This directory is a compatibility entrypoint for older commands and imports.
The tutorial source now lives in `chapters/generative_models/vae/`.

1. Download [CelebA](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)
   Align&Cropped Images.

2. Modify the path of function `get_dataloader` in `main.py`.

3. Choose the operation explicitly:

```bash
uv run python -m chapters.generative_models.vae.code.main --mode train
uv run python -m chapters.generative_models.vae.code.main --mode reconstruct
uv run python -m chapters.generative_models.vae.code.main --mode generate
```

The default device is selected automatically. Use `--device cpu` or
`--device cuda:0` to choose one explicitly, and `--checkpoint PATH` to override
the default checkpoint. Training and reconstruction require CelebA; generation
only requires a checkpoint.

Acknowledgement: The code is inspired by
[PyTorch-VAE](https://github.com/AntixK/PyTorch-VAE).

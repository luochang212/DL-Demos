# VAE

1. Download [CelebA](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)
   Align&Cropped Images.

2. Modify the path of function `get_dataloader` in `main.py`.

3. Choose the operation explicitly:

```bash
uv run python dldemos/VAE/main.py --mode train
uv run python dldemos/VAE/main.py --mode reconstruct
uv run python dldemos/VAE/main.py --mode generate
```

The default device is selected automatically. Use `--device cpu` or
`--device cuda:0` to choose one explicitly, and `--checkpoint PATH` to override
the default checkpoint. Training and reconstruction require CelebA; generation
only requires a checkpoint.

Acknowledgement: The code is inspired by
[PyTorch-VAE](https://github.com/AntixK/PyTorch-VAE).

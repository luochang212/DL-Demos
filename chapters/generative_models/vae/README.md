# VAE Chapter

This directory is the source of truth for the VAE tutorial chapter.

## Layout

- `code/`: runnable PyTorch implementation used by the web tutorial.
- `experiments/`: small reproducible experiment configurations.
- `results/`: lightweight result summaries that can be committed.
- `assets/`: images and figures used by the tutorial.
- `lean/`: future Lean4 formula verification notes and proofs.
- `tests/`: chapter-local smoke tests.

## Run

```powershell
uv run python -m chapters.generative_models.vae.code.main --mode train --device cuda:0
uv run python -m chapters.generative_models.vae.code.main --mode reconstruct --device cuda:0
uv run python -m chapters.generative_models.vae.code.main --mode generate --device cuda:0
```

The default CelebA image directory is `data/celebA/img_align_celeba`.
Checkpoints and generated images are written under `work_dirs/vae/`.

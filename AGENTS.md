# AGENTS.md

This repository is a tutorial-first deep learning project. The web tutorial,
runnable code, experiments, results, and future Lean4 verification must stay
aligned by chapter.

## Repository Rules

- `website/` is the Docusaurus site. Do not put Python tutorial code there.
- `chapters/` is the source of truth for tutorial chapters.
- `dldemos/` is legacy compatibility only. Do not add new tutorial code there.
- `data/`, `work_dirs/`, checkpoints, full datasets, and generated bulk outputs
  must not be committed.
- Use ASCII in code unless the file is tutorial text or already intentionally
  Chinese/Unicode.

## Generative Model Migration Policy

Migrate all generative model chapters aggressively into `chapters/`.

Target chapters:

- `dldemos/VAE` -> `chapters/generative_models/vae`
- `dldemos/ddpm` -> `chapters/generative_models/ddpm`
- `dldemos/ddim` -> `chapters/generative_models/ddim`
- `dldemos/pixelcnn` -> `chapters/generative_models/pixelcnn`
- `dldemos/VQVAE` -> `chapters/generative_models/vqvae`

Required chapter layout:

```text
chapters/generative_models/<chapter>/
  README.md
  code/
  experiments/
  results/
  assets/
  lean/
  tests/
```

Migration rules:

- Move the real implementation into `chapters/generative_models/<chapter>/code/`.
- Update imports inside migrated code so chapter code does not import from its
  old `dldemos/<chapter>` package.
- Old `dldemos/<chapter>` modules may temporarily remain as thin wrappers only.
  They must not contain the canonical implementation.
- Default outputs must go under `work_dirs/<chapter>/`.
- Default checkpoints must go under `work_dirs/<chapter>/`.
- Tutorial docs must link to `chapters/generative_models/<chapter>/...`, not
  `dldemos/<chapter>/...`.
- Do not preserve old paths just to avoid breakage. Breakage is acceptable if
  tests or real runs expose what must be fixed.

## Code Reuse Rules

Do not create broad abstractions before at least two migrated chapters need the
same behavior.

Allowed shared utilities once duplication is real:

- device selection
- seed setup
- parent directory creation
- checkpoint load/save helpers
- image grid writing
- project/work directory path helpers

Do not create generic trainers, model base classes, or large config frameworks
unless a migrated chapter already proves the need.

## Validation SOP

Every generative model migration must pass three levels of validation.

### 1. Static Quality

Run the narrow check first:

```powershell
uv run ruff check chapters/generative_models/<chapter> dldemos/<LegacyChapter>
uv run ruff format --check chapters/generative_models/<chapter> dldemos/<LegacyChapter>
```

Before final commit, run the broader check:

```powershell
uv run ruff check chapters dldemos tests
uv run ruff format --check chapters dldemos tests
```

Only run `ruff format` on files intentionally touched by the migration.

### 2. Unit and Smoke Tests

Each chapter must have a CPU smoke test under:

```text
chapters/generative_models/<chapter>/tests/
```

The smoke test must:

- import the canonical chapter code from `chapters...`
- instantiate the core model
- run a forward pass on synthetic or tiny input
- compute the relevant loss when the chapter has a loss function
- run backward when practical
- run a sample/generate step when practical
- avoid requiring real datasets or pretrained checkpoints

Run:

```powershell
uv run pytest chapters/generative_models/<chapter>/tests -q
uv run pytest tests -q
```

### 3. Real Algorithm Run

Each migrated chapter must also define at least one runnable smoke command in
its `README.md` and, when useful, in `experiments/smoke.yaml`.

The real run must verify paths and side effects, not just imports. It must:

- use the migrated `python -m chapters...` entrypoint
- write outputs under `work_dirs/<chapter>/`
- create checkpoint/output parent directories automatically
- complete on CPU or one local GPU within a short debug budget
- avoid downloading large datasets unless the chapter explicitly documents it

Preferred commands:

```powershell
uv run python -m chapters.generative_models.<chapter>.code.main --help
uv run python -m chapters.generative_models.<chapter>.code.main <smoke args>
```

If the original algorithm cannot run without a real dataset or long training,
add a chapter-local debug mode or smoke script that uses synthetic or tiny data
while exercising the same model, loss, checkpoint, and output paths.

## Website Validation

After changing docs or paths:

```powershell
npm run build
```

Run this from `website/`.

Docs must not reference old `dldemos/<chapter>` source paths after a chapter is
migrated.

## Dependency Version Control

- Python dependencies are controlled by `pyproject.toml` and `uv.lock`.
- Website dependencies are controlled by `website/package.json` and
  `website/package-lock.json`.
- CI must use `uv sync --frozen` and `npm ci`.
- Do not add a dependency for one chapter unless that chapter needs it to run.
- Prefer dependency groups for heavy or optional chapter-specific dependencies.
- Do not run broad dependency upgrades during a migration unless the migration
  requires them.

## Commit Discipline

Use small commits:

1. Move canonical chapter code.
2. Update wrappers and imports.
3. Add chapter tests and smoke config.
4. Update website docs.

Do not mix unrelated cleanup with a chapter migration.

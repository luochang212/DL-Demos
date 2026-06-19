# AGENTS.md

This repository is a tutorial-first deep learning project. The web tutorial,
runnable code, formula derivations, and validation commands must stay aligned by
chapter.

## Project Origin

- This repository is maintained at `https://github.com/luochang212/DL-Demos`.
- It was forked from `https://github.com/SingleZombie/DL-Demos`.
- The original repository provides the inherited demo code foundation.
- This fork turns the inherited demos into a chapter-driven web tutorial with
  clearer derivations, runnable experiments, and visual results.
- When changing inherited code, preserve attribution and avoid implying the
  original author wrote new tutorial text or refactored chapter structure.

## Reference Policy

- Every tutorial chapter that explains a named method or model family must cite
  the paper that originally introduced that method in its reference section.
  Survey papers, implementation notes, and blog posts are useful additions, but
  they do not replace the original method paper.
- When a chapter teaches a later variant, cite both the variant paper and the
  foundational paper if the text relies on the foundation. For example, DDPM
  chapters should cite both Ho et al. and the earlier nonequilibrium
  thermodynamics diffusion paper when discussing the origin of diffusion
  probabilistic models.
- Zhou Yifan's blog at `https://zhouyifan.net/archives/` is an approved
  reference source for improving the generative model tutorials.
- Relevant examples include VAE, KL divergence, diffusion models, DDIM, and
  VQ-VAE articles.
- If tutorial text, explanation structure, intuition, or terminology is informed
  by a Zhou Yifan blog article, cite the exact article in the chapter's
  reference section.
- Do not copy long passages. Use the blog as a reference, then write original
  tutorial text tailored to this repository.
- Keep references specific. Prefer article-level links over only linking the
  archive page.

## Formula Derivation Policy

- Each tutorial chapter must keep the website page readable and place complete
  symbolic derivations under
  `chapters/generative_models/<chapter>/derivations/formulas.md`.
- Each chapter derivation directory must include a small Lean4 file that checks
  the key algebraic identities used by the derivation. Lean checks are for
  code-facing algebra, not for fully formalizing probability theory,
  neural-network training, or empirical modeling choices.
- Website chapters should link to `formulas.md` instead of embedding long proof
  details in the tutorial body.
- Run `lake build` after changing Lean derivation files.

## Repository Rules

- `website/` is the Docusaurus site. Do not put Python tutorial code there.
- `chapters/` is the source of truth for tutorial chapters.
- `dldemos/` has been removed. All tutorial code now lives under `chapters/`.
- `data/README.md` documents local dataset conventions and may be committed.
  Real files under `data/`, `work_dirs/`, checkpoints, full datasets, and
  generated bulk outputs must not be committed.
- Use ASCII in code unless the file is tutorial text or already intentionally
  Chinese/Unicode.

## Data Policy

- Keep one top-level `data/README.md` for repository-wide dataset conventions.
- Each chapter that requires real data must document the data source, license or
  usage constraints when relevant, expected local directory layout, file format,
  and a minimal example.
- Each chapter that requires checkpoints or pretrained models must document the
  source, expected local path, size or disk-risk when relevant, and whether the
  file is required for training, reconstruction, generation, or evaluation.
- Chapter code should default to paths under `data/<dataset>/...` for inputs and
  `work_dirs/<chapter>/...` for outputs.
- Real validation must not depend only on full datasets. Provide a smoke command
  or test path that uses synthetic or tiny local data while exercising model,
  loss, checkpoint, and output paths.

## Generative Model Chapter Layout

All generative model chapters live under `chapters/generative_models/`:

- `chapters/generative_models/vae`
- `chapters/generative_models/ddpm`
- `chapters/generative_models/ddim`
- `chapters/generative_models/pixelcnn`
- `chapters/generative_models/vqvae`

Required chapter layout:

```text
chapters/generative_models/<chapter>/
  README.md
  code/
  derivations/
  tests/
```

Optional directories are allowed only when they contain real chapter material:

- `assets/`: tutorial figures or diagrams referenced by docs.
- `experiments/`: runnable experiment configs consumed by code or docs.
- `results/`: small committed result summaries referenced by docs.

Do not keep placeholder README files or smoke YAML files that are not consumed by
an actual command.

Chapter README files must include the runnable smoke command, any real-data
commands, data/checkpoint/pretrained-model requirements, and the narrow
validation commands for that chapter.

Migration rules:

- Move the real implementation into `chapters/generative_models/<chapter>/code/`.
- Update imports so chapter code is self-contained and does not import across
  chapter packages.
- Default outputs must go under `work_dirs/<chapter>/`.
- Default checkpoints must go under `work_dirs/<chapter>/`.
- Tutorial docs must link to paths under `chapters/generative_models/<chapter>/...`.
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
uv run ruff check chapters/generative_models/<chapter>
uv run ruff format --check chapters/generative_models/<chapter>
```

Before final commit, run the broader check:

```powershell
uv run ruff check chapters
uv run ruff format --check chapters
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
uv run pytest chapters -q
```

### 3. Real Algorithm Run

Each migrated chapter must also define at least one runnable smoke command in
its `README.md`.

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
add a chapter-local smoke mode or script that uses synthetic or tiny data while
exercising the same model, loss, checkpoint, and output paths.

## Website Validation

After changing docs or paths:

```powershell
npm run build
```

Run this from `website/`.

Docs must reference paths under `chapters/`; do not link to removed legacy paths.

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
2. Update imports and cross-references.
3. Add chapter tests and real smoke command.
4. Update website docs.

Do not mix unrelated cleanup with a chapter migration.

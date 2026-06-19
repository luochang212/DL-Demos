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
  reference source for improving tutorial chapters across all categories
  (generative models, fundamentals, CV, sequence models, etc.).
- When adding blog references, verify the article exists and is relevant;
  blog references supplement original method papers, not replace them.
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

## Chapter Improvement Pipeline

When improving a tutorial chapter, follow this pipeline in order. Each step
references the detailed policy or SOP that governs it. The generative model
chapters (`chapters/generative_models/`) are the completed reference
implementation of this pipeline.

### 1. Research & Tutorial

- **MUST** check [Zhou Yifan's blog](https://zhouyifan.net/archives/) for relevant
  articles on the topic.  If the blog cannot be accessed (HTTP error, timeout, or
  DNS failure), **stop and report the error** — do not proceed to later steps.
  Use `curl` as a fallback when `WebFetch` is blocked; the site is hosted on
  GitHub Pages.
- Read the relevant blog article(s) **before** writing or revising the tutorial
  text.  Extract teaching structure, intuitive explanations, visual metaphors,
  and step-by-step derivations.  The tutorial must genuinely reflect blog
  insights, not merely cite the article as an afterthought.
- Update the chapter's website doc (`website/docs/.../<chapter>.mdx`) with
  clearer derivations, better intuition, and improved terminology informed by
  the blog.
- Cite the original method paper and any referenced blog articles (see
  [Reference Policy](#reference-policy)).  Blog citations must include the
  exact article title and URL.  Place them prominently in the "参考资料"
  section, not buried in footnotes.

### 2. Complete Formulas

- Write full symbolic derivations in
  `chapters/.../<chapter>/derivations/formulas.md`.
- Add a Lean4 file (`derivations/<chapter>.lean`) that checks the key algebraic
  identities used in the derivations (see
  [Formula Derivation Policy](#formula-derivation-policy)).
- The website doc should link to `formulas.md` instead of embedding long proofs
  in the tutorial body.

### 3. Complete Code

- Place the canonical implementation under `chapters/.../<chapter>/code/`.
- Keep imports self-contained — do not import across chapter packages.
- Provide a CLI entrypoint (`main.py`) that supports training, evaluation, and a
  smoke mode that uses synthetic or tiny data.
- Default outputs and checkpoints must go under `work_dirs/<chapter>/`.
- Follow [Code Reuse Rules](#code-reuse-rules) when extracting shared utilities.

### 4. Data & Checkpoint Documentation

- Document the data source, license or usage constraints, expected local
  directory layout, file format, and a minimal example in the chapter README
  (see [Data Policy](#data-policy)).
- If the chapter requires checkpoints or pretrained models, document the source,
  expected local path, size, and whether the file is required for training,
  reconstruction, generation, or evaluation.

### 5. Verify Code

- Write a CPU smoke test under `chapters/.../<chapter>/tests/`. It must
  instantiate the model, run a forward pass, compute loss, run backward, and
  perform a sample/generate step — all on synthetic or tiny input.
- Run the smoke test and the real algorithm entrypoint (see
  [Validation SOP](#validation-sop)).
- Run `ruff check` and `ruff format --check` on the changed paths.

### 6. Verify Formulas

- Run `lake build` to compile all Lean derivation files.
- Fix any type errors or unproven identities before considering the chapter
  complete.

### 7. Build & Commit

- Run `npm run build` from `website/` to verify the site builds with all doc
  changes.
- Commit in small batches following [Commit Discipline](#commit-discipline):
  research and doc → formulas and Lean → code → tests → website docs.

## Validation SOP

Every chapter improvement must pass three levels of validation.

### 1. Static Quality

Run the narrow check first:

```powershell
uv run ruff check chapters/<category>/<chapter>
uv run ruff format --check chapters/<category>/<chapter>
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
chapters/<category>/<chapter>/tests/
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
uv run pytest chapters/<category>/<chapter>/tests -q
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

## Non-Generative Chapter Improvement

The Chapter Improvement Pipeline applies to all chapters, but non-generative
chapters (fundamentals, cv, sequence_models, training_tricks, engineering) follow
a lighter variant.  Formula derivations and Lean verification are optional for
chapters whose primary teaching value is in code implementation rather than
probabilistic derivation.

### Adapted Pipeline (Non-Generative)

1. **Research & References** — ensure every website doc has a "参考资料" section
   citing the original method paper.  Fundamentals chapters cite foundational
   work (Cox 1958 for logistic regression, Rumelhart et al. 1986 for
   backpropagation, etc.).  Applied chapters cite both the original method and
   the code-framework paper where relevant.
2. **Code Canonicalization** — if a chapter only has Jupyter notebooks, extract
   the core model classes into `code/model.py`.  Keep imports self-contained.
3. **Smoke Test** — write a CPU smoke test that instantiates the model, runs
   forward pass, computes loss, and runs backward on synthetic data.  See
   [Smoke Test Patterns](#smoke-test-patterns) for handling optional
   dependencies.
4. **README** — every chapter (and each topic subdirectory) must have a README
   with the runnable smoke command, any real-data commands, and
   data/checkpoint requirements.
5. **Validate** — run smoke tests, `ruff check`, `ruff format --check`, and
   `npm run build` from `website/`.  Commit each chapter improvement separately.

### Smoke Test Patterns

When a chapter depends on libraries that are not in the default dependency set
(e.g. `tensorflow` for resnet, `torchtext` for sentiment_analysis), use one of
these patterns:

**Pattern A — skip on missing dependency:**

```python
pytest.importorskip('tensorflow', reason='TensorFlow not installed')
```

Use when the test can exercise the real model classes while the dependency is
not part of the standard `uv sync` install.

**Pattern B — mock the dependency:**

```python
import sys
from unittest.mock import MagicMock

sys.modules['torchtext'] = MagicMock()
sys.modules['torchtext.data'] = MagicMock()
sys.modules['torchtext.vocab'] = MagicMock()
```

Use when the model class itself does not use the dependency but a module-level
import would trigger a download or fail.  The mocks must be in `sys.modules`
*before* importing from the chapter.

**Pattern C — extract model to avoid the dependency:**

When the module-level code has a hard dependency on a heavy library, extract the
pure model class into `code/model.py` and write the smoke test against the
extracted module.  The Jupyter notebook or legacy script can import from `code/`
after extraction.

### Jupyter Notebook Chapters

If a chapter's canonical implementation lives only in `.ipynb` files:

1. Extract the core model classes into `code/model.py` as regular Python modules.
2. Write smoke tests that import from `code/model.py`.
3. The notebooks remain as reference experiments — they may import from `code/`
   or keep their inline definitions.
4. Add a `README.md` documenting both the smoke test command and the notebook
   launch command.

## Current Chapter State (2026-06-19)

All 22 topics across 6 chapter categories have been verified to pass smoke
tests, ruff checks, and website build.

### Completion Summary

| Category | Topics | Code | Smoke Test | README | References |
|---|---|---|---|---|---|
| generative_models | 5 | ✅ | ✅ | ✅ | ✅ + formulas |
| training_tricks | 3 | ✅ | ✅ | ✅ | ✅ |
| fundamentals | 4 | ✅ | ✅ | ✅ | ✅ |
| sequence_models | 4 | ✅ | ✅ | ✅ | ✅ |
| cv | 4 | ✅ | ✅ ⚠️ | ✅ | ✅ |
| engineering | 2 | ✅ | ✅ | ✅ | ✅ + formulas (fourier_feature) |

⚠️ resnet smoke test requires TensorFlow (skipped when unavailable).

### Future Work

- **Formula derivations**: Consider adding `formulas.md` and Lean verification
  for transformer (attention math), basic_cnn (convolution backprop), and
  advanced_optimizer (SGD/Momentum/Adam update rules).
- **Zhou Yifan blog references**: When blog access is restored, check for
  articles on attention, transformer, CNN basics, and optimization that could
  improve non-generative tutorial chapters.
- **resnet PyTorch migration**: The current resnet implementation is
  TensorFlow-only.  Consider adding a PyTorch version to match the rest of the
  project.
- **smoke test coverage**: sentiment_analysis and resnet smoke tests use
  dependency mocking/skipping.  When those dependencies are promoted to the
  default install, remove the mocks.

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

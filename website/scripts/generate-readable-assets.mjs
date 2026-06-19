import crypto from 'node:crypto';
import fs from 'node:fs';
import path from 'node:path';
import katex from 'katex';

const repoRoot = path.resolve(process.cwd(), '..');
const outputRoot = path.resolve(process.cwd(), 'static', 'readable');
const contentRoot = path.join(outputRoot, 'content');

const includeRoots = [
  'chapters',
];

const readablePatterns = [
  /\/code\/.*\.(py|md)$/u,
  /\/derivations\/.*\.(md|lean)$/u,
  // Chapter-level files that haven't been migrated to code/ subdirectory yet
  /\/chapters\/[^/]+\/[^/]+\/[^/]+\.(py|ipynb)$/u,
];

const ignoredSegments = new Set([
  '__pycache__',
  '.pytest_cache',
  '.ruff_cache',
  '.venv',
  'data',
  'work_dirs',
  'website',
]);

function toPosixPath(value) {
  return value.split(path.sep).join('/');
}

function escapeHtml(value) {
  return value
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&#39;');
}

function renderMath(tex, displayMode) {
  try {
    return katex.renderToString(tex, {
      displayMode,
      throwOnError: false,
      strict: false,
    });
  } catch {
    return displayMode
      ? `<pre class="readable-math-error">${escapeHtml(tex)}</pre>`
      : `<code>${escapeHtml(tex)}</code>`;
  }
}

function renderInline(value) {
  const tokens = [];
  let rest = value;

  rest = rest.replace(/`([^`]+)`/gu, (_, code) => {
    const token = `@@CODE_${tokens.length}@@`;
    tokens.push(`<code>${escapeHtml(code)}</code>`);
    return token;
  });

  rest = rest.replace(/\$([^$\n]+)\$/gu, (_, tex) => {
    const token = `@@MATH_${tokens.length}@@`;
    tokens.push(renderMath(tex, false));
    return token;
  });

  rest = escapeHtml(rest);
  for (const [index, html] of tokens.entries()) {
    rest = rest.replaceAll(`@@CODE_${index}@@`, html);
    rest = rest.replaceAll(`@@MATH_${index}@@`, html);
  }
  return rest;
}

function renderMarkdown(source) {
  const lines = source.replace(/\r\n/g, '\n').split('\n');
  const html = [];
  let paragraph = [];
  let listItems = [];
  let inFence = false;
  let fenceLang = '';
  let fenceLines = [];

  function flushParagraph() {
    if (paragraph.length === 0) {
      return;
    }
    html.push(`<p>${renderInline(paragraph.join(' '))}</p>`);
    paragraph = [];
  }

  function flushList() {
    if (listItems.length === 0) {
      return;
    }
    html.push(`<ul>${listItems.map((item) => `<li>${renderInline(item)}</li>`).join('')}</ul>`);
    listItems = [];
  }

  function flushFence() {
    const languageClass = fenceLang ? ` class="language-${escapeHtml(fenceLang)}"` : '';
    html.push(
      `<pre class="readable-code"><code${languageClass}>${escapeHtml(fenceLines.join('\n'))}</code></pre>`,
    );
    fenceLines = [];
    fenceLang = '';
  }

  for (let index = 0; index < lines.length; index += 1) {
    const line = lines[index];

    if (line.startsWith('```')) {
      if (inFence) {
        flushFence();
        inFence = false;
      } else {
        flushParagraph();
        flushList();
        inFence = true;
        fenceLang = line.slice(3).trim().split(/\s+/u)[0] ?? '';
      }
      continue;
    }

    if (inFence) {
      fenceLines.push(line);
      continue;
    }

    if (line.trim() === '$$') {
      flushParagraph();
      flushList();
      const mathLines = [];
      index += 1;
      while (index < lines.length && lines[index].trim() !== '$$') {
        mathLines.push(lines[index]);
        index += 1;
      }
      html.push(`<div class="readable-math">${renderMath(mathLines.join('\n'), true)}</div>`);
      continue;
    }

    if (line.trim() === '') {
      flushParagraph();
      flushList();
      continue;
    }

    const heading = /^(#{1,6})\s+(.+)$/u.exec(line);
    if (heading) {
      flushParagraph();
      flushList();
      const level = heading[1].length;
      html.push(`<h${level}>${renderInline(heading[2])}</h${level}>`);
      continue;
    }

    const listItem = /^-\s+(.+)$/u.exec(line);
    if (listItem) {
      flushParagraph();
      listItems.push(listItem[1]);
      continue;
    }

    paragraph.push(line.trim());
  }

  if (inFence) {
    flushFence();
  }
  flushParagraph();
  flushList();

  return `<div class="readable-markdown">${html.join('\n')}</div>`;
}

function renderCode(source, language) {
  const lines = source.replace(/\r\n/g, '\n').split('\n');
  const rendered = lines
    .map((line, index) => (
      `<span class="readable-code-line" data-line="${index + 1}">${escapeHtml(line) || ' '}</span>`
    ))
    .join('\n');
  return `<pre class="readable-code readable-code-with-lines"><code class="language-${language}">${rendered}</code></pre>`;
}

function hasFrontmatter(source) {
  return source.startsWith('---\n') && source.indexOf('\n---\n', 4) !== -1;
}

function stripFrontmatter(source) {
  if (!hasFrontmatter(source)) {
    return source;
  }
  const end = source.indexOf('\n---\n', 4);
  return source.slice(end + 5).trimStart();
}

function detectKind(repoPath) {
  if (repoPath.endsWith('.lean')) {
    return 'lean';
  }
  if (repoPath.includes('/derivations/')) {
    return 'formula';
  }
  if (repoPath.endsWith('.md')) {
    return 'markdown';
  }
  return 'code';
}

function detectLanguage(repoPath) {
  if (repoPath.endsWith('.py')) {
    return 'python';
  }
  if (repoPath.endsWith('.lean')) {
    return 'lean';
  }
  if (repoPath.endsWith('.md')) {
    return 'markdown';
  }
  return 'text';
}

function titleFor(repoPath) {
  if (repoPath.endsWith('/derivations/formulas.md')) {
    const chapter = repoPath.split('/').at(-3);
    return `${chapter?.toUpperCase() ?? 'Formula'} 完整公式推导`;
  }
  return repoPath.split('/').slice(-3).join('/');
}

function shouldInclude(repoPath) {
  return readablePatterns.some((pattern) => pattern.test(`/${repoPath}`));
}

function walk(relativeDir) {
  const absoluteDir = path.join(repoRoot, relativeDir);
  const entries = fs.readdirSync(absoluteDir, { withFileTypes: true });
  const files = [];

  for (const entry of entries) {
    if (ignoredSegments.has(entry.name)) {
      continue;
    }
    const nextRelative = path.join(relativeDir, entry.name);
    if (entry.isDirectory()) {
      files.push(...walk(nextRelative));
    } else if (entry.isFile()) {
      const repoPath = toPosixPath(nextRelative);
      if (shouldInclude(repoPath)) {
        files.push(repoPath);
      }
    }
  }

  return files;
}

function renderEntry(repoPath) {
  const source = fs.readFileSync(path.join(repoRoot, repoPath), 'utf8');
  const kind = detectKind(repoPath);
  const language = detectLanguage(repoPath);
  const id = crypto.createHash('sha1').update(repoPath).digest('hex').slice(0, 16);
  const contentDir = path.join(contentRoot, id);
  const body = kind === 'formula' || kind === 'markdown'
    ? renderMarkdown(stripFrontmatter(source))
    : renderCode(source, language);

  fs.mkdirSync(contentDir, { recursive: true });
  fs.writeFileSync(
    path.join(contentDir, 'index.html'),
    `<article class="readable-entry" data-path="${escapeHtml(repoPath)}">${body}</article>`,
  );

  return {
    id,
    repoPath,
    title: titleFor(repoPath),
    kind,
    language,
    contentUrl: `/readable/content/${id}/`,
  };
}

fs.rmSync(outputRoot, { recursive: true, force: true });
fs.mkdirSync(contentRoot, { recursive: true });

const files = includeRoots.flatMap((root) => walk(root)).sort();
const entries = files.map(renderEntry);

fs.writeFileSync(
  path.join(outputRoot, 'manifest.json'),
  JSON.stringify({
    generatedAt: new Date().toISOString(),
    entries,
  }, null, 2),
);

import React, { useState, useMemo } from 'react';
import katex from 'katex';
import { Highlight } from 'prism-react-renderer';
import { usePrismTheme } from '@docusaurus/theme-common';
import styles from './FormulaCode.module.css';

// ── 类型定义 ──────────────────────────────────────────────

interface FormulaPart {
  label: string;
  tex: string;
}

interface FormulaCodeMapping {
  partIndex: number;
  lines: number[];
}

interface FormulaCodeProps {
  formulaPrefix?: string;
  formulaParts: FormulaPart[];
  code: string;
  language?: string;
  mappings: FormulaCodeMapping[];
  title?: string;
}

// ── 组件 ──────────────────────────────────────────────────

export default function FormulaCode({
  formulaPrefix,
  formulaParts,
  code,
  language = 'python',
  mappings,
  title,
}: FormulaCodeProps) {
  const [activePartIndex, setActivePartIndex] = useState<number | null>(null);
  const [activeLines, setActiveLines] = useState<Set<number>>(new Set());
  const prismTheme = usePrismTheme();

  // 预计算双向查找表
  const partToLines = useMemo(() => {
    const map = new Map<number, number[]>();
    for (const m of mappings) {
      map.set(m.partIndex, m.lines);
    }
    return map;
  }, [mappings]);

  const lineToParts = useMemo(() => {
    const map = new Map<number, number[]>();
    for (const m of mappings) {
      for (const line of m.lines) {
        const existing = map.get(line) ?? [];
        existing.push(m.partIndex);
        map.set(line, existing);
      }
    }
    return map;
  }, [mappings]);

  // ── 事件处理 ───────────────────────────────────────────

  const handlePartEnter = (index: number) => {
    const lines = partToLines.get(index) ?? [];
    setActivePartIndex(index);
    setActiveLines(new Set(lines));
  };

  const handlePartLeave = () => {
    setActivePartIndex(null);
    setActiveLines(new Set());
  };

  const handleLineEnter = (line: number) => {
    const parts = lineToParts.get(line) ?? [];
    if (parts.length > 0) {
      setActivePartIndex(parts[0]);
      // 合并所有相关代码行
      const allLines = new Set<number>();
      for (const p of parts) {
        const ls = partToLines.get(p) ?? [];
        for (const l of ls) allLines.add(l);
      }
      setActiveLines(allLines);
    }
  };

  const handleLineLeave = () => {
    setActivePartIndex(null);
    setActiveLines(new Set());
  };

  // ── KaTeX 渲染 ─────────────────────────────────────────

  const renderKatex = (tex: string): string => {
    try {
      return katex.renderToString(tex, {
        displayMode: false,
        throwOnError: false,
        strict: false,
      });
    } catch {
      return tex;
    }
  };

  const prefixHtml = formulaPrefix ? renderKatex(formulaPrefix) : null;
  const partHtmls = formulaParts.map((p) => renderKatex(p.tex));

  const hasActive = activePartIndex !== null;

  return (
    <div className={styles.formulaCodeContainer}>
      {/* ── 公式区域 ── */}
      <div className={styles.formulaArea}>
        {prefixHtml && (
          <span className={styles.formulaPrefix} dangerouslySetInnerHTML={{ __html: prefixHtml }} />
        )}
        {formulaParts.map((part, i) => (
          <span
            key={i}
            className={`${styles.formulaPartWrapper}`}
            onMouseEnter={() => handlePartEnter(i)}
            onMouseLeave={handlePartLeave}
          >
            <span
              className={`${styles.formulaPart} ${activePartIndex === i ? styles.formulaPartActive : ''}`}
              dangerouslySetInnerHTML={{ __html: partHtmls[i] }}
            />
            <span
              className={`${styles.tooltip} ${activePartIndex === i ? styles.tooltipVisible : ''}`}
            >
              {part.label}
            </span>
          </span>
        ))}
      </div>

      {/* ── 连接指示条 ── */}
      <div className={`${styles.connector} ${hasActive ? styles.connectorVisible : ''}`} />

      {/* ── 代码区域 ── */}
      <div className={styles.codeArea}>
        {title && <div className={styles.codeTitle}>{title}</div>}
        <Highlight theme={prismTheme} code={code.trimEnd()} language={language}>
          {({ tokens, getTokenProps }) => (
            <pre className={styles.codeBlock}>
              {tokens.map((line, i) => (
                <div
                  key={i}
                  className={`${styles.codeLine} ${activeLines.has(i) ? styles.codeLineActive : ''}`}
                  onMouseEnter={() => handleLineEnter(i)}
                  onMouseLeave={handleLineLeave}
                >
                  {line.map((token, j) => (
                    <span key={j} {...getTokenProps({ token })} />
                  ))}
                </div>
              ))}
            </pre>
          )}
        </Highlight>
      </div>
    </div>
  );
}
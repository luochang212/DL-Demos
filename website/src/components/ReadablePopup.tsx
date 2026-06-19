import React, {useEffect, useMemo, useState} from 'react';
import useBaseUrl from '@docusaurus/useBaseUrl';
import styles from './ReadablePopup.module.css';

export type ReadableEntry = {
  id: string;
  repoPath: string;
  title: string;
  kind: 'code' | 'formula' | 'lean' | 'markdown';
  language?: string;
  contentUrl: string;
};

type Props = {
  entry: ReadableEntry | null;
  onClose: () => void;
};

export function useReadableContent(entry: ReadableEntry | null) {
  const baseUrl = useBaseUrl('/');
  const [html, setHtml] = useState('');
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (!entry) {
      setHtml('');
      return;
    }

    let cancelled = false;
    setLoading(true);
    fetch(`${baseUrl.replace(/\/$/, '')}${entry.contentUrl}`)
      .then((response) => {
        if (!response.ok) {
          throw new Error(`Failed to load ${entry.contentUrl}`);
        }
        return response;
      })
      .then((response) => response.text())
      .then((content) => {
        if (!cancelled) {
          setHtml(content);
          setLoading(false);
        }
      })
      .catch(() => {
        if (!cancelled) {
          setHtml('<p>内容加载失败。</p>');
          setLoading(false);
        }
      });

    return () => {
      cancelled = true;
    };
  }, [baseUrl, entry]);

  return {html, loading};
}

export function readableSharePath(repoPath: string) {
  return `/read?path=${encodeURIComponent(repoPath)}`;
}

export function readableEmbedPath(repoPath: string) {
  return `${readableSharePath(repoPath)}&embed=1`;
}

export default function ReadablePopup({entry, onClose}: Props) {
  const baseUrl = useBaseUrl('/');

  const shareUrl = useMemo(() => {
    if (!entry || typeof window === 'undefined') {
      return '';
    }
    return new URL(
      `${baseUrl.replace(/\/$/, '')}${readableSharePath(entry.repoPath)}`,
      window.location.origin,
    ).toString();
  }, [baseUrl, entry]);

  const embedUrl = useMemo(() => {
    if (!entry) {
      return '';
    }
    return `${baseUrl.replace(/\/$/, '')}${readableEmbedPath(entry.repoPath)}`;
  }, [baseUrl, entry]);

  useEffect(() => {
    function onKeyDown(event: KeyboardEvent) {
      if (event.key === 'Escape') {
        onClose();
      }
    }

    if (entry) {
      window.addEventListener('keydown', onKeyDown);
    }
    return () => {
      window.removeEventListener('keydown', onKeyDown);
    };
  }, [entry, onClose]);

  if (!entry) {
    return null;
  }

  async function copyLink() {
    if (shareUrl) {
      await navigator.clipboard.writeText(shareUrl);
    }
  }

  return (
    <>
      <div className={styles.backdrop} onClick={onClose} />
      <aside className={styles.panel} aria-label="Readable file popup">
        <div className={styles.header}>
          <div className={styles.meta}>
            <div className={styles.title}>{entry.title}</div>
            <div className={styles.path}>{entry.repoPath}</div>
          </div>
          <div className={styles.actions}>
            <button className={styles.button} type="button" onClick={copyLink}>
              复制链接
            </button>
            <a className={styles.button} href={shareUrl} target="_blank" rel="noreferrer">
              新标签页
            </a>
            <button className={styles.button} type="button" onClick={onClose}>
              关闭
            </button>
          </div>
        </div>
        <iframe className={styles.frame} src={embedUrl} title={entry.title} />
      </aside>
    </>
  );
}

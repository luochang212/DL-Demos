import React, {useEffect, useMemo, useState} from 'react';
import Head from '@docusaurus/Head';
import useBaseUrl from '@docusaurus/useBaseUrl';
import {ReadableEntry, readableSharePath, useReadableContent} from '@site/src/components/ReadablePopup';
import styles from './read.module.css';

type Manifest = {
  entries: ReadableEntry[];
};

function normalizeRepoPath(value: string | null) {
  if (!value) {
    return null;
  }
  return decodeURIComponent(value).replace(/^\/+/, '').split(/[?#]/u)[0];
}

export default function ReadPage() {
  const baseUrl = useBaseUrl('/');
  const [entry, setEntry] = useState<ReadableEntry | null>(null);
  const [embed, setEmbed] = useState(false);
  const {html, loading} = useReadableContent(entry);

  useEffect(() => {
    const search = new URLSearchParams(window.location.search);
    const targetPath = normalizeRepoPath(search.get('path'));
    setEmbed(search.get('embed') === '1');
    if (!targetPath) {
      return;
    }

    fetch(`${baseUrl.replace(/\/$/, '')}/readable/manifest.json`)
      .then((response) => response.json())
      .then((manifest: Manifest) => {
        setEntry(manifest.entries.find((candidate) => candidate.repoPath === targetPath) ?? null);
      })
      .catch(() => setEntry(null));
  }, [baseUrl]);

  const title = entry?.title ?? 'Readable file';
  const shareUrl = useMemo(() => {
    if (!entry || typeof window === 'undefined') {
      return '';
    }
    return new URL(
      `${baseUrl.replace(/\/$/, '')}${readableSharePath(entry.repoPath)}`,
      window.location.origin,
    ).toString();
  }, [baseUrl, entry]);

  async function copyLink() {
    if (shareUrl) {
      await navigator.clipboard.writeText(shareUrl);
    }
  }

  return (
    <>
      <Head>
        <title>{title}</title>
        <meta name="description" content="Readable local source file" />
      </Head>
      <main className={`${styles.page} ${embed ? styles.embedded : ''}`}>
        {!embed && (
          <div className={styles.toolbar}>
            <div className={styles.meta}>
              <div className={styles.title}>{title}</div>
              <div className={styles.path}>{entry?.repoPath ?? '未找到内容'}</div>
            </div>
            <button className={styles.button} type="button" onClick={copyLink}>
              复制链接
            </button>
          </div>
        )}
        <section className={styles.content}>
          {loading ? <p>加载中...</p> : <div dangerouslySetInnerHTML={{__html: html || '<p>未找到内容。</p>'}} />}
        </section>
      </main>
    </>
  );
}

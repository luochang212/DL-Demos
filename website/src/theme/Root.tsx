import React, {useCallback, useEffect, useMemo, useState} from 'react';
import useBaseUrl from '@docusaurus/useBaseUrl';
import ReadablePopup, {ReadableEntry} from '@site/src/components/ReadablePopup';

type Manifest = {
  entries: ReadableEntry[];
};

const githubBlobPattern =
  /^https:\/\/github\.com\/luochang212\/DL-Demos\/blob\/(?:master|main)\/(.+)$/u;

function normalizeRepoPath(value: string) {
  const clean = decodeURIComponent(value).replace(/^\/+/, '').split(/[?#]/u)[0];
  return clean;
}

function hrefToRepoPath(href: string, baseUrl: string, currentHref: string) {
  const githubMatch = githubBlobPattern.exec(href);
  if (githubMatch) {
    return normalizeRepoPath(githubMatch[1]);
  }

  if (href.startsWith('chapters/') || href.startsWith('/chapters/')) {
    return normalizeRepoPath(href);
  }

  const url = new URL(href, currentHref);
  if (url.origin !== window.location.origin) {
    return null;
  }

  const basePath = baseUrl.replace(/\/$/, '');
  if (!url.pathname.startsWith(basePath)) {
    return null;
  }

  const sitePath = url.pathname.slice(basePath.length).replace(/^\/+/, '');
  return normalizeRepoPath(sitePath);
}

function isModifiedClick(event: MouseEvent) {
  return event.metaKey || event.ctrlKey || event.shiftKey || event.altKey || event.button !== 0;
}

export default function Root({children}: {children: React.ReactNode}) {
  const baseUrl = useBaseUrl('/');
  const [manifest, setManifest] = useState<Manifest | null>(null);
  const [activePath, setActivePath] = useState<string | null>(null);

  useEffect(() => {
    fetch(`${baseUrl.replace(/\/$/, '')}/readable/manifest.json`)
      .then((response) => response.json())
      .then((data: Manifest) => setManifest(data))
      .catch(() => setManifest({entries: []}));
  }, [baseUrl]);

  const entryByPath = useMemo(() => {
    const map = new Map<string, ReadableEntry>();
    for (const entry of manifest?.entries ?? []) {
      map.set(entry.repoPath, entry);
    }
    return map;
  }, [manifest]);

  const closePopup = useCallback(() => {
    const url = new URL(window.location.href);
    url.searchParams.delete('read');
    window.history.pushState({}, '', url);
    setActivePath(null);
  }, []);

  useEffect(() => {
    if (!activePath) {
      return;
    }

    const originalOverflow = document.body.style.overflow;
    const originalOverscrollBehavior = document.body.style.overscrollBehavior;
    const originalPosition = document.body.style.position;
    const originalTop = document.body.style.top;
    const originalWidth = document.body.style.width;
    const originalDocumentOverflow = document.documentElement.style.overflow;
    const originalDocumentOverscrollBehavior = document.documentElement.style.overscrollBehavior;
    const scrollY = window.scrollY;
    document.body.style.overflow = 'hidden';
    document.body.style.overscrollBehavior = 'none';
    document.body.style.position = 'fixed';
    document.body.style.top = `-${scrollY}px`;
    document.body.style.width = '100%';
    document.documentElement.style.overflow = 'hidden';
    document.documentElement.style.overscrollBehavior = 'none';

    return () => {
      document.body.style.overflow = originalOverflow;
      document.body.style.overscrollBehavior = originalOverscrollBehavior;
      document.body.style.position = originalPosition;
      document.body.style.top = originalTop;
      document.body.style.width = originalWidth;
      document.documentElement.style.overflow = originalDocumentOverflow;
      document.documentElement.style.overscrollBehavior = originalDocumentOverscrollBehavior;
      window.scrollTo(0, scrollY);
    };
  }, [activePath]);

  useEffect(() => {
    if (!manifest) {
      return;
    }

    const search = new URLSearchParams(window.location.search);
    const readPath = search.get('read');
    if (readPath && entryByPath.has(normalizeRepoPath(readPath))) {
      setActivePath(normalizeRepoPath(readPath));
    }
  }, [entryByPath, manifest]);

  useEffect(() => {
    if (!manifest) {
      return;
    }

    function onPopState() {
      const readPath = new URLSearchParams(window.location.search).get('read');
      const normalized = readPath ? normalizeRepoPath(readPath) : null;
      setActivePath(
        normalized && entryByPath.has(normalized) ? normalized : null
      );
    }

    window.addEventListener('popstate', onPopState);
    return () => window.removeEventListener('popstate', onPopState);
  }, [entryByPath, manifest]);

  useEffect(() => {
    if (!manifest) {
      return;
    }

    function onClick(event: MouseEvent) {
      if (event.defaultPrevented || isModifiedClick(event)) {
        return;
      }

      const target = event.target as HTMLElement | null;
      const anchor = target?.closest('a');
      if (!(anchor instanceof HTMLAnchorElement)) {
        return;
      }

      const href = anchor.getAttribute('href');
      if (!href || href.startsWith('#')) {
        return;
      }

      const repoPath = hrefToRepoPath(href, baseUrl, window.location.href);
      if (!repoPath || !entryByPath.has(repoPath)) {
        return;
      }

      event.preventDefault();
      const url = new URL(window.location.href);
      url.searchParams.set('read', repoPath);
      window.history.pushState({}, '', url);
      setActivePath(repoPath);
    }

    document.addEventListener('click', onClick);
    return () => {
      document.removeEventListener('click', onClick);
    };
  }, [baseUrl, entryByPath, manifest]);

  const activeEntry = activePath ? entryByPath.get(activePath) ?? null : null;

  return (
    <>
      {children}
      <ReadablePopup entry={activeEntry} onClose={closePopup} />
    </>
  );
}

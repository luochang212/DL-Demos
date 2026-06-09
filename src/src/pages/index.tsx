import React from 'react';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Link from '@docusaurus/Link';

export default function Home() {
  const {siteConfig} = useDocusaurusContext();
  return (
    <main style={{
      display: 'flex',
      flexDirection: 'column',
      alignItems: 'center',
      justifyContent: 'center',
      minHeight: '80vh',
      padding: '2rem',
    }}>
      <h1 style={{fontSize: '3rem', marginBottom: '0.5rem'}}>
        📚 DL-Demos 深度学习教程
      </h1>
      <p style={{fontSize: '1.2rem', color: 'var(--ifm-color-emphasis-600)', marginBottom: '2rem'}}>
        从原理到代码，深度学习经典算法的逐行解析
      </p>
      <div style={{display: 'flex', gap: '1rem', flexWrap: 'wrap', justifyContent: 'center'}}>
        <Link
          className="button button--primary button--lg"
          to="/docs/">
          开始阅读
        </Link>
        <Link
          className="button button--secondary button--lg"
          to="https://github.com/luochang212/DL-Demos">
          GitHub 仓库
        </Link>
      </div>
    </main>
  );
}

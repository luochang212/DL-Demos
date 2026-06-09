/** @type {import('@docusaurus/plugin-content-docs').SidebarsConfig} */
const sidebars = {
  tutorialSidebar: [
    {
      type: 'doc',
      id: 'intro',
      label: '🏠 项目简介',
    },
    {
      type: 'category',
      label: '🎨 生成模型',
      collapsible: true,
      collapsed: false,
      link: {
        type: 'generated-index',
        title: '生成模型系列',
        description:
          '从 VAE 到扩散模型，系统讲解深度生成模型的核心原理与 PyTorch 实现。',
        slug: '/generative-models',
      },
      items: [
        {
          type: 'doc',
          id: 'generative-models/vae',
          label: '1. VAE — 变分自编码器',
        },
        {
          type: 'doc',
          id: 'generative-models/ddpm',
          label: '2. DDPM — 去噪扩散概率模型',
        },
        {
          type: 'doc',
          id: 'generative-models/ddim',
          label: '3. DDIM — 去噪扩散隐式模型',
        },
        {
          type: 'doc',
          id: 'generative-models/pixelcnn',
          label: '4. PixelCNN — 自回归像素生成',
        },
        {
          type: 'doc',
          id: 'generative-models/vqvae',
          label: '5. VQVAE — 向量量化变分自编码器',
        },
      ],
    },
    {
      type: 'category',
      label: '⚙️ 训练技巧',
      collapsible: true,
      collapsed: false,
      link: {
        type: 'generated-index',
        title: '训练技巧系列',
        description:
          '从参数初始化到优化器，系统讲解深度学习训练中的关键技巧与从零实现。',
        slug: '/training-tricks',
      },
      items: [
        {
          type: 'doc',
          id: 'training-tricks/initialization',
          label: '1. 参数初始化',
        },
        {
          type: 'doc',
          id: 'training-tricks/regularization',
          label: '2. 正则化',
        },
        {
          type: 'doc',
          id: 'training-tricks/optimizer',
          label: '3. 优化器',
        },
      ],
    },
    {
      type: 'category',
      label: '🚀 工程实践',
      collapsible: true,
      collapsed: false,
      link: {
        type: 'generated-index',
        title: '工程实践系列',
        description:
          '从隐式表示到分布式训练，掌握深度学习前沿工程技巧。',
        slug: '/engineering',
      },
      items: [
        {
          type: 'doc',
          id: 'engineering/fourier-feature',
          label: '1. 傅里叶特征',
        },
        {
          type: 'doc',
          id: 'engineering/distributed',
          label: '2. 分布式训练',
        },
      ],
    },
  ],
};

export default sidebars;

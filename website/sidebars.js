/** @type {import('@docusaurus/plugin-content-docs').SidebarsConfig} */
const sidebars = {
  tutorialSidebar: [
    {
      type: 'doc',
      id: 'intro',
      label: '🏠 项目简介',
    },
    {
      type: 'doc',
      id: 'getting-started/environment',
      label: '🧰 环境、运行与验证',
    },
    {
      type: 'category',
      label: '🧠 基础神经网络',
      collapsible: true,
      collapsed: false,
      link: {
        type: 'generated-index',
        title: '基础神经网络系列',
        description:
          '从 Logistic Regression 到深层 MLP，建立前向传播、反向传播与分类损失的完整基础。',
        slug: '/fundamentals',
      },
      items: [
        'fundamentals/logistic-regression',
        'fundamentals/shallow-network',
        'fundamentals/deep-network',
        'fundamentals/multiclass-classification',
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
      label: '🖼️ CNN 与视觉',
      collapsible: true,
      collapsed: false,
      link: {
        type: 'generated-index',
        title: 'CNN 与视觉系列',
        description:
          '从卷积运算的基础实现到残差网络，系统讲解 CNN 的核心原理与代码实现。',
        slug: '/cv',
      },
      items: [
        {
          type: 'doc',
          id: 'cv/basic-cnn',
          label: '1. CNN 基础',
        },
        {
          type: 'doc',
          id: 'cv/resnet',
          label: '2. ResNet — 残差网络',
        },
        {
          type: 'doc',
          id: 'cv/style-transfer',
          label: '3. Style Transfer',
        },
        {
          type: 'doc',
          id: 'cv/iou-nms',
          label: '4. IoU 与 NMS',
        },
      ],
    },
    {
      type: 'category',
      label: '🔗 序列模型',
      collapsible: true,
      collapsed: false,
      link: {
        type: 'generated-index',
        title: '序列模型系列',
        description:
          '从循环神经网络到情感分析实战，掌握序列建模的核心方法。',
        slug: '/sequence-models',
      },
      items: [
        {
          type: 'doc',
          id: 'sequence-models/rnn',
          label: '1. RNN — 循环神经网络',
        },
        {
          type: 'doc',
          id: 'sequence-models/sentiment-analysis',
          label: '2. 情感分析',
        },
        {
          type: 'doc',
          id: 'sequence-models/attention',
          label: '3. Attention',
        },
        {
          type: 'doc',
          id: 'sequence-models/transformer',
          label: '4. Transformer',
        },
      ],
    },
    {
      type: 'category',
      label: '🎨 生成模型',
      collapsible: true,
      collapsed: false,
      link: {
        type: 'doc',
        id: 'generative-models/overview',
      },
      items: [
        {
          type: 'doc',
          id: 'generative-models/overview',
          label: '0. 概览',
        },
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

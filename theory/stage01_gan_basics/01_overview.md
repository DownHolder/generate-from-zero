# 1. 什么是生成对抗网络

**生成对抗网络（Generative Adversarial Network, GAN）** 是一类生成模型，由两个神经网络——**生成器（generator）** 和**判别器（discriminator）**——通过相互博弈来学习数据的分布。这个框架由 Ian Goodfellow 等人于 2014 年提出（Goodfellow et al., 2014）。

## 1.1 问题设定

假设我们有一个数据集，其中每个样本 $\boldsymbol{x}$ 来自某个未知的真实分布 $p_{\text{data}}(\boldsymbol{x})$。我们的目标是学习一个模型，能够从这个分布中采样出新的、逼真的样本。

直接对 $p_{\text{data}}$ 建模是困难的。高维数据的分布极其复杂——以 MNIST 为例，28×28 的手写数字图像生活在 $784$ 维空间中，明确写出 $p(\boldsymbol{x})$ 的解析形式几乎不可能。

## 1.2 核心直觉

GAN 用一个巧妙的间接方法绕开了这个问题。我们不直接对 $p_{\text{data}}$ 建模，而是训练一个生成器 $G$，让它从简单的噪声分布 $p_{\boldsymbol{z}}(\boldsymbol{z})$（例如标准正态分布）开始，经过一系列非线性变换，逐步将噪声映射成逼真的数据样本。

为了判断生成器产出的质量，我们同时训练一个判别器 $D$。$D$ 的任务很简单：分辨输入是来自真实数据集还是来自生成器。$G$ 的任务则是尽可能欺骗 $D$——产出的样本要让 $D$ 无法与真实数据区分。

这个设定类似于**伪造者与警察的博弈**。伪造者不断精进技艺，警察不断学习辨别真伪。当警察无法区分真伪（即 $D$ 对任何输入都输出 $0.5$）时，伪造者就成功了。

## 1.3 形式化定义

生成器是一个可微函数 $G(\boldsymbol{z}; \theta^{(g)})$，将噪声 $\boldsymbol{z} \sim p_{\boldsymbol{z}}$ 映射到数据空间。判别器是另一个可微函数 $D(\boldsymbol{x}; \theta^{(d)})$，输出一个标量，表示 $\boldsymbol{x}$ 来自真实数据（而非生成器）的概率。

我们同时训练 $D$ 使其最大化正确分类的概率，训练 $G$ 使其最小化 $\log(1 - D(G(\boldsymbol{z})))$。这构成了一个**极小极大博弈（minimax game）**：

$$
\min_G \max_D V(D, G) = \mathbb{E}_{\boldsymbol{x} \sim p_{\text{data}}}[\log D(\boldsymbol{x})]
+ \mathbb{E}_{\boldsymbol{z} \sim p_{\boldsymbol{z}}}[\log(1 - D(G(\boldsymbol{z})))]
$$

这个公式的意思是：

- 判别器 $D$ 希望 $\log D(\boldsymbol{x})$ 更大（对真实数据输出接近 $1$），同时 $\log(1 - D(G(\boldsymbol{z})))$ 也更大（对生成数据输出接近 $0$）。因此 $D$ 试图**最大化** $V$。
- 生成器 $G$ 希望 $D(G(\boldsymbol{z}))$ 接近 $1$，即 $\log(1 - D(G(\boldsymbol{z})))$ 更小。因此 $G$ 试图**最小化** $V$。

## 1.4 非饱和损失

在实际训练中，上述 minimax 目标对 $G$ 的梯度在训练早期过于微弱。当 $D$ 能够轻松分辨真假时，$\log(1 - D(G(\boldsymbol{z})))$ 处于饱和区，导数接近零。Goodfellow 等人建议改为让 $G$ **最大化** $\log D(G(\boldsymbol{z}))$，这与原目标在固定点上等价，但提供了更强的早期梯度信号。

$$
\text{generator loss} = -\mathbb{E}_{\boldsymbol{z} \sim p_{\boldsymbol{z}}}[\log D(G(\boldsymbol{z}))]
$$

这就是非饱和损失（non-saturating loss）——GAN 训练中最常用的形式。

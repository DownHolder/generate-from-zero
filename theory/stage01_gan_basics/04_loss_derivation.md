# 4. 损失函数推导

本章从信息论和博弈论两个角度，完整推导 GAN 的损失函数。我们假设读者熟悉交叉熵和 KL 散度的基本定义。

## 4.1 从二分类问题出发

$D$ 本质上是一个二分类器。对于输入 $\boldsymbol{x}$，它输出 $\boldsymbol{x}$ 属于"真实数据"类别的概率（而非"生成数据"类别）。二分类交叉熵的标准形式为：

$$
\ell_{\text{BCE}} = -\frac{1}{m} \sum_{i=1}^m \left[ y_i \log \hat{y}_i + (1 - y_i) \log(1 - \hat{y}_i) \right]
$$

其中 $y_i \in \{0, 1\}$ 是真标签，$\hat{y}_i \in (0, 1)$ 是预测概率。对于真实样本 $y_i = 1$，损失变为 $-\log \hat{y}_i$；对于生成样本 $y_i = 0$，损失变为 $-\log(1 - \hat{y}_i)$。

## 4.2 D 的视角：最大化正确分类概率

在一个 batch 中有 $m$ 个真实样本和 $m$ 个生成样本。$D$ 的输出记为 $D(\boldsymbol{x}) \in (0, 1)$。$D$ 希望：

- 对真实样本 $\boldsymbol{x}^{(i)}$：$D(\boldsymbol{x}^{(i)}) \to 1$，即 $\log D(\boldsymbol{x}^{(i)}) \to 0^-$
- 对生成样本 $G(\boldsymbol{z}^{(i)})$：$D(G(\boldsymbol{z}^{(i)})) \to 0$，即 $\log (1 - D(G(\boldsymbol{z}^{(i)}))) \to 0^-$

因此 $D$ 要最大化：

$$
\frac{1}{m} \sum_{i=1}^m \log D(\boldsymbol{x}^{(i)}) + \frac{1}{m} \sum_{i=1}^m \log(1 - D(G(\boldsymbol{z}^{(i)})))
$$

当 $m \to \infty$ 时，这等价于最大化期望形式：

$$
V(D, G) = \mathbb{E}_{\boldsymbol{x} \sim p_{\text{data}}}[\log D(\boldsymbol{x})] + \mathbb{E}_{\boldsymbol{z} \sim p_{\boldsymbol{z}}}[\log(1 - D(G(\boldsymbol{z})))]
$$

## 4.3 G 的视角：最小化被识破的概率

$G$ 无法控制真实样本，只能控制 $D$ 对假样本的判别结果。在原 minimax 目标中，$G$ 要最小化 $\log(1 - D(G(\boldsymbol{z})))$：

$$
G^* = \arg\min_G \mathbb{E}_{\boldsymbol{z}} [\log(1 - D(G(\boldsymbol{z})))]
$$

但正如我们在 §1.4 讨论的，这个目标在训练早期梯度饱和。因此实践中 $G$ 改为最大化 $\log D(G(\boldsymbol{z}))$，等价于最小化：

$$
J^{(G)} = -\mathbb{E}_{\boldsymbol{z}} [\log D(G(\boldsymbol{z}))]
$$

这被称为**非饱和生成器损失**（non-saturating generator loss）。

## 4.4 全局最优解：$p_g = p_{\text{data}}$

固定 $G$ 时，我们可以求出最优 $D$。对 $D$ 的期望积分形式求导：

$$
V = \int_{\boldsymbol{x}} \left[ p_{\text{data}}(\boldsymbol{x}) \log D(\boldsymbol{x}) + p_g(\boldsymbol{x}) \log(1 - D(\boldsymbol{x})) \right] d\boldsymbol{x}
$$

被积函数形如 $f(D) = a \log D + b \log(1 - D)$，在 $D^* = a / (a + b)$ 处取得最大值。因此在每个点 $\boldsymbol{x}$ 上：

$$
D^*(\boldsymbol{x}) = \frac{p_{\text{data}}(\boldsymbol{x})}{p_{\text{data}}(\boldsymbol{x}) + p_g(\boldsymbol{x})}
\tag{4.1}
$$

将 $D^*$ 代入 $V$：

$$
\begin{aligned}
V(D^*, G) &= \mathbb{E}_{\boldsymbol{x} \sim p_{\text{data}}} \left[ \log \frac{p_{\text{data}}}{p_{\text{data}} + p_g} \right]
+ \mathbb{E}_{\boldsymbol{x} \sim p_g} \left[ \log \frac{p_g}{p_{\text{data}} + p_g} \right] \\
&= -\log 4 + \text{KL}\left(p_{\text{data}} \middle\| \frac{p_{\text{data}} + p_g}{2}\right) + \text{KL}\left(p_g \middle\| \frac{p_{\text{data}} + p_g}{2}\right) \\
&= -\log 4 + 2 \cdot \text{JSD}(p_{\text{data}} \| p_g)
\end{aligned}
$$

JS 散度是非负的，当且仅当 $p_g = p_{\text{data}}$ 时取零。因此 $V$ 的全局最小值为 $-\log 4$，在 $p_g = p_{\text{data}}$ 时达到。

这个推导意味着：**如果我们能够训练 $D$ 到最优然后更新 $G$，那么每一步 $G$ 的更新都在减小 $p_g$ 与 $p_{\text{data}}$ 之间的 JS 散度。**

## 4.5 为什么实践中达不到最优

上述推导假设 $D$ 在每次 $G$ 更新前都达到了最优。在实践中，我们只运行一步（或少步）梯度下降来更新 $D$。此外，$G$ 和 $D$ 的容量是有限的——MLP 网络无法完美表示最优判别函数。因此实际训练的轨迹偏离了 JS 散度最小化的理论路径。

这种理论与实践的差距，正是后续 GAN 变体（DCGAN, WGAN-GP）试图弥合的。它们通过更好的架构和更合适的距离度量，使实际训练行为更接近理论预期。

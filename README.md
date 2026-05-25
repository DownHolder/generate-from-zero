# generative-from-zero

从零开始构建生成式（图像）模型的教学项目。

## 项目结构

```
generative_from_zero/
├── code/        # 工程路线
├── theory/      # 理论路线
```

## 学习路线

| Stage | 模型 | 核心知识 | 难度 |
|-------|------|---------|:----:|
| 01 | MLP GAN | 对抗训练范式，minimax game | ★ |
| 02 | DCGAN | 卷积 GAN，BatchNorm | ★ |
| 03 | cGAN | 条件生成，标签引导 | ★★ |
| 04 | WGAN-GP | Wasserstein 距离，Gradient Penalty | ★★ |
| 05 | VAE | 变分推断，ELBO，重参数化 | ★★ |

## 快速开始

```bash
pip install -r requirements.txt
cd code/stage01_gan_basics
python main.py       # 训练
python generate.py   # 生成图片
```

## 设计理念

- **对比式学习**：每个 stage 只保留与上一阶段的**差异**，公共代码抽取到 `code/common/`
- **两条路线**：`code/` 动手实现，`theory/` 理论深入
- **从零开始**：从 MLP GAN 开始

## 相关资源

- [PLAN.md](./PLAN.md) — 完整学习路线与论文清单
- [model_comparison.md](./model_comparison.md) — 5 个模型的对比分析报告

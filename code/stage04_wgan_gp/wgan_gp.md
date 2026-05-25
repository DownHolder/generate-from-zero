# Stage 04: WGAN-GP — 更好的距离度量

> **前提**：请先完成 [Stage 02](../stage02_dcgan/dcgan.md)（DCGAN），理解卷积 GAN 架构。
>
> **本阶段改了 models.py 和 train.py**。Generator 架构与 DCGAN 完全相同。

## 回顾：DCGAN 的训练问题

看 model_comparison.md 中 DCGAN 的数据：

```
D_loss = 0.28     ← D 几乎完美分辨真假（理想均衡值是 1.386）
G_loss = 3.15     ← G 完全骗不过 D（理想均衡值是 0.693）
方差  = 0.82      ← 训练剧烈震荡
```

根源在 **JS 散度**：当真实分布和生成分布几乎不重叠时（高维空间中是常态），JS 散度退化为常数 `log 2`，G 的梯度消失。

## 理论：JS 散度 → Wasserstein 距离

WGAN 的核心想法：换一个更好的距离度量。

- **JS 散度**：分布不重叠时 = log 2（常数），梯度 = 0
- **Wasserstein 距离（推土机距离）**："把一堆土从分布 P 搬到分布 Q 的最少工作量"，处处平滑

WGAN 将 Wasserstein 距离转化为 critic（不再叫 discriminator）的优化目标：

$$ W(P_r, P_g) = \max_{||f||_{Lip} \le 1} \mathbb{E}_{x \sim P_r}[f(x)] - \mathbb{E}_{x \sim P_g}[f(x)] $$

其中 `||f||_{Lip} ≤ 1` 是 Lipschitz 约束——函数不能变化太快。**WGAN-GP 用 Gradient Penalty（梯度惩罚）来实现这个约束**：

$$ GP = \lambda \cdot \mathbb{E}_{\hat{x}} \left[(||\nabla_{\hat{x}} f(\hat{x})||_2 - 1)^2\right] $$

直觉：在真假图像的连线（插值）上，惩罚那些梯度范数偏离 1 的点。

## 实现：两个文件变了

### config.py

```python
from pathlib import Path

# ========== Paths ==========
ROOT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT_DIR.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
SAMPLE_DIR = ROOT_DIR / "samples"
CHECKPOINT_DIR = ROOT_DIR / "checkpoints"

# ========== Dataset ==========
DATASET = "mnist"
IMAGE_SIZE = 28
CHANNELS = 1
NUM_WORKERS = 8

# ========== Model Architecture ==========
LATENT_DIM = 100
DCGAN_FEATURES = 64

# ========== WGAN-GP (特有) ==========
N_CRITIC = 5
LAMBDA_GP = 10

# ========== Training ==========
BATCH_SIZE = 128
EPOCHS = 50
LEARNING_RATE = 0.0002
BETA1 = 0.5
SAMPLE_EVERY = 1
CHECKPOINT_EVERY = 10

# ========== Inference ==========
NUM_SAMPLE_IMAGES = 64

# ========== Runtime ==========
DEVICE = "auto"
SEED = 42
MIN_GPU_MEMORY_MB = 4096
```

新增 `N_CRITIC = 5`（critic 每轮更新 5 次，generator 更新 1 次）和 `LAMBDA_GP = 10`（梯度惩罚的权重）。

### models.py — Critic 去掉 BatchNorm

```python
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, latent_dim, channels, image_size, features=64):
        super().__init__()
        self.latent_dim = latent_dim
        self.channels = channels
        self.image_size = image_size
        self.features = features

        mult = features * 4
        self.init_size = image_size // 4

        self.fc = nn.Sequential(
            nn.Linear(latent_dim, mult * self.init_size * self.init_size),
            nn.ReLU(inplace=True),
        )

        self.upsample = nn.Sequential(
            nn.BatchNorm2d(mult),
            nn.ConvTranspose2d(mult, features * 2, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(features * 2),
            nn.ConvTranspose2d(features * 2, features, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(features),
            nn.ConvTranspose2d(features, channels, 3, 1, 1),
            nn.Tanh(),
        )

    def forward(self, z):
        out = self.fc(z)
        out = out.view(out.size(0), self.features * 4, self.init_size, self.init_size)
        return self.upsample(out)


class Critic(nn.Module):
    def __init__(self, channels, image_size, features=64):
        super().__init__()
        self.channels = channels
        self.image_size = image_size
        self.features = features

        self.net = nn.Sequential(
            nn.Conv2d(channels, features, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(features, features * 2, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(features * 2, features * 4, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(features * 4, 1, 3, 1, 0),
            nn.Flatten(),
        )

    def forward(self, images):
        return self.net(images)
```

**Generator 完全不变**（与 DCGAN 相同）。

**Discriminator → Critic 的变化**：

```
DCGAN Discriminator:           WGAN-GP Critic:
    Conv2d → LeakyReLU            Conv2d → LeakyReLU
    Conv2d → BN → LeakyReLU       Conv2d → LeakyReLU    ← 注意：没有 BN
    Conv2d → BN → LeakyReLU       Conv2d → LeakyReLU    ← 也没有 BN
    Conv2d → Flatten → logit      Conv2d → Flatten → score
```

**为什么去掉 BatchNorm？** Gradient Penalty 需要对**每个样本**独立计算梯度。BatchNorm 会引入 batch 内样本间的依赖，使得单个样本的梯度不再独立，违反了 Lipschitz 约束条件。

另外，Critic 的输出不再叫 logit——它是一个**实数分数**（可正可负），代表"这个图像有多真"的 Wasserstein 评分，而非概率。

### train.py — Wasserstein Loss + Gradient Penalty

```python
import tqdm
import torch

from common import make_noise, save_checkpoint, save_gan_samples


def compute_gradient_penalty(critic, real, fake, device):
    batch_size = real.size(0)
    epsilon = torch.rand(batch_size, 1, 1, 1, device=device)
    interpolates = epsilon * real + (1 - epsilon) * fake
    interpolates.requires_grad_(True)

    d_interpolates = critic(interpolates)

    grads = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(d_interpolates),
        create_graph=True,
    )[0]

    grads = grads.view(batch_size, -1)
    gp = ((grads.norm(2, dim=1) - 1) ** 2).mean()
    return gp


def train_gan(
    generator, critic, dataloader, optimizer_g, optimizer_c,
    fixed_noise, device, latent_dim, epochs, n_critic, lambda_gp,
    sample_every, checkpoint_every, sample_dir, checkpoint_dir,
    viz, window,
):
    for epoch in range(1, epochs + 1):
        last_c_loss = 0.0
        last_g_loss = 0.0
        last_gp = 0.0

        for real_images, _ in tqdm.tqdm(dataloader):
            real_images = real_images.to(device)
            batch_size = real_images.size(0)

            for _ in range(n_critic):
                with torch.no_grad():
                    noise = make_noise(batch_size, latent_dim, device)
                    fake_images = generator(noise)

                c_real = critic(real_images).mean()
                c_fake = critic(fake_images).mean()
                c_loss = -(c_real - c_fake)

                gp = compute_gradient_penalty(critic, real_images, fake_images, device)
                c_loss_total = c_loss + lambda_gp * gp

                optimizer_c.zero_grad()
                c_loss_total.backward()
                optimizer_c.step()

            noise = make_noise(batch_size, latent_dim, device)
            fake_images = generator(noise)
            g_loss = -critic(fake_images).mean()

            optimizer_g.zero_grad()
            g_loss.backward()
            optimizer_g.step()

            last_c_loss = c_loss.item()
            last_g_loss = g_loss.item()
            last_gp = gp.item()

        viz.line(X=[epoch], Y=[[last_c_loss, last_g_loss]], win=window, update="append")

        print(
            f"Epoch [{epoch:03d}/{epochs}] "
            f"C loss: {last_c_loss:.4f} | GP: {last_gp:.4f} | G loss: {last_g_loss:.4f}"
        )

        if epoch % sample_every == 0:
            save_gan_samples(epoch, generator, fixed_noise, sample_dir)

        if epoch % checkpoint_every == 0 or epoch == epochs:
            save_checkpoint(epoch, checkpoint_dir, generator=generator, critic=critic)
```

**逐段解读**：

**Critic Loss**：
```python
c_loss = -(critic(real).mean() - critic(fake).mean())
```
这就是 Wasserstein 距离的蒙特卡洛估计：`max [E[f(real)] - E[f(fake)]]`。注意**没有 BCEWithLogitsLoss**，没有 `real_labels` / `fake_labels`。

**Gradient Penalty**：
```python
def compute_gradient_penalty(critic, real, fake, device):
    epsilon = torch.rand(batch, 1, 1, 1)          # 随机插值系数
    interpolates = epsilon * real + (1 - epsilon) * fake  # 真假图像的凸组合
    interpolates.requires_grad_(True)

    d_interpolates = critic(interpolates)
    grads = torch.autograd.grad(
        outputs=d_interpolates, inputs=interpolates,
        grad_outputs=torch.ones_like(d_interpolates),
        create_graph=True,
    )[0]
    gp = ((grads.norm(2) - 1) ** 2).mean()        # 惩罚偏离 L2=1 的梯度
    return gp
```

`torch.autograd.grad` 计算 critic 对插值点的梯度，`create_graph=True` 保留计算图（因为 GP 项本身需要被优化）。

**训练节奏**：
```python
for _ in range(n_critic):       # Critic 更新 5 次
    ...
# Generator 更新 1 次
```
为什么 5:1？Critic 需要先学好 Wasserstein 距离的估计，G 才能得到准确的梯度方向。

**Generator Loss**：
```python
g_loss = -critic(fake_images).mean()
```
G 的目标：最大化 critic 对假图的评分（让假图看起来"真"）。

### main.py

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import torch.optim as optim

import config
from common import get_dataloader, get_device, make_noise, prepare_dirs, set_seed, get_visualizer
from models import Critic, Generator
from train import train_gan


def main():
    set_seed(config.SEED)
    prepare_dirs(config.DATA_DIR, config.SAMPLE_DIR, config.CHECKPOINT_DIR)

    # ==== Step 1: 数据加载 ====
    device = get_device(config.DEVICE, config.MIN_GPU_MEMORY_MB)
    dataloader = get_dataloader(
        config.IMAGE_SIZE, config.DATA_DIR, config.BATCH_SIZE, config.NUM_WORKERS,
    )

    # ==== Step 2: 模型选择（★ 相对 DCGAN 的差异：Critic 无 BatchNorm） ====
    generator = Generator(
        config.LATENT_DIM, config.CHANNELS, config.IMAGE_SIZE, config.DCGAN_FEATURES
    ).to(device)
    critic = Critic(
        config.CHANNELS, config.IMAGE_SIZE, config.DCGAN_FEATURES
    ).to(device)

    # ==== Step 3: 训练策略（★ Wasserstein loss + Gradient Penalty） ====
    optimizer_g = optim.Adam(generator.parameters(), lr=config.LEARNING_RATE, betas=(config.BETA1, 0.999))
    optimizer_c = optim.Adam(critic.parameters(), lr=config.LEARNING_RATE, betas=(config.BETA1, 0.999))

    viz, window = get_visualizer("WGAN-GP", legend=["critic", "generator"])
    fixed_noise = make_noise(config.NUM_SAMPLE_IMAGES, config.LATENT_DIM, device)

    print(f"Device: {device}")
    print(f"Dataset: {config.DATASET}")
    print(f"Model: WGAN-GP")

    train_gan(
        generator=generator, critic=critic, dataloader=dataloader,
        optimizer_g=optimizer_g, optimizer_c=optimizer_c,
        fixed_noise=fixed_noise, device=device, latent_dim=config.LATENT_DIM,
        epochs=config.EPOCHS, n_critic=config.N_CRITIC, lambda_gp=config.LAMBDA_GP,
        sample_every=config.SAMPLE_EVERY, checkpoint_every=config.CHECKPOINT_EVERY,
        sample_dir=config.SAMPLE_DIR, checkpoint_dir=config.CHECKPOINT_DIR,
        viz=viz, window=window,
    )


if __name__ == "__main__":
    main()
```

注意 `viz` 的 legend 参数：`["critic", "generator"]`（不再叫 discriminator）。

### generate.py

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import torch
from torchvision.utils import save_image

import config
from common import get_device, make_noise, prepare_dirs, set_seed
from models import Generator


def main():
    set_seed(config.SEED)
    prepare_dirs(config.DATA_DIR, config.SAMPLE_DIR, config.CHECKPOINT_DIR)

    # ==== Step 1: 加载模型 ====
    device = get_device(config.DEVICE, config.MIN_GPU_MEMORY_MB)
    checkpoint_path = config.CHECKPOINT_DIR / "latest.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError("No checkpoint found. Run `python main.py` first.")

    generator = Generator(
        config.LATENT_DIM, config.CHANNELS, config.IMAGE_SIZE, config.DCGAN_FEATURES
    ).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    generator.load_state_dict(checkpoint["generator"])
    generator.eval()

    # ==== Step 2: 准备输入 ====
    noise = make_noise(config.NUM_SAMPLE_IMAGES, config.LATENT_DIM, device)

    # ==== Step 3: 生成并保存 ====
    with torch.no_grad():
        images = generator(noise)
        images = (images + 1) / 2
        save_image(images, config.SAMPLE_DIR / "generated.png", nrow=8)

    print(f"Saved generated images to {config.SAMPLE_DIR / 'generated.png'}")


if __name__ == "__main__":
    main()
```

推理与 DCGAN 完全一致——WGAN-GP 只影响训练，不影响推理。

## 运行

```bash
cd code/stage04_wgan_gp
python main.py
```

训练时间约为 DCGAN 的 5 倍（N_CRITIC=5）。

## 与 DCGAN 对比

| | Stage 02 (DCGAN) | Stage 04 (WGAN-GP) |
|---|---|---|
| G 架构 | ConvTranspose2d | **完全相同** |
| D/C 架构 | Conv2d + BatchNorm | Conv2d **无** BatchNorm |
| D/C 输出 | logit（配合 BCE） | 实数分数（Wasserstein） |
| Loss | BCEWithLogitsLoss | Critic: `-(E[f(real)] - E[f(fake)])`; G: `-E[f(fake)]` |
| 约束 | 无 | Gradient Penalty (`λ=10`) |
| 更新比 | 1:1 | 5:1 (C:G) |
| 训练时长 | ~3 min | ~28 min |

model_comparison.md 证实：C_loss 从 5.0 持续收敛到 0.7（Wasserstein 距离逐步缩小），而 DCGAN 的 D_loss 暴跌到 0.28 后不动。

## 学习检查点

1. 为什么 Wasserstein 距离在分布不重叠时仍然能提供梯度，而 JS 散度不行？
2. 为什么 WGAN-GP 要移除 BatchNorm？
3. `torch.autograd.grad` 在这里计算的是什么？`create_graph=True` 为什么必要？
4. `N_CRITIC=5` 换成 `N_CRITIC=1` 会怎样？

---

上一篇：[Stage 03 — cGAN](../stage03_cgan/cgan.md)
下一篇：[Stage 05 — VAE](../stage05_vae/vae.md)

# Stage 05: VAE — 一条完全不同的路

> **前提**：建议先理解 GAN 系列（Stage 01-04），再来看 VAE。两者是生成模型的两大范式。
>
> **本阶段是新的基础路线**，所有文件都不同。

## 前言：GAN 之外的另一种可能

GAN 的思路是"对抗博弈"：Generator 和 Discriminator 互相竞争。

VAE（Variational Autoencoder，变分自编码器）走了完全不同的路：**学习数据的概率分布**。

```
图像 x → Encoder → 隐变量 z（分布的均值 μ + 方差 σ²）
          ↓ 重参数化采样
       z ~ N(μ, σ²) → Decoder → 重建 x̂
```

训练完成后，直接从 N(0,1) 采样 z 送入 Decoder，就能生成新图像。

与 GAN 的本质区别：
- GAN：单向（噪声→图像），隐空间无结构
- VAE：双向（图像↔隐变量），隐空间有语义结构——两个数字的 z 之间线性插值，Decoder 会输出平滑过渡

## 理论：ELBO

VAE 优化的是 **ELBO（Evidence Lower Bound，证据下界）**：

$$ \mathcal{L} = \underbrace{\mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)]}_{\text{MSE 重建}} - \underbrace{D_{KL}(q_\phi(z|x) || p(z))}_{\text{KL 散度正则}} $$

- **重建项**：Decoder 从隐变量 z 还原原图的能力。实现为 MSE loss。
- **KL 正则项**：Encoder 输出的后验 q(z|x) 离先验 N(0,1) 有多远。这迫使不同数字的隐编码平滑分布在 N(0,1) 周围。

### 重参数化技巧（Reparameterization Trick）

采样 `z ~ N(μ, σ²)` 不可导——梯度无法流过随机操作。解决方案：

```python
z = μ + σ ⊙ ε    # ε ~ N(0, 1) — 外部噪声，视为常量
```

梯度可以流经 μ 和 σ，ε 被当作不参与优化的常量。

## 实现：全链路不同

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

# ========== Model Architecture (VAE 特有) ==========
LATENT_DIM = 20

# ========== Training (VAE 使用不同优化器参数) ==========
BATCH_SIZE = 128
EPOCHS = 50
LEARNING_RATE = 0.001
BETA1 = 0.9
SAMPLE_EVERY = 1
CHECKPOINT_EVERY = 10

# ========== Inference ==========
NUM_SAMPLE_IMAGES = 64

# ========== Runtime ==========
DEVICE = "auto"
SEED = 42
MIN_GPU_MEMORY_MB = 4096
```

与 GAN 系列的配置差异：
- `LATENT_DIM = 20`：VAE 用更小的隐空间（GAN 是 100）。隐空间越小，VAE 被迫学得越紧凑
- `LEARNING_RATE = 0.001, BETA1 = 0.9`：接近标准分类任务的 Adam 参数（GAN 用 `lr=0.0002, beta1=0.5` 是特例）

### models.py — Encoder + Decoder + VAE

```python
import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, image_dim, latent_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(image_dim, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
        )
        self.mu_head = nn.Linear(256, latent_dim)
        self.logvar_head = nn.Linear(256, latent_dim)

    def forward(self, x):
        h = self.net(x)
        return self.mu_head(h), self.logvar_head(h)


class Decoder(nn.Module):
    def __init__(self, latent_dim, image_dim, channels, image_size):
        super().__init__()
        self.channels = channels
        self.image_size = image_size

        self.net = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, image_dim),
            nn.Tanh(),
        )

    def forward(self, z):
        x = self.net(z)
        return x.view(-1, self.channels, self.image_size, self.image_size)


class VAE(nn.Module):
    def __init__(self, latent_dim, channels, image_size):
        super().__init__()
        image_dim = channels * image_size * image_size
        self.encoder = Encoder(image_dim, latent_dim)
        self.decoder = Decoder(latent_dim, image_dim, channels, image_size)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar
```

三部分解析：

**Encoder**（图 → 分布参数）：
```
图像 (1,28,28) → Flatten(784) → Linear(512) → ReLU → Linear(256) → ReLU
    → mu_head: Linear(20)       ← 隐变量均值 μ
    → logvar_head: Linear(20)    ← 隐变量 log 方差 log σ²
```
输出的是分布的**参数**（μ 和 log σ²），不是直接的隐变量。用 `log σ²` 而非 σ² 是为了保证数值稳定（指数后恒正）。

**Decoder**（隐变量 → 图）：
```
z (20,) → Linear(256) → ReLU → Linear(512) → ReLU → Linear(784) → Tanh
```
与 Stage 01 Generator 结构相同，但输入是 20 维而非 100 维。

**VAE 容器**：
```python
def forward(self, x):
    mu, logvar = self.encode(x)               # 编码为分布
    z = self.reparameterize(mu, logvar)        # 重参数化采样
    x_recon = self.decode(z)                   # 解码重建
    return x_recon, mu, logvar
```

### train.py — ELBO 训练

```python
import tqdm
import torch
import torch.nn.functional as F
import visdom

from common import save_checkpoint


def get_vae_visualizer(environment="VAE"):
    viz = visdom.Visdom(env=environment)
    window = viz.line(
        Y=torch.full((1, 2), float("nan")),
        X=[0],
        win="loss",
        opts={
            "showlegend": True,
            "title": "VAE Loss",
            "xlabel": "epoch",
            "ylabel": "loss",
            "legend": ["reconstruction", "kl_divergence"],
        },
    )
    return viz, window


def save_vae_samples(epoch, vae, fixed_noise, sample_dir):
    from torchvision.utils import save_image
    vae.eval()
    with torch.no_grad():
        fake_images = vae.decode(fixed_noise)
        fake_images = (fake_images + 1) / 2
        save_image(fake_images, sample_dir / f"epoch_{epoch:03d}.png", nrow=8)
    vae.train()


def train_vae(
    vae, dataloader, optimizer, fixed_noise, device,
    epochs, sample_every, checkpoint_every, sample_dir, checkpoint_dir,
    viz, window,
):
    for epoch in range(1, epochs + 1):
        last_recon_loss = 0.0
        last_kl_loss = 0.0

        for real_images, _ in tqdm.tqdm(dataloader):
            real_images = real_images.to(device)
            batch_size = real_images.size(0)

            x_recon, mu, logvar = vae(real_images)

            # 重建 loss：MSE，按样本求和后取平均
            recon_loss = F.mse_loss(
                x_recon.view(batch_size, -1),
                real_images.view(batch_size, -1),
                reduction="sum"
            ) / batch_size

            # KL 散度：N(μ,σ²) 与 N(0,1) 之间的解析解
            kl_loss = -0.5 * torch.sum(
                1 + logvar - mu.pow(2) - logvar.exp()
            ) / batch_size

            loss = recon_loss + kl_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            last_recon_loss = recon_loss.item()
            last_kl_loss = kl_loss.item()

        viz.line(X=[epoch], Y=[[last_recon_loss, last_kl_loss]], win=window, update="append")

        print(
            f"Epoch [{epoch:03d}/{epochs}] "
            f"Recon: {last_recon_loss:.4f} | KL: {last_kl_loss:.4f}"
        )

        if epoch % sample_every == 0:
            save_vae_samples(epoch, vae, fixed_noise, sample_dir)

        if epoch % checkpoint_every == 0 or epoch == epochs:
            save_checkpoint(epoch, checkpoint_dir, vae=vae, optimizer=optimizer)
```

**关键差异**（对比 GAN 的 train.py）：

1. **只有一个优化器**。VAE 不分 Generator 和 Discriminator——Encoder 和 Decoder 一起优化同一个 ELBO 目标。
2. **loss 由两项相加**：`recon_loss`（重建质量）+ `kl_loss`（隐空间规整度），不存在对抗博弈。
3. **KL 散度有解析解**：`KL(N(μ,σ²) || N(0,1)) = -0.5 × Σ(1 + log(σ²) - μ² - σ²)`，直接从 μ 和 logvar 计算。
4. **生成采样只用 Decoder**：`vae.decode(fixed_noise)`，Encoder 只在训练时使用。
5. **checkpoint 保存整个 VAE 和 optimizer**：方便继续训练。

### main.py

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import torch.optim as optim

import config
from common import get_dataloader, get_device, make_noise, prepare_dirs, set_seed
from models import VAE
from train import train_vae, get_vae_visualizer


def main():
    set_seed(config.SEED)
    prepare_dirs(config.DATA_DIR, config.SAMPLE_DIR, config.CHECKPOINT_DIR)

    # ==== Step 1: 数据加载 ====
    device = get_device(config.DEVICE, config.MIN_GPU_MEMORY_MB)
    dataloader = get_dataloader(
        config.IMAGE_SIZE, config.DATA_DIR, config.BATCH_SIZE, config.NUM_WORKERS,
    )

    # ==== Step 2: 模型选择（★ VAE：Encoder + Decoder，非对抗式） ====
    vae = VAE(config.LATENT_DIM, config.CHANNELS, config.IMAGE_SIZE).to(device)

    # ==== Step 3: 训练策略（★ ELBO = MSE(重建) + KL(正则化)） ====
    optimizer = optim.Adam(vae.parameters(), lr=config.LEARNING_RATE, betas=(config.BETA1, 0.999))

    viz, window = get_vae_visualizer("VAE")
    fixed_noise = make_noise(config.NUM_SAMPLE_IMAGES, config.LATENT_DIM, device)

    print(f"Device: {device}")
    print(f"Dataset: {config.DATASET}")
    print(f"Latent dim: {config.LATENT_DIM}")

    train_vae(
        vae=vae, dataloader=dataloader, optimizer=optimizer,
        fixed_noise=fixed_noise, device=device,
        epochs=config.EPOCHS, sample_every=config.SAMPLE_EVERY,
        checkpoint_every=config.CHECKPOINT_EVERY,
        sample_dir=config.SAMPLE_DIR, checkpoint_dir=config.CHECKPOINT_DIR,
        viz=viz, window=window,
    )


if __name__ == "__main__":
    main()
```

### generate.py

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import torch
from torchvision.utils import save_image

import config
from common import get_device, make_noise, prepare_dirs, set_seed
from models import VAE


def main():
    set_seed(config.SEED)
    prepare_dirs(config.DATA_DIR, config.SAMPLE_DIR, config.CHECKPOINT_DIR)

    # ==== Step 1: 加载模型 ====
    device = get_device(config.DEVICE, config.MIN_GPU_MEMORY_MB)
    checkpoint_path = config.CHECKPOINT_DIR / "latest.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError("No checkpoint found. Run `python main.py` first.")

    vae = VAE(config.LATENT_DIM, config.CHANNELS, config.IMAGE_SIZE).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    vae.load_state_dict(checkpoint["vae"])
    vae.eval()

    # ==== Step 2: 准备输入（VAE 从 N(0,1) 采样隐变量解码） ====
    noise = make_noise(config.NUM_SAMPLE_IMAGES, config.LATENT_DIM, device)

    # ==== Step 3: 生成并保存 ====
    with torch.no_grad():
        images = vae.decode(noise)
        images = (images + 1) / 2
        save_image(images, config.SAMPLE_DIR / "generated.png", nrow=8)

    print(f"Saved generated images to {config.SAMPLE_DIR / 'generated.png'}")


if __name__ == "__main__":
    main()
```

与 GAN generate.py 的关键差异：调用 `vae.decode(noise)` 而非 `generator(noise)`。VAE 的生成不需要经过 Encoder——直接从 N(0,1) 采样送入 Decoder。

## 运行

```bash
cd code/stage05_vae
python main.py
```

## 与 GAN 系列对比

| | GAN（Stage 01-04） | VAE（Stage 05） |
|---|---|---|
| 范式 | 对抗博弈 | 概率建模（变分推断） |
| 网络 | Generator + Discriminator | Encoder + Decoder |
| 隐空间维度 | 100 | 20 |
| Loss | BCE / Wasserstein | MSE(重建) + KL(正则化) |
| 优化器数量 | 2 个（各管各） | 1 个（Encoder+Decoder 一起） |
| 训练稳定度 | 依赖技巧 | 非常稳定 |
| 隐空间结构 | 无 | 连续、可插值 |
| 生成质量 | 清晰 | 偏模糊（MSE 的平均效应） |
| 训练时长 | ~3 min（DCGAN） | ~3 min |

VAE 牺牲了锐度，换来了隐空间的结构化——这是 GAN 天然不具备的特性。

## 学习检查点

1. ELBO 的两个项分别的物理含义是什么？
2. 重参数化技巧解决了什么问题？如果没有它，梯度能流到 μ 和 σ 吗？
3. 为什么 `latent_dim=20` 而非 100？增大 latent_dim 会怎样？
4. VAE 生成图像偏模糊的根本原因是什么？
5. VAE 的 checkpoint 为什么要保存 optimizer？GAN 的 checkpoint 需要吗？

---

上一篇：[Stage 04 — WGAN-GP](../stage04_wgan_gp/wgan_gp.md)

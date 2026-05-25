# Stage 02: DCGAN — 用卷积升级 GAN

> **前提**：请先完成 [Stage 01](../stage01_gan_basics/gan_basics.md)，理解 MLP GAN 的完整流程。
>
> **本阶段只改了 models.py**，其余全部复用 `common/`。

## 回顾：MLP GAN 的问题

Stage 01 的 Generator 和 Discriminator 都是全连接层。图像 `28×28=784` 像素被展平成向量处理：

1. **丢失空间结构**：相邻像素的 2D 关系被忽略
2. **参数量爆炸**：`Linear(784, 512)` 就有 40 万参数，扩展到 64×64 彩色图像直接不可行

卷积层天然保留空间结构，且通过参数共享（同一个 kernel 在整张图上滑动）大幅降低参数量。

## 理论

DCGAN（Deep Convolutional GAN）的核心贡献是一套稳定的卷积 GAN 架构设计准则：

1. 用 **`ConvTranspose2d`（转置卷积/反卷积）** 做上采样，不用全连接层来放大特征
2. 用 **`Conv2d`（步长卷积）** 做下采样，不用池化层（MaxPool 会丢失信息）
3. G 和 D 都用 **BatchNorm** 稳定训练（D 中 BN 是老版 GAN 不推荐的，但 DCGAN 实验证明放在卷积 D 中可以工作）
4. G 中间层用 **ReLU**（输出层用 Tanh），D 用 **LeakyReLU**
5. **去掉所有全连接隐藏层**

## 实现：只有 models.py 变了

本目录只有 4 个文件（config / models / main / generate）。**训练逻辑与 stage01 完全相同**，直接导入 `common/gan.py` 中的 `train_bce_gan`。

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

# ========== Model Architecture (DCGAN 特有) ==========
LATENT_DIM = 100
DCGAN_FEATURES = 64

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

唯一新增：`DCGAN_FEATURES = 64`，控制第一层卷积的通道数（模型容量）。

### models.py — ★ 唯一变更

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
        self.init_size = image_size // 4  # 28 -> 7

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


class Discriminator(nn.Module):
    def __init__(self, channels, image_size, features=64):
        super().__init__()
        self.channels = channels
        self.image_size = image_size
        self.features = features

        self.net = nn.Sequential(
            nn.Conv2d(channels, features, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(features, features * 2, 4, 2, 1),
            nn.BatchNorm2d(features * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(features * 2, features * 4, 4, 2, 1),
            nn.BatchNorm2d(features * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(features * 4, 1, 3, 1, 0),
            nn.Flatten(),
        )

    def forward(self, images):
        return self.net(images)
```

**Generator 的数据流变化**（对比 stage01 的 `Linear × 4 → 图像`）：

```
噪声 z (100,) 
    → Linear(100, 256×7×7)          ← 仍然用一层 Linear 做初始投射
    → reshape (256, 7, 7)            ← 变成 4D 特征图 [B, C, H, W]
    → ConvTranspose2d(256→128, k=4, s=2, p=1)   7→14  (上采样×2)
    → BN + ReLU
    → ConvTranspose2d(128→64,  k=4, s=2, p=1)   14→28 (上采样×2)
    → BN + ReLU
    → ConvTranspose2d(64→1,   k=3, s=1, p=1)   28→28 (保持尺寸)
    → Tanh
```

转置卷积的尺寸公式：`output = stride × (input - 1) + kernel - 2 × padding`
- 例：`ConvTranspose2d(k=4, s=2, p=1)` → `2×(7-1) + 4 - 2 = 14`，恰好放大 2 倍

**Discriminator 的数据流变化**（对比 stage01 的 `Flatten → Linear × 3 → logit`）：

```
图像 (1, 28, 28)
    → Conv2d(1→64, k=4, s=2, p=1)   28→14 (下采样×1/2)
    → LeakyReLU
    → Conv2d(64→128, k=4, s=2, p=1)  14→7
    → BN + LeakyReLU
    → Conv2d(128→256, k=4, s=2, p=1)  7→3
    → BN + LeakyReLU
    → Conv2d(256→1, k=3, s=1, p=0)    3→1
    → Flatten → logit
```

D 的最后一层直接卷积到 `1×1` 的空间尺寸，然后展平为单个 logit。不使用全连接层。

### main.py — 三步组装（与 stage01 结构相同）

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import torch.nn as nn
import torch.optim as optim

import config
from common import get_dataloader, get_device, make_noise, prepare_dirs, set_seed, get_visualizer
from common.gan import train_bce_gan
from models import Discriminator, Generator


def main():
    set_seed(config.SEED)
    prepare_dirs(config.DATA_DIR, config.SAMPLE_DIR, config.CHECKPOINT_DIR)

    # ==== Step 1: 数据加载 ====
    device = get_device(config.DEVICE, config.MIN_GPU_MEMORY_MB)
    dataloader = get_dataloader(
        config.IMAGE_SIZE, config.DATA_DIR, config.BATCH_SIZE, config.NUM_WORKERS,
    )

    # ==== Step 2: 模型选择（★ 相对 stage01 的差异：Conv 替代 MLP） ====
    generator = Generator(
        config.LATENT_DIM, config.CHANNELS, config.IMAGE_SIZE, config.DCGAN_FEATURES
    ).to(device)
    discriminator = Discriminator(
        config.CHANNELS, config.IMAGE_SIZE, config.DCGAN_FEATURES
    ).to(device)

    # ==== Step 3: 训练策略 ====
    criterion = nn.BCEWithLogitsLoss()
    optimizer_g = optim.Adam(generator.parameters(), lr=config.LEARNING_RATE, betas=(config.BETA1, 0.999))
    optimizer_d = optim.Adam(discriminator.parameters(), lr=config.LEARNING_RATE, betas=(config.BETA1, 0.999))

    viz, window = get_visualizer("DCGAN")
    fixed_noise = make_noise(config.NUM_SAMPLE_IMAGES, config.LATENT_DIM, device)

    print(f"Device: {device}")
    print(f"Dataset: {config.DATASET}")

    train_bce_gan(
        generator=generator,
        discriminator=discriminator,
        dataloader=dataloader,
        criterion=criterion,
        optimizer_g=optimizer_g,
        optimizer_d=optimizer_d,
        fixed_noise=fixed_noise,
        device=device,
        latent_dim=config.LATENT_DIM,
        epochs=config.EPOCHS,
        sample_every=config.SAMPLE_EVERY,
        checkpoint_every=config.CHECKPOINT_EVERY,
        sample_dir=config.SAMPLE_DIR,
        checkpoint_dir=config.CHECKPOINT_DIR,
        viz=viz,
        window=window,
    )


if __name__ == "__main__":
    main()
```

注意 `train_bce_gan` 来自 `common.gan`——**没有 train.py 放在本目录**。训练逻辑（BCE loss, 1:1 交替更新）与 stage01 完全一致。

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

训练循环 `train_bce_gan`、`save_checkpoint`、`save_gan_samples` 全部复用 `common/`，与 stage01 一致，不再赘述。感兴趣可查看 `common/gan.py` 和 `common/training.py`。

## 运行

```bash
cd code/stage02_dcgan
python main.py
```

## 与 Stage 01 对比

| | Stage 01 (MLP GAN) | Stage 02 (DCGAN) |
|---|---|---|
| G 架构 | Linear × 4 | Linear(投射) + ConvTranspose2d × 3 |
| D 架构 | Flatten + Linear × 3 | Conv2d × 4 |
| 空间结构 | 丢失（展平为向量） | 保留（2D 卷积核） |
| BatchNorm | 只在 G 中 | G 和 D 都使用 |
| G 激活 | LeakyReLU | ReLU |
| 训练逻辑 | BCE, 1:1 | **完全相同**（导入 common/gan.py） |
| train.py | 有 | **无**（复用 common/） |

## 学习检查点

1. `ConvTranspose2d(kernel=4, stride=2, padding=1)` 为什么恰好让特征图放大 2 倍？
2. D 中的 `Conv2d(kernel=4, stride=2, padding=1)` 为什么让特征图缩小一半？
3. DCGAN 论文规定了哪些架构准则？
4. `features=64` 表示什么？增大这个值会对模型产生什么影响？

---

上一篇：[Stage 01 — MLP GAN](../stage01_gan_basics/gan_basics.md)
下一篇：[Stage 03 — cGAN](../stage03_cgan/cgan.md)

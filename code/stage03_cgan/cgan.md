# Stage 03: cGAN — 告诉模型"我要生成数字 3"

> **前提**：请先完成 [Stage 01](../stage01_gan_basics/gan_basics.md)。
>
> **本阶段只改了 models.py**（与 main.py 多传一行 `fixed_labels`），其余全部复用 `common/`。

## 回顾：无条件 GAN 的局限

Stage 01/02 的 GAN 只能"随机生成一个数字"——你无法指定生成哪个数字。

```
噪声 z → Generator → "某个数字"  ← 不可控
```

cGAN（Conditional GAN）解决这个问题：**把条件信息同时喂给 G 和 D**。

## 理论

原版 GAN 的 minimax game：

$$ \min_G \max_D \mathbb{E}_x[\log D(x)] + \mathbb{E}_z[\log(1 - D(G(z)))] $$

cGAN 把标签 y 加进去：

$$ \min_G \max_D \mathbb{E}_{x,y}[\log D(x, y)] + \mathbb{E}_{z,y}[\log(1 - D(G(z, y), y))] $$

变化就一个字：**y**。G 收到 `(噪声, 标签)` → "我需要生成 y 类的图像"。D 收到 `(图像, 标签)` → "这张图是否真的属于 y 类"。这让 G 和 D 在标签的约束下博弈。

## 实现：只有 models.py 变了

本目录只有 4 个文件。所有变化在 `models.py`，训练逻辑复用 `common/gan.py` 的 `train_bce_gan`（通过 `fixed_labels` 参数启用条件分支）。

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

# ========== Model Architecture (cGAN 特有) ==========
LATENT_DIM = 100
NUM_CLASSES = 10

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

新增 `NUM_CLASSES = 10`（MNIST 有 10 个数字类别）。

### models.py — ★ 唯一变更

```python
import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, latent_dim, num_classes, channels, image_size):
        super().__init__()
        self.label_embedding = nn.Embedding(num_classes, num_classes)
        input_dim = latent_dim + num_classes
        image_dim = channels * image_size * image_size

        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, image_dim),
            nn.Tanh(),
        )

        self.latent_dim = latent_dim
        self.channels = channels
        self.image_size = image_size

    def forward(self, z, labels):
        label_embed = self.label_embedding(labels)
        x = torch.cat([z, label_embed], dim=1)
        images = self.net(x)
        return images.view(-1, self.channels, self.image_size, self.image_size)


class Discriminator(nn.Module):
    def __init__(self, num_classes, channels, image_size):
        super().__init__()
        self.label_embedding = nn.Embedding(num_classes, num_classes)
        image_dim = channels * image_size * image_size
        input_dim = image_dim + num_classes

        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
        )

    def forward(self, images, labels):
        label_embed = self.label_embedding(labels)
        x = torch.cat([images.view(images.size(0), -1), label_embed], dim=1)
        return self.net(x)
```

**相对 stage01 的两个改动**：

**改动一：Generator 接收标签**

Stage 01 的 forward：
```python
def forward(self, z):
    return self.net(z)           # 只接收噪声
```

Stage 03 的 forward：
```python
def forward(self, z, labels):
    label_embed = self.label_embedding(labels)    # (batch, 10)
    x = torch.cat([z, label_embed], dim=1)         # (batch, 110)
    return self.net(x)
```

`nn.Embedding(10, 10)` 是一个可学习的查找表——把标签 0-9 映射为 10 维稠密向量，拼接到噪声后面。输入维度从 100 变为 110。

**改动二：Discriminator 也接收标签**

```python
def forward(self, images, labels):
    x = torch.cat([images.view(batch, -1), label_embed], dim=1)
    return self.net(x)
```

D 看到图像+标签的组合，才能判断"图像是否匹配标签"。这迫使 G 生成与标签一致的图像。

**为什么不用 one-hot？** `nn.Embedding` 比 one-hot 多一个优势：嵌入向量是可学习的。模型可以学到"3 和 8 的嵌入很相似"（因为它们长得像），而 one-hot 认为所有类别等距。

### main.py — 多传一行 `fixed_labels`

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import torch
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

    # ==== Step 2: 模型选择（★ 相对 stage01 的差异：G/D 加入 Label Embedding） ====
    generator = Generator(
        config.LATENT_DIM, config.NUM_CLASSES, config.CHANNELS, config.IMAGE_SIZE
    ).to(device)
    discriminator = Discriminator(
        config.NUM_CLASSES, config.CHANNELS, config.IMAGE_SIZE
    ).to(device)

    # ==== Step 3: 训练策略（传入 fixed_labels 启用条件生成） ====
    criterion = nn.BCEWithLogitsLoss()
    optimizer_g = optim.Adam(generator.parameters(), lr=config.LEARNING_RATE, betas=(config.BETA1, 0.999))
    optimizer_d = optim.Adam(discriminator.parameters(), lr=config.LEARNING_RATE, betas=(config.BETA1, 0.999))

    viz, window = get_visualizer("cGAN")
    fixed_noise = make_noise(config.NUM_SAMPLE_IMAGES, config.LATENT_DIM, device)
    fixed_labels = torch.arange(8, device=device).repeat(8)

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
        fixed_labels=fixed_labels,
    )


if __name__ == "__main__":
    main()
```

与 stage02 的 main.py 相比，只有两处新增：
1. `fixed_labels = torch.arange(8, device=device).repeat(8)` — 为可视化生成 64 张固定标签图片（标签 0-7，各 8 张）
2. `train_bce_gan(... fixed_labels=fixed_labels)` — 多传一行，触发 `common/gan.py` 中的条件分支

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
        config.LATENT_DIM, config.NUM_CLASSES, config.CHANNELS, config.IMAGE_SIZE
    ).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    generator.load_state_dict(checkpoint["generator"])
    generator.eval()

    # ==== Step 2: 准备输入（条件生成需指定标签） ====
    noise = make_noise(config.NUM_SAMPLE_IMAGES, config.LATENT_DIM, device)
    labels = torch.arange(8, device=device).repeat(8)

    # ==== Step 3: 生成并保存 ====
    with torch.no_grad():
        images = generator(noise, labels)
        images = (images + 1) / 2
        save_image(images, config.SAMPLE_DIR / "generated.png", nrow=8)

    print(f"Saved generated images to {config.SAMPLE_DIR / 'generated.png'}")


if __name__ == "__main__":
    main()
```

与 stage02 generate.py 的差别：`generator(noise, labels)` 多传了标签参数，且用 `torch.arange(8).repeat(8)` 产生 0-7 的标签序列。

### 复用的 common/gan.py

`train_bce_gan` 的条件分支逻辑（复用的训练代码见 `common/gan.py`）：
- 当 `fixed_labels is not None` 时：`generator(noise, batch_labels)`、`discriminator(images, batch_labels)`
- 当 `fixed_labels is None` 时：`generator(noise)`、`discriminator(images)`

同一个函数同时服务于 stage02（无条件）和 stage03（有条件）。

## 运行

```bash
cd code/stage03_cgan
python main.py
```

## 与 Stage 01 对比

| | Stage 01 (GAN) | Stage 03 (cGAN) |
|---|---|---|
| G 输入 | 噪声 z (100) | 噪声 z (100) + 标签嵌入 (10) = (110) |
| D 输入 | 图像 (784) | 图像 (784) + 标签嵌入 (10) = (794) |
| 标签嵌入 | 无 | `nn.Embedding(10, 10)` |
| 生成控制 | 不可指定数字 | 可指定数字 |
| 训练循环 | BCE, 1:1 | BCE, 1:1（**完全相同**，复用 common/gan.py） |

## 学习检查点

1. cGAN 的条件信息为何同时喂给 G 和 D？只给 G 不给 D 会怎样？
2. `nn.Embedding` 和 one-hot 编码的优劣？
3. `torch.arange(8).repeat(8)` 产生的标签数组长什么样？对应生成的图片布局是怎样的？

---

上一篇：[Stage 02 — DCGAN](../stage02_dcgan/dcgan.md)
下一篇：[Stage 04 — WGAN-GP](../stage04_wgan_gp/wgan_gp.md)

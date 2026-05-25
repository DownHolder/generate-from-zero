# Stage 01: MLP GAN — 从这里开始

> 阅读本教程前，确保已安装依赖：`pip install torch torchvision tqdm visdom`

## 前言

这是整个学习路线的起点。在动手写代码之前，先花 30 秒理解我们要做什么。

**GAN（Generative Adversarial Network，生成对抗网络）** 的核心思想：让两个网络互相博弈。

- **Generator（生成器，G）**：输入随机噪声向量，输出"假"图像。目标是骗过 Discriminator。
- **Discriminator（判别器，D）**：输入一张图像，输出它是"真"还是"假"的分数。目标是识破 Generator。

这个博弈过程就是一个 minimax game：

$$ \min_G \max_D \mathbb{E}_{x \sim p_{data}}[\log D(x)] + \mathbb{E}_{z \sim p_z}[\log(1 - D(G(z)))] $$

本章使用最简单的 **MLP（全连接层）** 构建 G 和 D，在 MNIST 手写数字上训练。目标是跑通完整的生成式模型工程流程。

---

## 1. 项目文件总览

本目录有 7 个 `.py` 文件，构成一个最小但完整的深度学习工程骨架：

```text
stage01_gan_basics/
├── config.py      # 所有超参数和路径
├── dataset.py     # 数据加载（MNIST → DataLoader）
├── models.py      # Generator 和 Discriminator 的网络定义
├── train.py       # 训练循环的核心逻辑
├── utils.py       # 通用工具（GPU 选择、随机种子、可视化）
├── main.py        # 总控入口：把所有模块"组装"起来
└── generate.py    # 推理入口：加载训练好的模型生成图片
```

**阅读顺序**：config → dataset → models → utils → train → main → generate

---

## 2. 逐文件精读

### 2.1 config.py

把所有"可能会改的数值"集中在一个文件里，调参时不用翻来翻去。

```python
from pathlib import Path

# ========== Paths ==========
ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR.parent / "data"
SAMPLE_DIR = ROOT_DIR / "samples"
CHECKPOINT_DIR = ROOT_DIR / "checkpoints"

# ========== Dataset ==========
DATASET = "mnist"
IMAGE_SIZE = 28
CHANNELS = 1
NUM_WORKERS = 8

# ========== Model Architecture ==========
LATENT_DIM = 100

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

# ========== GPU Selection ==========
MIN_GPU_MEMORY_MB = 4096
```

要点：
- `LATENT_DIM = 100`：噪声向量维度，越大 G 的输入信息越丰富
- `LEARNING_RATE = 0.0002, BETA1 = 0.5`：GAN 社区的经验参数，和普通分类任务（`lr=0.001, beta1=0.9`）不同
- `Normalize((0.5,), (0.5,))` 把像素归一到 `[-1, 1]`，配合 G 最后的 `Tanh`

### 2.2 dataset.py — 数据加载

```python
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_dataloader(image_size, data_dir, batch_size, num_workers):
    transform = transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )

    dataset = datasets.MNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=transform,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    return dataloader
```

三个关键操作：
1. `Resize` — 确保图像尺寸一致
2. `ToTensor` — PIL Image → `[0, 1]` 的 tensor
3. `Normalize((0.5,), (0.5,))` — 从 `[0, 1]` 映射到 `[-1, 1]`，公式：`(x - 0.5) / 0.5`

`drop_last=True` 丢弃最后一个不完整 batch，避免 BatchNorm 在单个样本上出错。

### 2.3 models.py — 网络结构

```python
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, latent_dim, channels, image_size):
        super().__init__()
        image_dim = channels * image_size * image_size

        self.net = nn.Sequential(
            nn.Linear(latent_dim, 256),
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

    def forward(self, z):
        images = self.net(z)
        return images.view(-1, self.channels, self.image_size, self.image_size)


class Discriminator(nn.Module):
    def __init__(self, channels, image_size):
        super().__init__()
        image_dim = channels * image_size * image_size

        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(image_dim, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
        )

    def forward(self, images):
        return self.net(images)
```

**Generator 数据流**：
```
噪声 z (100,) → Linear(256) → LeakyReLU → Linear(512) → BN → LeakyReLU
    → Linear(1024) → BN → LeakyReLU → Linear(784) → Tanh → reshape (1,28,28)
```

**Discriminator 数据流**：
```
图像 (1,28,28) → Flatten → Linear(512) → LeakyReLU → Linear(256) → LeakyReLU → Linear(1) → logit
```

设计要点：
- **Generator 用 `Tanh` 输出 `[-1, 1]`**：与数据归一化范围一致
- **Discriminator 最后一层没有 Sigmoid**：配合 `BCEWithLogitsLoss` 使用，数值更稳定。`BCEWithLogitsLoss = Sigmoid + BCELoss` 的融合算符，比分开写更稳定
- **`BatchNorm1d` 在 G 的中间层**：加速收敛，稳定训练。D 中不加（这是早期 GAN 的常见做法）
- **`LeakyReLU(0.2)`**：负半轴保留 0.2 倍斜率，防止神经元"死亡"

### 2.4 utils.py — 工具箱

```python
import os
import random
import subprocess
import visdom
import torch


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _detect_free_gpu(min_free_memory_mb=1024):
    gpu_info = []
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.free,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
        for line in result.stdout.strip().split("\n"):
            line = line.strip()
            if not line:
                continue
            parts = [p.strip() for p in line.split(", ")]
            if len(parts) >= 4:
                try:
                    idx = int(parts[0])
                    mem_free = int(parts[1])
                    mem_total = int(parts[2])
                    util_gpu = int(parts[3])
                    gpu_info.append((idx, mem_free, mem_total, util_gpu))
                except ValueError:
                    continue
    except (subprocess.SubprocessError, FileNotFoundError, ValueError):
        return None

    if not gpu_info:
        return None

    eligible = [(idx, free, total, util) for idx, free, total, util in gpu_info if free >= min_free_memory_mb]

    if eligible:
        selected = max(eligible, key=lambda x: x[1])
    else:
        selected = max(gpu_info, key=lambda x: x[1])

    idx, mem_free, mem_total, util_gpu = selected
    print(
        f"Selected GPU {idx}: "
        f"{mem_free}MiB free / {mem_total}MiB total, "
        f"utilization {util_gpu}%"
    )
    return idx


def get_device(device, min_free_memory_mb=1024):
    if device == "auto":
        if not torch.cuda.is_available():
            print("CUDA is not available, using CPU")
            return torch.device("cpu")

        gpu_idx = _detect_free_gpu(min_free_memory_mb)
        if gpu_idx is not None:
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_idx)
            device_obj = torch.device("cuda")
            props = torch.cuda.get_device_properties(device_obj)
            print(f"Using GPU {gpu_idx}: {props.name}")
            return device_obj

        print("No free GPU detected via nvidia-smi, falling back to default CUDA device")
        return torch.device("cuda")

    if device.startswith("cuda"):
        try:
            gpu_idx = int(device.split(":")[1]) if ":" in device else 0
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_idx)
            return torch.device("cuda")
        except (IndexError, ValueError):
            return torch.device(device)

    return torch.device(device)


def prepare_dirs(data_dir, sample_dir, checkpoint_dir):
    data_dir.mkdir(parents=True, exist_ok=True)
    sample_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)


def make_noise(batch_size, latent_dim, device):
    return torch.randn(batch_size, latent_dim, device=device)

def get_visualizer(environment):
    viz = visdom.Visdom(env=environment)
    window = viz.line(
        Y=torch.full((1,2), float("nan")),
        X=[0],
        win="loss",
        opts={
            'showlegend': True,
            'title': "loss",
            'xlabel': "rate",
            'ylabel': "loss",
            'legend': ["discriminator", "generator"],
        })
    return viz, window
```

要点：
- `get_device("auto")`：自动调用 `nvidia-smi` 找到空闲显存最大的 GPU，无 GPU 回退 CPU
- `make_noise`：从 `N(0, 1)` 标准正态采样，这是 GAN 的默认选择
- `get_visualizer`：创建 Visdom 实时 loss 曲线，`update="append"` 模式可以逐 epoch 追加数据点

### 2.5 train.py — 训练循环（核心）

这是整个项目的心脏。配合代码注释逐行理解：

```python
import tqdm
import torch
from torchvision.utils import save_image

from utils import make_noise


def save_checkpoint(epoch, generator, discriminator, checkpoint_dir):
    checkpoint = {
        "epoch": epoch,
        "generator": generator.state_dict(),
        "discriminator": discriminator.state_dict(),
    }
    torch.save(checkpoint, checkpoint_dir / f"epoch_{epoch:03d}.pt")
    torch.save(checkpoint, checkpoint_dir / "latest.pt")


def save_samples(epoch, generator, fixed_noise, sample_dir):
    generator.eval()
    with torch.no_grad():
        fake_images = generator(fixed_noise)
        fake_images = (fake_images + 1) / 2
        save_image(fake_images, sample_dir / f"epoch_{epoch:03d}.png", nrow=8)
    generator.train()


def train_gan(
    generator,
    discriminator,
    dataloader,
    criterion,
    optimizer_g,
    optimizer_d,
    fixed_noise,
    device,
    latent_dim,
    epochs,
    sample_every,
    checkpoint_every,
    sample_dir,
    checkpoint_dir,
    viz,
    window
):
    for epoch in range(1, epochs + 1):
        last_d_loss = 0.0
        last_g_loss = 0.0

        for real_images, _ in tqdm.tqdm(dataloader):
            real_images = real_images.to(device)
            batch_size = real_images.size(0)

            # ---- Step 1: 训练 Discriminator ----
            # 真图标签 = 1，假图标签 = 0
            real_labels = torch.ones(batch_size, 1, device=device)
            fake_labels = torch.zeros(batch_size, 1, device=device)

            noise = make_noise(batch_size, latent_dim, device)
            fake_images = generator(noise)

            # D 对真图的判断 → 应该接近 1
            real_logits = discriminator(real_images)
            # D 对假图的判断 → 应该接近 0
            fake_logits = discriminator(fake_images.detach())
            d_loss_real = criterion(real_logits, real_labels)
            d_loss_fake = criterion(fake_logits, fake_labels)
            d_loss = d_loss_real + d_loss_fake

            optimizer_d.zero_grad()
            d_loss.backward()
            optimizer_d.step()

            # ---- Step 2: 训练 Generator ----
            noise = make_noise(batch_size, latent_dim, device)
            fake_images = generator(noise)
            fake_logits = discriminator(fake_images)
            # G 的目标：让 D 把假图判为真（标签设为 1）
            g_loss = criterion(fake_logits, real_labels)

            optimizer_g.zero_grad()
            g_loss.backward()
            optimizer_g.step()

            last_d_loss = d_loss.item()
            last_g_loss = g_loss.item()

        # ---- 可视化与日志 ----
        viz.line(
            X=[epoch],
            Y=[[last_d_loss, last_g_loss]],
            win=window,
            update="append",
        )

        print(
            f"Epoch [{epoch:03d}/{epochs}] "
            f"D loss: {last_d_loss:.4f} | G loss: {last_g_loss:.4f}"
        )

        if epoch % sample_every == 0:
            save_samples(epoch, generator, fixed_noise, sample_dir)

        if epoch % checkpoint_every == 0 or epoch == epochs:
            save_checkpoint(epoch, generator, discriminator, checkpoint_dir)
```

**逐段解读**：

**训练 D（判别器）**：
```python
# D 的目标：真图判为真(1)，假图判为假(0)
d_loss_real = criterion(D(real_images),   ones)   # 真图→1
d_loss_fake = criterion(D(G(noise).detach()), zeros)  # 假图→0
d_loss = d_loss_real + d_loss_fake
```
`.detach()` 的含义：更新 D 时，把 G 的输出从计算图中"摘掉"，避免计算 G 的梯度。这让反向传播只更新 D。

**训练 G（生成器）**：
```python
# G 的目标：让 D 把假图判为真
g_loss = criterion(D(G(noise)), ones)  # 假图→1（欺骗 D）
```
这就是"非饱和技巧"：在 minimax game 中，G 原本要最小化 `log(1-D(G(z)))`，但该函数在 D(G(z)) 接近 0 时梯度很弱。改为最大化 `log D(G(z))`（即标签设为 1），梯度更强，训练更容易。

**save_samples**：用一个固定的 `fixed_noise` 每 epoch 生成对比图。噪声相同，只有 G 在进步，可以直观观察生成质量的演化。

**save_checkpoint**：保存的字典包含 epoch 号和模型 state_dict（不含模型结构，只含参数值），文件体积小，加载时需要重新创建模型对象。

### 2.6 main.py — 组装入口

main.py 是"搭积木"的地方——把前面所有模块拼成一个可运行的管线：

```python
import torch.nn as nn
import torch.optim as optim

import config
from dataset import get_dataloader
from models import Discriminator, Generator
from train import train_gan
from utils import get_device, make_noise, prepare_dirs, set_seed, get_visualizer


def main():
    set_seed(config.SEED)
    prepare_dirs(config.DATA_DIR, config.SAMPLE_DIR, config.CHECKPOINT_DIR)

    device = get_device(config.DEVICE, config.MIN_GPU_MEMORY_MB)
    dataloader = get_dataloader(
        config.IMAGE_SIZE,
        config.DATA_DIR,
        config.BATCH_SIZE,
        config.NUM_WORKERS,
    )

    generator = Generator(config.LATENT_DIM, config.CHANNELS, config.IMAGE_SIZE).to(device)
    discriminator = Discriminator(config.CHANNELS, config.IMAGE_SIZE).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer_g = optim.Adam(
        generator.parameters(),
        lr=config.LEARNING_RATE,
        betas=(config.BETA1, 0.999),
    )
    optimizer_d = optim.Adam(
        discriminator.parameters(),
        lr=config.LEARNING_RATE,
        betas=(config.BETA1, 0.999),
    )

    viz, window = get_visualizer("GAN")

    fixed_noise = make_noise(config.NUM_SAMPLE_IMAGES, config.LATENT_DIM, device)

    print(f"Device: {device}")
    print(f"Dataset: {config.DATASET}")
    print(f"Images per epoch: {len(dataloader.dataset)}")

    train_gan(
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

main.py 的 5 步组装流程：
1. **准备环境**：`set_seed` + `prepare_dirs` + `get_device`
2. **加载数据**：`get_dataloader` → DataLoader
3. **创建模型**：`Generator` + `Discriminator`，`.to(device)` 移动到 GPU
4. **配置优化器和 loss**：两个独立的 Adam（G 和 D 各一个）+ `BCEWithLogitsLoss`
5. **启动训练**：调用 `train_gan`，传入所有准备好的组件

### 2.7 generate.py — 推理生成

训练完成后，用 generate.py 生成新图片：

```python
import torch
from torchvision.utils import save_image

import config
from models import Generator
from utils import get_device, make_noise, prepare_dirs, set_seed


def main():
    set_seed(config.SEED)
    prepare_dirs(config.DATA_DIR, config.SAMPLE_DIR, config.CHECKPOINT_DIR)

    device = get_device(config.DEVICE, config.MIN_GPU_MEMORY_MB)
    checkpoint_path = config.CHECKPOINT_DIR / "latest.pt"

    if not checkpoint_path.exists():
        raise FileNotFoundError("No checkpoint found. Run `python GAN/train.py` first.")

    generator = Generator(config.LATENT_DIM, config.CHANNELS, config.IMAGE_SIZE).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    generator.load_state_dict(checkpoint["generator"])
    generator.eval()

    noise = make_noise(config.NUM_SAMPLE_IMAGES, config.LATENT_DIM, device)
    with torch.no_grad():
        images = generator(noise)
        images = (images + 1) / 2
        save_image(images, config.SAMPLE_DIR / "generated.png", nrow=8)

    print(f"Saved generated images to {config.SAMPLE_DIR / 'generated.png'}")


if __name__ == "__main__":
    main()
```

推理三步曲：
1. **加载模型**：创建 Generator 对象 → `load_state_dict` 加载训练好的参数 → `eval()` 切换推理模式
2. **准备输入**：`make_noise` 采样随机噪声 z
3. **生成并保存**：`G(z)` → rescale `[-1,1] → [0,1]` → `save_image`

---

## 3. 运行

```bash
# 终端 1：启动 Visdom 可视化
python -m visdom.server

# 终端 2：训练
cd code/stage01_gan_basics
python main.py

# 训练完成后生成图片
python generate.py
```

打开 `http://localhost:8097` 查看实时 loss 曲线。生成的图片在 `samples/` 目录。

---

## 4. 学习检查点

1. GAN 的 minimax game 公式是什么？G 和 D 分别优化什么项？
2. Generator 输入是什么（形状？分布？），输出是什么（形状？范围？）
3. D 的最后一层为什么不用 Sigmoid？`BCEWithLogitsLoss` 比 `Sigmoid + BCELoss` 好在哪里？
4. 训练 G 时为什么要 `.detach()` D 的输出？`.detach()` 做了什么？
5. `save_samples` 中 `(fake_images + 1) / 2` 的作用是什么？

---

下一篇：[Stage 02 — DCGAN](../stage02_dcgan/dcgan.md)

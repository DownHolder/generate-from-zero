import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

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

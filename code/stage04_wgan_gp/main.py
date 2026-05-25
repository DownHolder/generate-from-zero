import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

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
    optimizer_g = optim.Adam(generator.parameters(), lr=config.LEARNING_RATE, betas=(config.BETA1, config.BETA2))
    optimizer_c = optim.Adam(critic.parameters(), lr=config.LEARNING_RATE, betas=(config.BETA1, config.BETA2))

    viz, window = get_visualizer("WGAN-GP", legend=["critic", "generator"])
    fixed_noise = make_noise(config.NUM_SAMPLE_IMAGES, config.LATENT_DIM, device)

    print(f"Device: {device}")
    print(f"Dataset: {config.DATASET}")
    print(f"Model: WGAN-GP")

    train_gan(
        generator=generator,
        critic=critic,
        dataloader=dataloader,
        optimizer_g=optimizer_g,
        optimizer_c=optimizer_c,
        fixed_noise=fixed_noise,
        device=device,
        latent_dim=config.LATENT_DIM,
        epochs=config.EPOCHS,
        n_critic=config.N_CRITIC,
        lambda_gp=config.LAMBDA_GP,
        sample_every=config.SAMPLE_EVERY,
        checkpoint_every=config.CHECKPOINT_EVERY,
        sample_dir=config.SAMPLE_DIR,
        checkpoint_dir=config.CHECKPOINT_DIR,
        viz=viz,
        window=window,
    )


if __name__ == "__main__":
    main()

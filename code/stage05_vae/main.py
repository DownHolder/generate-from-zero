import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

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
        vae=vae,
        dataloader=dataloader,
        optimizer=optimizer,
        fixed_noise=fixed_noise,
        device=device,
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

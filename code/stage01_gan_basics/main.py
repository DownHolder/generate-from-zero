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

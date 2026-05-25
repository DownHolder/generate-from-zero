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

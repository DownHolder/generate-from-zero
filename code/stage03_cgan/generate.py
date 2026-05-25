import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

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

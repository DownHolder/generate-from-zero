import torch
from torchvision.utils import save_image


def save_checkpoint(epoch, checkpoint_dir, **state_dicts):
    checkpoint = {"epoch": epoch, **state_dicts}
    torch.save(checkpoint, checkpoint_dir / f"epoch_{epoch:03d}.pt")
    torch.save(checkpoint, checkpoint_dir / "latest.pt")


def save_gan_samples(epoch, generator, fixed_noise, sample_dir, fixed_labels=None):
    generator.eval()
    with torch.no_grad():
        if fixed_labels is not None:
            fake = generator(fixed_noise, fixed_labels)
        else:
            fake = generator(fixed_noise)
        fake = (fake + 1) / 2
        save_image(fake, sample_dir / f"epoch_{epoch:03d}.png", nrow=8)
    generator.train()

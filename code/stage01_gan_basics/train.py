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

            real_labels = torch.ones(batch_size, 1, device=device)
            fake_labels = torch.zeros(batch_size, 1, device=device)

            noise = make_noise(batch_size, latent_dim, device)
            fake_images = generator(noise)

            real_logits = discriminator(real_images)
            fake_logits = discriminator(fake_images.detach())
            d_loss_real = criterion(real_logits, real_labels)
            d_loss_fake = criterion(fake_logits, fake_labels)
            d_loss = d_loss_real + d_loss_fake

            optimizer_d.zero_grad()
            d_loss.backward()
            optimizer_d.step()

            noise = make_noise(batch_size, latent_dim, device)
            fake_images = generator(noise)
            fake_logits = discriminator(fake_images)
            g_loss = criterion(fake_logits, real_labels)

            optimizer_g.zero_grad()
            g_loss.backward()
            optimizer_g.step()

            last_d_loss = d_loss.item()
            last_g_loss = g_loss.item()

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


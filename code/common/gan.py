import tqdm
import torch

from .utils import make_noise
from .training import save_checkpoint, save_gan_samples


def train_bce_gan(
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
    window,
    fixed_labels=None,
):
    for epoch in range(1, epochs + 1):
        last_d_loss = 0.0
        last_g_loss = 0.0

        for real_images, labels in tqdm.tqdm(dataloader):
            real_images = real_images.to(device)
            batch_size = real_images.size(0)

            real_targets = torch.ones(batch_size, 1, device=device)
            fake_targets = torch.zeros(batch_size, 1, device=device)

            noise = make_noise(batch_size, latent_dim, device)

            # ---- 条件 GAN 时传入 labels ----
            if fixed_labels is not None:
                batch_labels = labels.to(device)
                fake_images = generator(noise, batch_labels)
                real_logits = discriminator(real_images, batch_labels)
                fake_logits = discriminator(fake_images.detach(), batch_labels)
            else:
                fake_images = generator(noise)
                real_logits = discriminator(real_images)
                fake_logits = discriminator(fake_images.detach())

            d_loss_real = criterion(real_logits, real_targets)
            d_loss_fake = criterion(fake_logits, fake_targets)
            d_loss = d_loss_real + d_loss_fake

            optimizer_d.zero_grad()
            d_loss.backward()
            optimizer_d.step()

            # ---- Generator update ----
            noise = make_noise(batch_size, latent_dim, device)

            if fixed_labels is not None:
                batch_labels = labels.to(device)
                fake_images = generator(noise, batch_labels)
                fake_logits = discriminator(fake_images, batch_labels)
            else:
                fake_images = generator(noise)
                fake_logits = discriminator(fake_images)

            g_loss = criterion(fake_logits, real_targets)

            optimizer_g.zero_grad()
            g_loss.backward()
            optimizer_g.step()

            last_d_loss = d_loss.item()
            last_g_loss = g_loss.item()

        viz.line(X=[epoch], Y=[[last_d_loss, last_g_loss]], win=window, update="append")

        print(
            f"Epoch [{epoch:03d}/{epochs}] "
            f"D loss: {last_d_loss:.4f} | G loss: {last_g_loss:.4f}"
        )

        if epoch % sample_every == 0:
            save_gan_samples(epoch, generator, fixed_noise, sample_dir, fixed_labels)

        if epoch % checkpoint_every == 0 or epoch == epochs:
            save_checkpoint(epoch, checkpoint_dir, generator=generator, discriminator=discriminator)

import tqdm
import torch

from common import make_noise, save_checkpoint, save_gan_samples


def compute_gradient_penalty(critic, real, fake, device):
    batch_size = real.size(0)
    epsilon = torch.rand(batch_size, 1, 1, 1, device=device)
    interpolates = epsilon * real + (1 - epsilon) * fake
    interpolates.requires_grad_(True)

    d_interpolates = critic(interpolates)

    grads = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(d_interpolates),
        create_graph=True,
    )[0]

    grads = grads.view(batch_size, -1)
    gp = ((grads.norm(2, dim=1) - 1) ** 2).mean()
    return gp


def train_gan(
    generator,
    critic,
    dataloader,
    optimizer_g,
    optimizer_c,
    fixed_noise,
    device,
    latent_dim,
    epochs,
    n_critic,
    lambda_gp,
    sample_every,
    checkpoint_every,
    sample_dir,
    checkpoint_dir,
    viz,
    window,
):
    for epoch in range(1, epochs + 1):
        last_c_loss = 0.0
        last_g_loss = 0.0
        last_gp = 0.0

        for real_images, _ in tqdm.tqdm(dataloader):
            real_images = real_images.to(device)
            batch_size = real_images.size(0)

            for _ in range(n_critic):
                with torch.no_grad():
                    noise = make_noise(batch_size, latent_dim, device)
                    fake_images = generator(noise)

                c_real = critic(real_images).mean()
                c_fake = critic(fake_images).mean()
                c_loss = -(c_real - c_fake)

                gp = compute_gradient_penalty(critic, real_images, fake_images, device)
                c_loss_total = c_loss + lambda_gp * gp

                optimizer_c.zero_grad()
                c_loss_total.backward()
                optimizer_c.step()

            noise = make_noise(batch_size, latent_dim, device)
            fake_images = generator(noise)
            g_loss = -critic(fake_images).mean()

            optimizer_g.zero_grad()
            g_loss.backward()
            optimizer_g.step()

            last_c_loss = c_loss.item()
            last_g_loss = g_loss.item()
            last_gp = gp.item()

        viz.line(X=[epoch], Y=[[last_c_loss, last_g_loss]], win=window, update="append")

        print(
            f"Epoch [{epoch:03d}/{epochs}] "
            f"C loss: {last_c_loss:.4f} | GP: {last_gp:.4f} | G loss: {last_g_loss:.4f}"
        )

        if epoch % sample_every == 0:
            save_gan_samples(epoch, generator, fixed_noise, sample_dir)

        if epoch % checkpoint_every == 0 or epoch == epochs:
            save_checkpoint(epoch, checkpoint_dir, generator=generator, critic=critic)

import tqdm
import torch
import torch.nn.functional as F
import visdom

from common import save_checkpoint


def get_vae_visualizer(environment="VAE"):
    viz = visdom.Visdom(env=environment)
    window = viz.line(
        Y=torch.full((1, 2), float("nan")),
        X=[0],
        win="loss",
        opts={
            "showlegend": True,
            "title": "VAE Loss",
            "xlabel": "epoch",
            "ylabel": "loss",
            "legend": ["reconstruction", "kl_divergence"],
        },
    )
    return viz, window


def save_vae_samples(epoch, vae, fixed_noise, sample_dir):
    from torchvision.utils import save_image
    vae.eval()
    with torch.no_grad():
        fake_images = vae.decode(fixed_noise)
        fake_images = (fake_images + 1) / 2
        save_image(fake_images, sample_dir / f"epoch_{epoch:03d}.png", nrow=8)
    vae.train()


def train_vae(
    vae,
    dataloader,
    optimizer,
    fixed_noise,
    device,
    epochs,
    sample_every,
    checkpoint_every,
    sample_dir,
    checkpoint_dir,
    viz,
    window,
):
    for epoch in range(1, epochs + 1):
        last_recon_loss = 0.0
        last_kl_loss = 0.0

        for real_images, _ in tqdm.tqdm(dataloader):
            real_images = real_images.to(device)
            batch_size = real_images.size(0)

            x_recon, mu, logvar = vae(real_images)

            recon_loss = F.mse_loss(
                x_recon.view(batch_size, -1), real_images.view(batch_size, -1), reduction="sum"
            ) / batch_size
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / batch_size
            loss = recon_loss + kl_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            last_recon_loss = recon_loss.item()
            last_kl_loss = kl_loss.item()

        viz.line(X=[epoch], Y=[[last_recon_loss, last_kl_loss]], win=window, update="append")

        print(
            f"Epoch [{epoch:03d}/{epochs}] "
            f"Recon: {last_recon_loss:.4f} | KL: {last_kl_loss:.4f}"
        )

        if epoch % sample_every == 0:
            save_vae_samples(epoch, vae, fixed_noise, sample_dir)

        if epoch % checkpoint_every == 0 or epoch == epochs:
            save_checkpoint(epoch, checkpoint_dir, vae=vae, optimizer=optimizer)

from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_dataloader(image_size, data_dir, batch_size, num_workers, dataset_name="mnist"):
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])

    if dataset_name == "mnist":
        dataset = datasets.MNIST(
            root=data_dir, train=True, download=True, transform=transform,
        )
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

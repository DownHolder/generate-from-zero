import os
import random
import subprocess
import visdom
import torch


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _detect_free_gpu(min_free_memory_mb=1024):
    gpu_info = []
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.free,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
        for line in result.stdout.strip().split("\n"):
            line = line.strip()
            if not line:
                continue
            parts = [p.strip() for p in line.split(", ")]
            if len(parts) >= 4:
                try:
                    idx = int(parts[0])
                    mem_free = int(parts[1])
                    mem_total = int(parts[2])
                    util_gpu = int(parts[3])
                    gpu_info.append((idx, mem_free, mem_total, util_gpu))
                except ValueError:
                    continue
    except (subprocess.SubprocessError, FileNotFoundError, ValueError):
        return None

    if not gpu_info:
        return None

    eligible = [(idx, free, total, util) for idx, free, total, util in gpu_info if free >= min_free_memory_mb]

    if eligible:
        selected = max(eligible, key=lambda x: x[1])
    else:
        selected = max(gpu_info, key=lambda x: x[1])

    idx, mem_free, mem_total, util_gpu = selected
    print(
        f"Selected GPU {idx}: "
        f"{mem_free}MiB free / {mem_total}MiB total, "
        f"utilization {util_gpu}%"
    )
    return idx


def get_device(device, min_free_memory_mb=1024):
    if device == "auto":
        if not torch.cuda.is_available():
            print("CUDA is not available, using CPU")
            return torch.device("cpu")

        gpu_idx = _detect_free_gpu(min_free_memory_mb)
        if gpu_idx is not None:
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_idx)
            device_obj = torch.device("cuda")
            props = torch.cuda.get_device_properties(device_obj)
            print(f"Using GPU {gpu_idx}: {props.name}")
            return device_obj

        print("No free GPU detected via nvidia-smi, falling back to default CUDA device")
        return torch.device("cuda")

    if device.startswith("cuda"):
        try:
            gpu_idx = int(device.split(":")[1]) if ":" in device else 0
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_idx)
            return torch.device("cuda")
        except (IndexError, ValueError):
            return torch.device(device)

    return torch.device(device)


def prepare_dirs(data_dir, sample_dir, checkpoint_dir):
    data_dir.mkdir(parents=True, exist_ok=True)
    sample_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)


def make_noise(batch_size, latent_dim, device):
    return torch.randn(batch_size, latent_dim, device=device)

def get_visualizer(environment):
    viz = visdom.Visdom(env=environment)
    window = viz.line(
        Y=torch.full((1,2), float("nan")),
        X=[0],
        win="loss",
        opts={
            'showlegend': True,  # 显示网格
            'title': "loss",
            'xlabel': "rate",  # x轴标签
            'ylabel': "loss",  # y轴标签
            'legend': ["discriminator", "generator"],
        })
    return viz, window

from pathlib import Path

# ========== Paths ==========
ROOT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT_DIR.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
SAMPLE_DIR = ROOT_DIR / "samples"
CHECKPOINT_DIR = ROOT_DIR / "checkpoints"

# ========== Dataset ==========
DATASET = "mnist"
IMAGE_SIZE = 28
CHANNELS = 1
NUM_WORKERS = 8

# ========== Model Architecture (VAE 特有) ==========
LATENT_DIM = 20

# ========== Training (VAE 使用不同优化器参数) ==========
BATCH_SIZE = 128
EPOCHS = 50
LEARNING_RATE = 0.001
BETA1 = 0.9
SAMPLE_EVERY = 10
CHECKPOINT_EVERY = 10

# ========== Inference ==========
NUM_SAMPLE_IMAGES = 64

# ========== Runtime ==========
DEVICE = "auto"
SEED = 42
MIN_GPU_MEMORY_MB = 4096

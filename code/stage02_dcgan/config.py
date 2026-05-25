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

# ========== Model Architecture (DCGAN 特有) ==========
LATENT_DIM = 100
DCGAN_FEATURES = 64

# ========== Training ==========
BATCH_SIZE = 128
EPOCHS = 50
LEARNING_RATE = 0.0002
BETA1 = 0.5
SAMPLE_EVERY = 10
CHECKPOINT_EVERY = 10

# ========== Inference ==========
NUM_SAMPLE_IMAGES = 64

# ========== Runtime ==========
DEVICE = "auto"
SEED = 42
MIN_GPU_MEMORY_MB = 4096

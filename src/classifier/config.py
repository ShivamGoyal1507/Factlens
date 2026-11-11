"""Configuration for the FACTLENS FEVER claim verifier."""

from pathlib import Path

MODEL_NAME = "microsoft/deberta-v3-base"
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# The training run identifies checkpoint-34917 as its best checkpoint.  The
# fallback is useful when only the separately exported adapter is copied.
MODEL_OUTPUT_DIR = PROJECT_ROOT / "experiments" / "classifier" / "output"
TRAINED_MODEL_DIR = (
    MODEL_OUTPUT_DIR
    / "fever_deberta_lora_final"
    / "checkpoints"
    / "checkpoint-34917"
)
EXPORTED_ADAPTER_DIR = (
    MODEL_OUTPUT_DIR / "fever_deberta_lora_final" / "best_lora_model"
)
BASE_MODEL_DIR = MODEL_OUTPUT_DIR / "fever-deberta-v3-base"

MAX_LENGTH = 256
NUM_LABELS = 3
LABEL2ID = {
    "SUPPORTS": 0,
    "REFUTES": 1,
    "NEI": 2,
}
ID2LABEL = {value: key for key, value in LABEL2ID.items()}

# These values document the adapter used to train the saved checkpoint.
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.10
LORA_TARGET_MODULES = ["query_proj", "value_proj"]

INFERENCE_BATCH_SIZE = 16
CONFIDENCE_MIN = 0.0
CONFIDENCE_MAX = 1.0


def _default_device() -> str:
    """Return CUDA when the installed PyTorch can use it, otherwise CPU."""
    try:
        import torch
    except ImportError:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


DEVICE = _default_device()

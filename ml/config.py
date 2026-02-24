"""Configuration helpers for ML training and inference."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import os

from ml import DEFAULT_VOCABULARY

# Core landmark/classifier configuration.
LANDMARK_DIM = 126       # both hands, 21 landmarks × 2 × 3
SEQUENCE_LENGTH = 30     # frames
D_MODEL = 256
N_HEADS = 8
N_LAYERS = 4
DROPOUT = 0.1
BATCH_SIZE = 32
MAX_EPOCHS = 100
EARLY_STOPPING_PATIENCE = 10


@dataclass(slots=True)
class MLConfig:
    """Runtime and training configuration for gesture modeling."""

    input_size: int = LANDMARK_DIM
    sequence_length: int = SEQUENCE_LENGTH
    predict_every_n_frames: int = 5
    confidence_threshold: float = 0.5
    model_architecture: str = "hybrid"
    model_checkpoint_path: Path = Path("ml/models/gesture_model.pt")
    vocabulary: list[str] = field(default_factory=lambda: DEFAULT_VOCABULARY.copy())

    @classmethod
    def from_env(cls) -> "MLConfig":
        """Build configuration from environment variables."""
        vocab_str = os.getenv("GESTURE_VOCAB", ",".join(DEFAULT_VOCABULARY))
        vocabulary = [token.strip() for token in vocab_str.split(",") if token.strip()]
        return cls(
            input_size=int(os.getenv("ML_INPUT_SIZE", str(LANDMARK_DIM))),
            sequence_length=int(os.getenv("ML_SEQUENCE_LENGTH", str(SEQUENCE_LENGTH))),
            predict_every_n_frames=int(os.getenv("ML_PREDICT_EVERY_N", "5")),
            confidence_threshold=float(os.getenv("ML_CONFIDENCE_THRESHOLD", "0.5")),
            model_architecture=os.getenv("ML_ARCHITECTURE", "hybrid"),
            model_checkpoint_path=Path(
                os.getenv("MODEL_CHECKPOINT_PATH", "ml/models/gesture_model.pt"),
            ),
            vocabulary=vocabulary or DEFAULT_VOCABULARY.copy(),
        )


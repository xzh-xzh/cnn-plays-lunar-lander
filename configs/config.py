from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, Literal


@dataclass(slots=True)
class TrainingConfig:
    # Environment
    ENV_ID: str = "LunarLander-v3"
    NUM_ENVS: int = 8
    VECTOR_ENV: Literal["sync", "async"] = "sync"
    FRAME_SIZE: int = 128
    STACK_SIZE: int = 16
    MAX_EPISODE_STEPS: int = 250
    SEED: int = 7

    # Runtime
    DEVICE: str = "auto"
    USE_MIXED_PRECISION: bool = True
    USE_TORCH_COMPILE: bool = False
    PIN_MEMORY: bool = True
    USE_MLFLOW: bool = False
    OUTPUT_DIR: str = "runs"
    CHECKPOINT_DIR: str = "checkpoints"
    LOG_INTERVAL: int = 1

    # Model
    model_name: str = "TinyCNNGRU"
    PRETRAINED_BACKBONE: bool = False
    FREEZE_RESNET: bool = True
    HIDDEN_SIZE: int = 256
    GRU_LAYERS: int = 1

    # PPO
    LEARNING_RATE: float = 3e-4
    BACKBONE_LEARNING_RATE: float | None = None
    GAMMA: float = 0.99
    GAE_LAMBDA: float = 0.95
    VALUE_COEFF: float = 0.5
    ENTROPY_COEFF: float = 0.01
    CLIP_EPSILON: float = 0.2
    TARGET_KL: float | None = None
    PPO_EPOCHS: int = 4
    BATCH_SIZE: int = 128
    NUM_ROUNDS: int = 1000
    NUM_STEPS_PER_ROLLOUT: int = 1024
    MAX_GRAD_NORM: float = 1.0
    CHECKPOINT_FREQUENCY: int = 50
    CHECKPOINT_PATH: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TrainingConfig":
        valid = {field.name for field in fields(cls)}
        unknown = sorted(set(data) - valid)
        if unknown:
            joined = ", ".join(unknown)
            raise ValueError(f"Unknown config key(s): {joined}")
        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        return {field.name: getattr(self, field.name) for field in fields(self)}

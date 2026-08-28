"""所有训练入口共享的基础训练超参数。"""

NUM_WORKERS = 8
BATCH_SIZE = 256
EPOCHS = 100
LEARNING_RATE = 1e-2
WEIGHT_DECAY = 1e-4
GRADIENT_CLIP = 1.0
WARMUP_EPOCHS = 5
SEED = 42
DETERMINISTIC = False
MIXED_PRECISION = True
EVAL_INTERVAL = 1
CHECKPOINT_INTERVAL = 5

TRAINING_PARAMETER_NAMES = (
    "NUM_WORKERS",
    "BATCH_SIZE",
    "EPOCHS",
    "LEARNING_RATE",
    "WEIGHT_DECAY",
    "GRADIENT_CLIP",
    "WARMUP_EPOCHS",
    "SEED",
    "DETERMINISTIC",
    "MIXED_PRECISION",
    "EVAL_INTERVAL",
    "CHECKPOINT_INTERVAL",
)


def print_training_config(values):
    """打印调用方实际生效的训练超参数，包括 Notebook 中的显式覆盖。"""

    print("训练超参数：")
    for name in TRAINING_PARAMETER_NAMES:
        if name in values:
            print(f"  {name} = {values[name]}")


__all__ = [
    "BATCH_SIZE",
    "CHECKPOINT_INTERVAL",
    "DETERMINISTIC",
    "EPOCHS",
    "EVAL_INTERVAL",
    "GRADIENT_CLIP",
    "LEARNING_RATE",
    "MIXED_PRECISION",
    "NUM_WORKERS",
    "SEED",
    "WARMUP_EPOCHS",
    "WEIGHT_DECAY",
    "print_training_config",
]

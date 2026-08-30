"""
oom-check: Predict whether a training run will fit on your GPU
before you actually run it and waste time on a crash.

Core idea: GPU memory during training = 
    model weights + gradients + optimizer state + activations
"""

BYTES_PER_DTYPE = {
    "fp32": 4,
    "fp16": 2,
    "bf16": 2,
    "int8": 1,
    "int4": 0.5,
}


def estimate_training_memory_gb(
    num_params_billion: float,
    dtype: str = "fp16",
    batch_size: int = 1,
    seq_len: int = 2048,
    hidden_size: int = 4096,
    num_layers: int = 32,
    optimizer: str = "adamw",
    lora: bool = False,
    lora_trainable_pct: float = 0.02,
    gradient_checkpointing: bool = False,
):
    bytes_per_param = BYTES_PER_DTYPE.get(dtype, 2)
    num_params = num_params_billion * 1e9

    weights_gb = (num_params * bytes_per_param) / 1e9

    if lora:
        trainable_params = num_params * lora_trainable_pct
    else:
        trainable_params = num_params
    gradients_gb = (trainable_params * bytes_per_param) / 1e9

    if optimizer == "adamw":
        optimizer_gb = (trainable_params * 4 * 2) / 1e9
    elif optimizer == "sgd":
        optimizer_gb = (trainable_params * 4 * 1) / 1e9
    else:
        optimizer_gb = 0

    activation_factor = 12
    activations_gb = (
        batch_size * seq_len * hidden_size * num_layers * activation_factor * bytes_per_param
    ) / 1e9

    if gradient_checkpointing:
        activations_gb = activations_gb / (num_layers ** 0.5)

    total_gb = weights_gb + gradients_gb + optimizer_gb + activations_gb

    return {
        "weights_gb": round(weights_gb, 2),
        "gradients_gb": round(gradients_gb, 2),
        "optimizer_gb": round(optimizer_gb, 2),
        "activations_gb": round(activations_gb, 2),
        "total_gb": round(total_gb, 2),
    }


def verdict(total_gb: float, gpu_vram_gb: float, safety_margin: float = 0.9):
    usable_vram = gpu_vram_gb * safety_margin
    if total_gb <= usable_vram:
        headroom = usable_vram - total_gb
        return f"Fits. Estimated {total_gb:.1f} GB used of ~{usable_vram:.1f} GB usable ({headroom:.1f} GB headroom)."
    else:
        overflow = total_gb - usable_vram
        return f"Won't fit. Estimated {total_gb:.1f} GB needed, only ~{usable_vram:.1f} GB usable ({overflow:.1f} GB over)."
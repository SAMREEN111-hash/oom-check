"""
oom-check: Predict whether a training run will fit on your GPU
before you actually run it and waste time on a crash.
"""

BYTES_PER_DTYPE = {
    "fp32": 4,
    "fp16": 2,
    "bf16": 2,
    "int8": 1,
    "int4": 0.5,
}

OPTIMIZER_BYTES_PER_PARAM = {
    "adamw": 8,
    "adamw_8bit": 2,
    "sgd": 4,
    "sgd_8bit": 1,
    "none": 0,
}

CUDA_CONTEXT_OVERHEAD_GB = 0.4
FRAGMENTATION_OVERHEAD_PCT = 0.10
ACTIVATION_UNCERTAINTY_PCT = 0.15


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
    attn_implementation: str = "flash",
    base_dtype: str = None,
):
    bytes_per_param = BYTES_PER_DTYPE.get(dtype, 2)
    num_params = num_params_billion * 1e9

    weight_bytes_per_param = BYTES_PER_DTYPE.get(base_dtype, bytes_per_param) if base_dtype else bytes_per_param
    weights_gb = (num_params * weight_bytes_per_param) / 1e9

    if base_dtype and not lora:
        raise ValueError(
            "base_dtype (quantized base model) only makes sense with lora=True. "
            "You can't meaningfully compute gradients through 4-bit/8-bit frozen "
            "weights in a full fine-tune - that's what LoRA adapters are for."
        )

    if lora:
        trainable_params = num_params * lora_trainable_pct
    else:
        trainable_params = num_params
    gradients_gb = (trainable_params * bytes_per_param) / 1e9

    opt_bytes = OPTIMIZER_BYTES_PER_PARAM.get(optimizer, 8)
    optimizer_gb = (trainable_params * opt_bytes) / 1e9

    linear_factor = 12
    linear_activations_gb = (
        batch_size * seq_len * hidden_size * num_layers * linear_factor * bytes_per_param
    ) / 1e9

    if attn_implementation == "flash":
        quadratic_activations_gb = 0
    else:
        num_heads = max(1, hidden_size // 128)
        quadratic_activations_gb = (
            batch_size * num_heads * (seq_len ** 2) * num_layers * bytes_per_param
        ) / 1e9

    activations_gb = linear_activations_gb + quadratic_activations_gb

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


def estimate_with_overhead(result: dict):
    fixed_total = result["weights_gb"] + result["gradients_gb"] + result["optimizer_gb"]
    act = result["activations_gb"]

    act_low = act * (1 - ACTIVATION_UNCERTAINTY_PCT)
    act_high = act * (1 + ACTIVATION_UNCERTAINTY_PCT)

    low = (fixed_total + act_low) * (1 + FRAGMENTATION_OVERHEAD_PCT) + CUDA_CONTEXT_OVERHEAD_GB
    point = (fixed_total + act) * (1 + FRAGMENTATION_OVERHEAD_PCT) + CUDA_CONTEXT_OVERHEAD_GB
    high = (fixed_total + act_high) * (1 + FRAGMENTATION_OVERHEAD_PCT) + CUDA_CONTEXT_OVERHEAD_GB

    return {
        "low_gb": round(low, 2),
        "point_gb": round(point, 2),
        "high_gb": round(high, 2),
    }


def verdict(total_gb: float, gpu_vram_gb: float, safety_margin: float = 0.9):
    usable_vram = gpu_vram_gb * safety_margin
    if total_gb <= usable_vram:
        headroom = usable_vram - total_gb
        return f"Fits. Estimated {total_gb:.1f} GB used of ~{usable_vram:.1f} GB usable ({headroom:.1f} GB headroom)."
    else:
        overflow = total_gb - usable_vram
        return f"Won't fit. Estimated {total_gb:.1f} GB needed, only ~{usable_vram:.1f} GB usable ({overflow:.1f} GB over)."


def verdict_with_range(overhead_result: dict, gpu_vram_gb: float):
    low, point, high = overhead_result["low_gb"], overhead_result["point_gb"], overhead_result["high_gb"]

    if high <= gpu_vram_gb:
        headroom = gpu_vram_gb - high
        return f"Fits. Estimated {low:.1f}-{high:.1f} GB (best guess ~{point:.1f} GB) of {gpu_vram_gb:.1f} GB available ({headroom:.1f} GB headroom even in the worst case)."
    elif low > gpu_vram_gb:
        overflow = low - gpu_vram_gb
        return f"Won't fit. Estimated {low:.1f}-{high:.1f} GB (best guess ~{point:.1f} GB) needed, only {gpu_vram_gb:.1f} GB available (at least {overflow:.1f} GB over, even in the best case)."
    else:
        return (
            f"Borderline. Estimated {low:.1f}-{high:.1f} GB (best guess ~{point:.1f} GB) "
            f"against {gpu_vram_gb:.1f} GB available. This could go either way in practice - "
            f"try it, but be ready to lower the batch size by one notch if it OOMs."
        )


def suggest_batch_size(
    num_params_billion, gpu_vram_gb, dtype="fp16", seq_len=2048,
    hidden_size=4096, num_layers=32, optimizer="adamw", lora=False,
    gradient_checkpointing=False, base_dtype=None, safety_margin=0.9, max_search=256,
):
    usable_vram = gpu_vram_gb * safety_margin
    best_fit = None
    for bs in range(max_search, 0, -1):
        result = estimate_training_memory_gb(
            num_params_billion=num_params_billion, dtype=dtype, batch_size=bs,
            seq_len=seq_len, hidden_size=hidden_size, num_layers=num_layers,
            optimizer=optimizer, lora=lora, gradient_checkpointing=gradient_checkpointing,
            base_dtype=base_dtype,
        )
        if result["total_gb"] <= usable_vram:
            best_fit = bs
            break
    return best_fit
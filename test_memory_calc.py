"""
Tests for the memory estimation logic.

These specific cases are all real, well-known real-world scenarios
that we manually verified match reality tonight. Locking them in as
tests means if the formulas change later, we immediately know if
something that used to be correct got broken.
"""

from memory_calc import estimate_training_memory_gb, verdict, suggest_batch_size


def test_llama7b_full_finetune_does_not_fit_on_24gb():
    result = estimate_training_memory_gb(
        num_params_billion=7, dtype="fp16", batch_size=4,
        seq_len=2048, hidden_size=4096, num_layers=32, optimizer="adamw",
    )
    assert result["total_gb"] > 24
    assert "Won't fit" in verdict(result["total_gb"], 24)


def test_llama7b_lora_finetune_fits_on_24gb():
    result = estimate_training_memory_gb(
        num_params_billion=7, dtype="fp16", batch_size=4,
        seq_len=2048, hidden_size=4096, num_layers=32, optimizer="adamw",
        lora=True, gradient_checkpointing=True,
    )
    assert result["total_gb"] < 24
    assert "Fits" in verdict(result["total_gb"], 24)


def test_bert_base_full_finetune_fits_on_8gb():
    result = estimate_training_memory_gb(
        num_params_billion=0.11, dtype="fp32", batch_size=16,
        seq_len=512, hidden_size=768, num_layers=12, optimizer="adamw",
    )
    assert result["total_gb"] < 8
    assert "Fits" in verdict(result["total_gb"], 8)


def test_llama13b_full_finetune_does_not_fit_on_80gb():
    result = estimate_training_memory_gb(
        num_params_billion=13, dtype="fp16", batch_size=4,
        seq_len=2048, hidden_size=5120, num_layers=40, optimizer="adamw",
    )
    assert result["total_gb"] > 80


def test_8bit_optimizer_uses_less_memory_than_regular_adamw():
    regular = estimate_training_memory_gb(num_params_billion=7, optimizer="adamw")
    eight_bit = estimate_training_memory_gb(num_params_billion=7, optimizer="adamw_8bit")
    assert eight_bit["optimizer_gb"] < regular["optimizer_gb"]


def test_lora_uses_far_less_gradient_memory_than_full_finetune():
    full = estimate_training_memory_gb(num_params_billion=7, lora=False)
    lora = estimate_training_memory_gb(num_params_billion=7, lora=True)
    assert lora["gradients_gb"] < full["gradients_gb"]


def test_gradient_checkpointing_reduces_activation_memory():
    without = estimate_training_memory_gb(num_params_billion=7, gradient_checkpointing=False)
    with_ckpt = estimate_training_memory_gb(num_params_billion=7, gradient_checkpointing=True)
    assert with_ckpt["activations_gb"] < without["activations_gb"]


def test_suggest_batch_size_returns_a_smaller_fitting_batch_size():
    suggested = suggest_batch_size(
        num_params_billion=7, gpu_vram_gb=24, dtype="fp16",
        seq_len=2048, hidden_size=4096, num_layers=32,
        optimizer="adamw_8bit", lora=True, gradient_checkpointing=True,
    )
    assert suggested is not None
    assert suggested >= 1
def test_flash_attention_uses_less_activation_memory_than_naive():
    """Flash attention avoids materializing the full N^2 attention score
    matrix, which should mean noticeably less activation memory,
    especially at longer sequence lengths."""
    flash = estimate_training_memory_gb(
        num_params_billion=7, seq_len=16384, attn_implementation="flash",
    )
    naive = estimate_training_memory_gb(
        num_params_billion=7, seq_len=16384, attn_implementation="naive",
    )
    assert flash["activations_gb"] < naive["activations_gb"]

def test_qlora_uses_far_less_weight_memory_than_regular_lora():
    """QLoRA (4-bit quantized base) should need much less memory for
    the base model weights than regular LoRA (fp16 base) - this is
    exactly what makes QLoRA able to fit large models on small GPUs."""
    regular_lora = estimate_training_memory_gb(
        num_params_billion=7, dtype="fp16", lora=True,
    )
    qlora = estimate_training_memory_gb(
        num_params_billion=7, dtype="fp16", lora=True, base_dtype="int4",
    )
    assert qlora["weights_gb"] < regular_lora["weights_gb"]


def test_base_dtype_without_lora_raises_error():
    """Quantized base weights only make sense with LoRA - using
    base_dtype without lora=True should raise a clear error, not
    silently give a wrong answer."""
    import pytest
    with pytest.raises(ValueError):
        estimate_training_memory_gb(
            num_params_billion=7, lora=False, base_dtype="int4",
        )
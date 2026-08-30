import argparse
from memory_calc import estimate_training_memory_gb, verdict, suggest_batch_size
from hf_lookup import fetch_model_config


def main():
    p = argparse.ArgumentParser(
        description="Predict whether your training run fits on your GPU, before you run it."
    )
    p.add_argument("--model", type=str, default=None, help="HF Hub model id, e.g. 'bert-base-uncased'. If given, auto-fetches hidden-size/num-layers/params.")
    p.add_argument("--params-billion", type=float, default=None, help="Model size in billions of params (not needed if --model is given)")
    p.add_argument("--gpu-vram-gb", type=float, required=True, help="Your GPU's VRAM in GB, e.g. 24")
    p.add_argument("--dtype", default="fp16", choices=["fp32", "fp16", "bf16", "int8", "int4"])
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--seq-len", type=int, default=2048)
    p.add_argument("--hidden-size", type=int, default=None, help="Not needed if --model is given")
    p.add_argument("--num-layers", type=int, default=None, help="Not needed if --model is given")
    p.add_argument("--optimizer", default="adamw", choices=["adamw", "adamw_8bit", "sgd", "sgd_8bit", "none"])
    p.add_argument("--lora", action="store_true")
    p.add_argument("--gradient-checkpointing", action="store_true")
    p.add_argument("--attn-implementation", default="flash", choices=["flash", "naive"], help="flash = memory-efficient attention (default, modern), naive = old-style full attention matrix")
    p.add_argument("--base-dtype", default=None, choices=["fp32", "fp16", "bf16", "int8", "int4"], help="For QLoRA: quantized precision of the frozen base model (e.g. int4), separate from --dtype used for the adapter")

    args = p.parse_args()

    if args.model:
        print(f"\nFetching config for '{args.model}' from Hugging Face Hub...")
        info = fetch_model_config(args.model)
        params_billion = info["params_billion"]
        hidden_size = info["hidden_size"]
        num_layers = info["num_layers"]
        print(f"Found: ~{info['params_billion']}B params [{info['param_source']}], hidden_size={hidden_size}, num_layers={num_layers}\n")
    else:
        if args.params_billion is None or args.hidden_size is None or args.num_layers is None:
            p.error("Either provide --model, or provide --params-billion, --hidden-size, and --num-layers manually.")
        params_billion = args.params_billion
        hidden_size = args.hidden_size
        num_layers = args.num_layers

    result = estimate_training_memory_gb(
        num_params_billion=params_billion,
        dtype=args.dtype,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        hidden_size=hidden_size,
        num_layers=num_layers,
        optimizer=args.optimizer,
        lora=args.lora,
        gradient_checkpointing=args.gradient_checkpointing,
        attn_implementation=args.attn_implementation,
        base_dtype=args.base_dtype,
    )

    print("--- Memory Breakdown ---")
    print(f"  Model weights:     {result['weights_gb']} GB")
    print(f"  Gradients:         {result['gradients_gb']} GB")
    print(f"  Optimizer state:   {result['optimizer_gb']} GB   (optimizer: {args.optimizer})")
    print(f"  Activations:       {result['activations_gb']} GB   (attention: {args.attn_implementation})")
    print(f"  TOTAL:             {result['total_gb']} GB")
    print()
    from memory_calc import estimate_with_overhead, verdict_with_range
    overhead = estimate_with_overhead(result)
    print(verdict_with_range(overhead, args.gpu_vram_gb))

    usable_vram = args.gpu_vram_gb * 0.9
    if result["total_gb"] > usable_vram:
        suggested = suggest_batch_size(
            num_params_billion=params_billion,
            gpu_vram_gb=args.gpu_vram_gb,
            dtype=args.dtype,
            seq_len=args.seq_len,
            hidden_size=hidden_size,
            num_layers=num_layers,
            optimizer=args.optimizer,
            lora=args.lora,
            gradient_checkpointing=args.gradient_checkpointing,
        )
        if suggested:
            print(f"Suggestion: try --batch-size {suggested} instead (largest size that fits).")
        else:
            print("Suggestion: even batch_size=1 doesn't fit. Try --lora, --gradient-checkpointing, a smaller dtype, or an 8-bit optimizer (--optimizer adamw_8bit).")
    print()


if __name__ == "__main__":
    main()
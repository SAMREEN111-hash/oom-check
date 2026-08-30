"""
Fetches a model's real config (hidden_size, num_layers, param count)
directly from the Hugging Face Hub, so users don't have to guess or
manually look these values up.

v2: uses the EXACT parameter count from safetensors metadata when
available (via model_info), falling back to our architecture-based
estimate only when that metadata isn't present.
"""

from huggingface_hub import hf_hub_download, model_info
import json


def fetch_model_config(model_id: str):
    try:
        config_path = hf_hub_download(repo_id=model_id, filename="config.json")
    except Exception as e:
        raise ValueError(
            f"Could not fetch config for '{model_id}'. "
            f"Check the model name is correct and public on huggingface.co. "
            f"Original error: {e}"
        )

    with open(config_path, "r") as f:
        config = json.load(f)

    hidden_size = (
        config.get("hidden_size")
        or config.get("d_model")
        or config.get("n_embd")
    )
    num_layers = (
        config.get("num_hidden_layers")
        or config.get("num_layers")
        or config.get("n_layer")
    )

    if hidden_size is None or num_layers is None:
        raise ValueError(
            f"Found config for '{model_id}' but couldn't identify hidden_size/num_layers. "
            f"You can still pass --hidden-size and --num-layers manually."
        )

    exact_params = _try_get_exact_param_count(model_id)

    if exact_params is not None:
        params_billion = round(exact_params / 1e9, 3)
        param_source = "exact (from safetensors metadata)"
    else:
        vocab_size = config.get("vocab_size", 32000)
        intermediate_size = config.get("intermediate_size", hidden_size * 4)
        params_per_layer = (
            4 * hidden_size * hidden_size
            + 2 * hidden_size * intermediate_size
        )
        total_params = (
            params_per_layer * num_layers
            + 2 * vocab_size * hidden_size
        )
        params_billion = round(total_params / 1e9, 3)
        param_source = "estimated (architecture-based approximation)"

    return {
        "model_id": model_id,
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "params_billion": params_billion,
        "param_source": param_source,
    }


def _try_get_exact_param_count(model_id: str):
    try:
        info = model_info(model_id)
        if info.safetensors is not None and info.safetensors.total is not None:
            return info.safetensors.total
    except Exception:
        pass
    return None
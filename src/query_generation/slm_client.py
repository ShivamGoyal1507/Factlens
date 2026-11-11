"""
Small-language-model (SLM) client for FACTLENS query generation.

Default model: Qwen/Qwen2.5-1.5B-Instruct (Hugging Face).
Override with the FACTLENS_QUERY_MODEL environment variable, e.g.
    FACTLENS_QUERY_MODEL=Qwen/Qwen2.5-0.5B-Instruct   (faster, lower quality)
    FACTLENS_QUERY_MODEL=Qwen/Qwen2.5-3B-Instruct     (slower, higher quality)

The model is loaded once per process and reused by every caller.
"""

import os
import threading
import time

DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
MODEL_NAME = os.environ.get("FACTLENS_QUERY_MODEL", DEFAULT_MODEL_NAME)

MAX_INPUT_TOKENS = 2048

_MODEL_CACHE = {
    "tokenizer": None,
    "model": None,
    "device": None,
}

_LOAD_LOCK = threading.Lock()


def _select_device():
    import torch

    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_slm():
    """Load the SLM once and return (tokenizer, model)."""
    if _MODEL_CACHE["model"] is not None:
        return _MODEL_CACHE["tokenizer"], _MODEL_CACHE["model"]

    with _LOAD_LOCK:
        if _MODEL_CACHE["model"] is not None:
            return _MODEL_CACHE["tokenizer"], _MODEL_CACHE["model"]

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        device = _select_device()
        dtype = torch.float32 if device.type == "cpu" else torch.float16

        print(f"[SLM] Loading {MODEL_NAME} on {device} ({dtype})")

        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=dtype)
        model.to(device)
        model.eval()

        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

        _MODEL_CACHE["tokenizer"] = tokenizer
        _MODEL_CACHE["model"] = model
        _MODEL_CACHE["device"] = device

        print("[SLM] Model loaded.")
        return tokenizer, model


def generate_chat(messages, max_new_tokens=192):
    """
    Run deterministic (greedy) generation on a list of chat messages.

    Args:
        messages: [{"role": "system"|"user"|"assistant", "content": str}, ...]
        max_new_tokens: generation budget.

    Returns:
        dict with "text", "generation_time_seconds", "input_tokens",
        "output_tokens" and "device".
    """
    if not isinstance(messages, list) or not messages:
        raise ValueError("messages must be a non-empty list.")
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be positive.")

    import torch

    tokenizer, model = load_slm()
    device = _MODEL_CACHE["device"]

    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )

    input_length = inputs["input_ids"].shape[1]
    if input_length > MAX_INPUT_TOKENS:
        raise ValueError(
            f"Prompt has {input_length} tokens; limit is {MAX_INPUT_TOKENS}."
        )

    inputs = {key: value.to(device) for key, value in inputs.items()}

    start = time.perf_counter()
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    elapsed = time.perf_counter() - start

    generated = outputs[0][input_length:]
    text = tokenizer.decode(generated, skip_special_tokens=True).strip()

    return {
        "text": text,
        "generation_time_seconds": round(elapsed, 3),
        "input_tokens": int(input_length),
        "output_tokens": int(len(generated)),
        "device": str(device),
    }


__all__ = ["MODEL_NAME", "load_slm", "generate_chat"]

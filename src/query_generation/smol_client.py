
"""
SmolLM2-360M-Instruct client for FACTLENS.

Loads the model once and reuses it for claim analysis
and search-query generation.
"""

import time
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM


MODEL_NAME = "HuggingFaceTB/SmolLM2-360M-Instruct"

MAX_INPUT_TOKENS = 2048

_MODEL_CACHE = {
    "tokenizer": None,
    "model": None,
    "device": None,
}


def load_smol_model():
    """Load SmolLM2 once and reuse the cached model."""

    if (
        _MODEL_CACHE["model"] is not None
        and _MODEL_CACHE["tokenizer"] is not None
    ):
        return _MODEL_CACHE["tokenizer"], _MODEL_CACHE["model"]

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    dtype = (
        torch.float16
        if device.type == "cuda"
        else torch.float32
    )

    print(f"[SMOL] Loading {MODEL_NAME}")
    print(f"[SMOL] Device: {device}")
    print(f"[SMOL] Data type: {dtype}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=dtype,
    )

    model.to(device)
    model.eval()

    # Ensure a valid padding token is available.
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    _MODEL_CACHE["tokenizer"] = tokenizer
    _MODEL_CACHE["model"] = model
    _MODEL_CACHE["device"] = device

    print("[SMOL] Model loaded successfully.")

    return tokenizer, model


def generate_smol_text(
    prompt,
    max_new_tokens=256,
    temperature=0.0,
):
    """
    Generate deterministic text using SmolLM2.

    Temperature is retained for compatibility with existing
    callers. Greedy decoding is used, so temperature is ignored.

    Returns:
        Dictionary containing generated text and timing information.
    """

    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("Prompt must be a non-empty string.")

    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be positive.")

    tokenizer, model = load_smol_model()

    device = _MODEL_CACHE["device"]

    messages = [
        {
            "role": "user",
            "content": prompt,
        }
    ]

    formatted_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    # Tokenize without truncation so factual information
    # is never silently removed from the prompt.
    inputs = tokenizer(
        formatted_prompt,
        return_tensors="pt",
        truncation=False,
    )

    input_length = inputs["input_ids"].shape[1]

    if input_length > MAX_INPUT_TOKENS:
        raise ValueError(
            f"Prompt contains {input_length} tokens, exceeding "
            f"the limit of {MAX_INPUT_TOKENS}. "
            "Reduce the prompt without removing factual information."
        )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    if device.type == "cuda":
        torch.cuda.synchronize()

    start_time = time.perf_counter()

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            repetition_penalty=1.05,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    if device.type == "cuda":
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - start_time

    # Decode only newly generated tokens.
    generated_tokens = outputs[0][input_length:]

    generated_text = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True,
    ).strip()

    return {
        "text": generated_text,
        "generation_time_seconds": round(elapsed, 3),
        "input_tokens": input_length,
        "output_tokens": len(generated_tokens),
        "device": str(device),
    }

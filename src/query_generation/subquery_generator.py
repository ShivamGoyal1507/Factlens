
import json
import re

from src.query_generation.complexity import get_max_queries
from src.query_generation.prompts import QUERY_GENERATION_PROMPT
from src.query_generation.validator import (
    is_valid_query,
    is_grounded_query,
    remove_duplicate_queries,
)


def _parse_json_queries(response):
    if not isinstance(response, str):
        return []

    response = response.strip()

    # Locate the JSON object if the model adds surrounding text.
    start = response.find("{")
    end = response.rfind("}")

    if start < 0 or end <= start:
        return []

    try:
        data = json.loads(response[start:end + 1])
    except json.JSONDecodeError:
        return []

    if not isinstance(data, dict):
        return []

    queries = data.get("queries")

    if not isinstance(queries, list):
        return []

    return [
        item.strip()
        for item in queries
        if isinstance(item, str) and item.strip()
    ]


def generate_subqueries(claim, analysis, complexity):
    """
    Generate search queries using FLAN-T5-Base.

    Stage 2 receives the original claim and the validated
    Stage 1 analysis.
    """

    if not isinstance(claim, str) or not claim.strip():
        return []

    if not isinstance(analysis, dict):
        return []

    max_queries = get_max_queries(complexity)

    if max_queries < 1:
        return []

    from src.query_generation.generator import (
        _load_flant5_model,
        _MODEL_CACHE,
    )

    loaded = _load_flant5_model()

    if loaded is None:
        return []

    tokenizer, model = loaded

    import torch

    prompt = QUERY_GENERATION_PROMPT.format(
        claim=claim,
        analysis=json.dumps(analysis, ensure_ascii=False),
        complexity=complexity,
        max_queries=max_queries,
    )

    device = _MODEL_CACHE["device"]

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=False,
    )

    input_length = inputs["input_ids"].shape[1]

    if input_length > 512:
        raise ValueError(
            f"Query-generation prompt has {input_length} tokens; "
            "maximum is 512. Shorten the prompt safely."
        )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=False,
            num_beams=4,
            num_return_sequences=1,
            repetition_penalty=1.1,
            early_stopping=True,
        )

    response = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True,
    )

    candidates = _parse_json_queries(response)

    valid = []

    
    for query in candidates:
        if not is_valid_query(query):
            continue

        if not is_grounded_query(query, claim, analysis):
            continue

        valid.append(query)


    return remove_duplicate_queries(valid)[:max_queries]

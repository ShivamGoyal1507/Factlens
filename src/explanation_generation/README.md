# Verdict Explainer

A lightweight fact-checking explanation generator using **Google FLAN-T5-base**. It generates concise, natural-language explanations for a given claim and its given verdict using the provided evidence.

## Features

- Uses FLAN-T5-base for text generation.
- Accepts one of three verdicts: `Supported`, `Refuted`, and `Not Enough Information`.
- Uses Pydantic for input schema definition and validation.
- Generates explanations in 2–3 sentences.

## Requirements

```bash
pip install torch transformers pydantic sentencepiece accelerate
```

## Usage

```python
from explainer import VerdictExplainer

explainer = VerdictExplainer()

claim = "The Eiffel Tower is located in Berlin."
verdict = "Refuted"

evidences = [
    "The Eiffel Tower is located in Paris, France.",
    "Berlin is the capital of Germany."
]

explanation = explainer.explain(claim, verdict, evidences)

print(explanation)
```

## Input

- `claim`: Claim being verified (`str`).
- `verdict`: Verification result (`str`). One of ["Supported", "Refuted", "Not Enough Information"].
- `evidences`: List of evidence texts (`list[str]`).

## Output

A short natural-language explanation of the verdict based on the supplied evidence.

## Model

[Google FLAN-T5-base](https://huggingface.co/google/flan-t5-base)

The pretrained model is downloaded automatically on first initialization. No additional fine-tuning is required.

"""FEVER claim verification using the trained DeBERTa LoRA adapter."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .config import (
    BASE_MODEL_DIR,
    DEVICE,
    EXPORTED_ADAPTER_DIR,
    ID2LABEL,
    INFERENCE_BATCH_SIZE,
    MAX_LENGTH,
    MODEL_NAME,
    NUM_LABELS,
    TRAINED_MODEL_DIR,
)


class ClaimVerifier:
    """Load and reuse the trained FEVER classifier for claim verification."""

    _EVIDENCE_FIELDS = ("text", "content", "evidence", "snippet")

    def __init__(
        self,
        model_dir: Path | str | None = None,
        device: str | None = None,
        batch_size: int = INFERENCE_BATCH_SIZE,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")

        self.device_name = device or DEVICE
        self.batch_size = batch_size
        self.model_dir = self._resolve_model_dir(model_dir)
        self._torch, auto_tokenizer, auto_model, peft_model = (
            self._import_inference_dependencies()
        )

        tokenizer_source = self._validate_tokenizer_files(self.model_dir)
        base_model_source = (
            BASE_MODEL_DIR if BASE_MODEL_DIR.is_dir() else MODEL_NAME
        )

        try:
            self.tokenizer = auto_tokenizer.from_pretrained(
                str(tokenizer_source),
                local_files_only=True,
            )
            base_model = auto_model.from_pretrained(str(base_model_source))
            self.model = peft_model.from_pretrained(
                base_model,
                str(self.model_dir),
                is_trainable=False,
            )
            self.model.to(self.torch.device(self.device_name))
            self.model.eval()
        except (OSError, RuntimeError, ValueError) as error:
            raise RuntimeError(
                "Could not load the FEVER base model, tokenizer, or trained "
                f"LoRA adapter from '{self.model_dir}': {error}"
            ) from error

        self._validate_model_output()

    @staticmethod
    def _import_inference_dependencies() -> tuple[Any, Any, Any, Any]:
        try:
            import torch
            from peft import PeftModel
            from transformers import (
                AutoModelForSequenceClassification,
                AutoTokenizer,
            )
        except ImportError as error:
            raise ImportError(
                "ClaimVerifier requires PyTorch, Transformers, and PEFT. "
                "Install the project's inference dependencies before loading "
                "the trained verifier."
            ) from error
        return torch, AutoTokenizer, AutoModelForSequenceClassification, PeftModel

    @staticmethod
    def _resolve_model_dir(model_dir: Path | str | None) -> Path:
        if model_dir is not None:
            resolved = Path(model_dir).expanduser().resolve()
            candidates = (resolved,)
        else:
            candidates = (TRAINED_MODEL_DIR, EXPORTED_ADAPTER_DIR)

        for candidate in candidates:
            if candidate.is_dir():
                return candidate

        searched = ", ".join(str(candidate) for candidate in candidates)
        raise FileNotFoundError(
            "No trained FEVER adapter directory was found. Checked: "
            f"{searched}"
        )

    @staticmethod
    def _validate_tokenizer_files(model_dir: Path) -> Path:
        required = ("adapter_config.json", "adapter_model.safetensors")
        missing = [name for name in required if not (model_dir / name).is_file()]
        if missing:
            raise FileNotFoundError(
                f"Trained adapter directory '{model_dir}' is missing: "
                + ", ".join(missing)
            )

        try:
            adapter_config = json.loads(
                (model_dir / "adapter_config.json").read_text(
                    encoding="utf-8"
                )
            )
        except (OSError, json.JSONDecodeError) as error:
            raise RuntimeError(
                f"Could not read adapter configuration in '{model_dir}': {error}"
            ) from error

        if not isinstance(adapter_config, dict):
            raise RuntimeError(
                f"Adapter configuration in '{model_dir}' must be a JSON object."
            )
        modules_to_save = adapter_config.get("modules_to_save", [])
        if "classifier" not in modules_to_save:
            raise RuntimeError(
                "The saved adapter does not contain the trained classifier "
                "head in modules_to_save."
            )

        tokenizer_files = ("tokenizer.json", "tokenizer_config.json")
        missing_tokenizer = [
            name for name in tokenizer_files if not (model_dir / name).is_file()
        ]
        if missing_tokenizer:
            raise FileNotFoundError(
                f"Tokenizer files are missing from '{model_dir}': "
                + ", ".join(missing_tokenizer)
            )
        return model_dir

    def _validate_model_output(self) -> None:
        configured_labels = getattr(self.model.config, "num_labels", NUM_LABELS)
        if configured_labels != NUM_LABELS:
            raise RuntimeError(
                f"Loaded model exposes {configured_labels} labels; expected "
                f"{NUM_LABELS}."
            )

    @classmethod
    def _extract_evidence_text(cls, evidence: Any) -> str:
        if isinstance(evidence, (str, bytes)) or not isinstance(
            evidence, Sequence
        ):
            raise TypeError("evidence must be a non-empty list or tuple")
        if not evidence:
            raise ValueError("evidence must contain at least one item")

        texts: list[str] = []
        for index, item in enumerate(evidence):
            text: Any = item if isinstance(item, str) else None
            if isinstance(item, Mapping):
                text = next(
                    (
                        item[field]
                        for field in cls._EVIDENCE_FIELDS
                        if field in item and item[field]
                    ),
                    None,
                )
            elif text is None:
                for field in cls._EVIDENCE_FIELDS:
                    value = getattr(item, field, None)
                    if value:
                        text = value
                        break

            if not isinstance(text, str) or not text.strip():
                raise ValueError(
                    f"evidence item {index} has no non-empty text field"
                )
            texts.append(text.strip())

        return " ".join(texts)

    @classmethod
    def _build_input(cls, claim: Any, evidence: Any) -> str:
        if not isinstance(claim, str) or not claim.strip():
            raise ValueError("claim must be a non-empty string")
        evidence_text = cls._extract_evidence_text(evidence)
        return f"Claim: {claim.strip()}\nEvidence: {evidence_text}"

    def _predict(self, inputs: list[tuple[str, str]]) -> list[dict[str, Any]]:
        texts = [self._build_input(claim, evidence) for claim, evidence in inputs]
        encoded = self.tokenizer(
            texts,
            max_length=MAX_LENGTH,
            truncation=True,
            padding=True,
            return_tensors="pt",
        )
        encoded = {
            key: value.to(self.torch.device(self.device_name))
            for key, value in encoded.items()
        }

        with self.torch.inference_mode():
            logits = self.model(**encoded).logits
            probabilities = self.torch.softmax(logits, dim=-1).cpu().tolist()

        results = []
        for (claim, _), row in zip(inputs, probabilities):
            if len(row) != NUM_LABELS:
                raise RuntimeError(
                    f"Model returned {len(row)} class probabilities; expected "
                    f"{NUM_LABELS}."
                )
            predicted_id = max(range(NUM_LABELS), key=row.__getitem__)
            class_probabilities = {
                ID2LABEL[index]: float(row[index]) for index in range(NUM_LABELS)
            }
            results.append(
                {
                    "claim": claim,
                    "verdict": ID2LABEL[predicted_id],
                    "confidence": float(row[predicted_id]),
                    "probabilities": class_probabilities,
                }
            )
        return results

    def verify(self, claim: str, evidence: Any) -> dict[str, Any]:
        """Return the model verdict and softmax probabilities for one claim."""
        return self._predict([(claim, evidence)])[0]

    def verify_batch(
        self, items: Sequence[Mapping[str, Any]]
    ) -> list[dict[str, Any]]:
        """Verify multiple ``{"claim": ..., "evidence": ...}`` items."""
        if isinstance(items, (str, bytes)) or not isinstance(items, Sequence):
            raise TypeError("items must be a sequence of mappings")
        if not items:
            return []

        normalized: list[tuple[str, Any]] = []
        for index, item in enumerate(items):
            if not isinstance(item, Mapping):
                raise TypeError(f"batch item {index} must be a mapping")
            if "claim" not in item or "evidence" not in item:
                raise ValueError(
                    f"batch item {index} must contain claim and evidence"
                )
            normalized.append((item["claim"], item["evidence"]))

        results: list[dict[str, Any]] = []
        for start in range(0, len(normalized), self.batch_size):
            results.extend(self._predict(normalized[start : start + self.batch_size]))
        return results

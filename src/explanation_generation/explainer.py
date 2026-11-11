import re
from typing import Literal

import torch
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"


class Evidence(BaseModel):
    text: str


class VerdictInput(BaseModel):
    claim: str
    verdict: Literal["Supported", "Refuted", "Not Enough Information"]
    evidences: list[Evidence] = Field(min_length=1)


_VERDICT_GUIDANCE = {
    "Supported": (
        "State the specific fact in the evidence that confirms the claim."
    ),
    "Refuted": (
        "State the specific fact in the evidence that contradicts the claim "
        "(for example a different person, place, date or number than the claim states)."
    ),
    "Not Enough Information": (
        "Say what the evidence does cover and what it does not establish about the claim."
    ),
}

SYSTEM_PROMPT = (
    "You explain fact-checking verdicts. The verdict is already decided; do not "
    "change or question it. Use only facts written in the evidence. Never state "
    "anything as true that the evidence does not say. Write plain text, no "
    "markdown, no lists, no headings."
)


class VerdictExplainer:
    """
    Generate a short natural-language explanation for a verdict.

    model_name: Hugging Face chat model. If it is the same model the query
    generator uses (src/query_generation/slm_client.py), that already-loaded
    model is reused instead of loading a second copy.
    """

    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or DEFAULT_MODEL_NAME

        shared = self._shared_query_model(self.model_name)
        if shared is not None:
            self.tokenizer, self.model = shared
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                dtype=torch.float16,
                device_map="auto",
            )

        self.model.eval()

    @staticmethod
    def _shared_query_model(model_name):
        try:
            from src.query_generation import slm_client
        except ImportError:
            return None
        if model_name != slm_client.MODEL_NAME:
            return None
        return slm_client.load_slm()

    def prepare_input(self, claim, verdict, evidences):
        return VerdictInput(
            claim=claim,
            verdict=verdict,
            evidences=[Evidence(text=evidence) for evidence in evidences],
        )

    def create_prompt(self, data: VerdictInput):
        evidence_text = "\n".join(
            f"Evidence {i}: {evidence.text}"
            for i, evidence in enumerate(data.evidences, 1)
        )
        return (
            f"Claim: {data.claim}\n"
            f"Verdict: {data.verdict}\n\n"
            f"{evidence_text}\n\n"
            f"Explain in at most 2 sentences why the claim is "
            f"{data.verdict.lower()}. {_VERDICT_GUIDANCE[data.verdict]} "
            f"Refer to the evidence number you use, e.g. (Evidence 1)."
        )

    @staticmethod
    def _clean(text):
        text = re.sub(r"[*#`]+", "", text)            # strip markdown
        text = re.sub(r"\s+", " ", text).strip()
        # Keep at most two complete sentences.
        sentences = re.split(r"(?<=[.!?])\s+", text)
        kept = " ".join(sentences[:2]).strip()
        if kept and kept[-1] not in ".!?":
            kept += "."
        return kept

    def explain(self, claim, verdict, evidences):
        """
        Generate a concise explanation for the given claim, verdict and evidence.

        Args:
            claim (str): The claim to be explained.
            verdict (str): "Supported", "Refuted" or "Not Enough Information".
            evidences (list[str]): Evidence texts; pass the passages that
                decided the verdict first.

        Returns:
            str: At most two sentences of plain text.
        """
        data = self.prepare_input(claim, verdict, evidences)

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": self.create_prompt(data)},
        ]

        inputs = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        ).to(self.model.device)

        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=110,
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        generated_tokens = outputs[0][inputs["input_ids"].shape[-1]:]
        explanation = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        return self._clean(explanation)

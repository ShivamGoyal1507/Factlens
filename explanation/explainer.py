import torch
from pydantic import BaseModel, Field
from typing import Literal
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

class Evidence(BaseModel):
    text: str
    # label: Literal["supporting", "opposing"]

class VerdictInput(BaseModel):
    claim: str
    verdict: Literal["Supported", "Refuted", "Not Enough Information"]
    evidences: list[Evidence] = Field(min_length=1)


class VerdictExplainer:

    def __init__(self):
        self.model_name = "google/flan-t5-base"

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            self.model_name,
            device_map="auto"
        )

        self.model.eval()

    def prepare_input(self, claim, verdict, evidences):
        evidence_objects = [Evidence(text=evidence) for evidence in evidences]

        data = VerdictInput(
            claim=claim,
            verdict=verdict,
            evidences=evidence_objects
        )

        return data

    def create_prompt(self, data: VerdictInput):
        evidence_text = ""

        # evidence_text = "\n".join(f"Evidence {i} ({evidence.label}): {evidence.text}" for i, evidence in enumerate(data.evidences, 1))
        evidence_text = "\n".join(f"Evidence {i}: {evidence.text}" for i, evidence in enumerate(data.evidences, 1))

        # prompt = f"""
        #     You are a fact-checking explanation assistant.

        #     Claim: {data.claim}

        #     Verdict: {data.verdict}

        #     Evidence:
        #     {evidence_text}

        #     Explain why the given verdict was reached in 2-3 simple sentences.

        #     Instructions:
        #     1. Use only the provided evidence.
        #     2. Consider both supporting and opposing evidence.
        #     3. Explain how the evidence relates to the claim.
        #     4. If evidence conflicts, acknowledge the conflict.
        #     5. If the verdict is Not Enough Information, explain why the evidence is insufficient to establish the claim.
        #     6. Do not introduce external facts or invent information.
        #     7. Do not change the given verdict.
        #     8. Use simple, neutral language.

        #     Return only the explanation.
        #     """
        # return prompt
    
        prompt = f"""
            Explain the following fact-checking result.

            Claim: {data.claim}
            Verdict: {data.verdict}

            Evidence:
            {evidence_text}

            Write a short explanation (2-3 sentences) describing
            how the evidence supports the verdict.

            Use only the evidence provided. 
            If evidence conflicts, acknowledge the conflict.
            Do not introduce external information.

            Explanation:
            """

        return prompt

    def explain(self, claim, verdict, evidences):
        """
        Generate a concise explanation for the given claim, verdict, and evidence.

        Args:
            claim (str): The claim to be explained.
            verdict (str): One of ["Supported", "Refuted", "Not Enough Information"]. The verdict associated with the claim.
            evidences (list[str]): A list of evidence texts supporting or refuting the claim.

        Returns:
            str: A concise explanation of the verdict based on the provided evidence.
        """
        # convert input to structured data
        data = self.prepare_input(claim, verdict, evidences)

        # generate prompt
        prompt = self.create_prompt(data)

        # tokenize input
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512
        )

        # generate explanation
        inputs = {
            key: value.to(self.model.device)
            for key, value in inputs.items()
        }
        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=100,
                # min_new_tokens=20,
                num_beams=4,
                no_repeat_ngram_size=3,
                early_stopping=True
            )

        # decode output
        explanation = self.tokenizer.decode(
            outputs[0],
            skip_special_tokens=True
        )

        return explanation.strip()
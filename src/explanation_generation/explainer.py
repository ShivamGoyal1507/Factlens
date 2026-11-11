import torch
from pydantic import BaseModel, Field
from typing import Literal
from transformers import AutoTokenizer, AutoModelForCausalLM

class Evidence(BaseModel):
    text: str
    # label: Literal["supporting", "opposing"]

class VerdictInput(BaseModel):
    claim: str
    verdict: Literal["Supported", "Refuted", "Not Enough Information"]
    evidences: list[Evidence] = Field(min_length=1)


class VerdictExplainer:

    def __init__(self):
        self.model_name = "Qwen/Qwen2.5-0.5B-Instruct"

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
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

        prompt = f"""
            Fact-checking task:

            Claim: {data.claim}

            Verdict: {data.verdict}

            Evidence:
            {evidence_text}

            Explain the verdict in 2 sentences.

            'For SUPPORTED': Explain what evidence establishes the claim.

            'For REFUTED': Explain what evidence contradicts the claim.

            'For NOT ENOUGH INFORMATION': Explain why the evidence does not establish
            whether the claim is true or false. Identify what cannot be concluded
            from the available evidence.

            Do not invent facts. Do not repeat the instructions.
            Write only the explanation.

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

        print(next(self.model.parameters()).dtype)
        print(f"Model memory: {self.model.get_memory_footprint() / 1024**2:.2f} MB")

        messages = [
            {
                "role": "system",
                "content": "You are an assistant that explains fact-checking results. Use only the provided evidence."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]

        inputs = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True
        ).to(self.model.device)

        self.model.generation_config.temperature = None
        self.model.generation_config.top_p = None
        self.model.generation_config.top_k = None

        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=120,
                do_sample=False
            )

        generated_tokens = outputs[0][inputs["input_ids"].shape[-1]:]

        explanation = self.tokenizer.decode(
            generated_tokens,
            skip_special_tokens=True
        )

        # # tokenize input
        # inputs = self.tokenizer(
        #     prompt,
        #     return_tensors="pt",
        #     truncation=True,
        #     max_length=512
        # )

        # # generate explanation
        # inputs = {
        #     key: value.to(self.model.device)
        #     for key, value in inputs.items()
        # }
        # with torch.inference_mode():
        #     outputs = self.model.generate(
        #         **inputs,
        #         max_new_tokens=200,
        #         # min_new_tokens=20,
        #         num_beams=4,
        #         repetition_penalty=1.2,
        #         no_repeat_ngram_size=3,
        #         early_stopping=True
        #     )
        
        # print(next(self.model.parameters()).dtype)
        # print(f"Model memory: {self.model.get_memory_footprint() / 1024**2:.2f} MB")

        # # decode output
        # explanation = self.tokenizer.decode(
        #     outputs[0],
        #     skip_special_tokens=True
        # )

        return explanation.strip()
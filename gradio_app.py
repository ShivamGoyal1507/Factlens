"""Basic Gradio interface for the FACTLENS claim-verification pipeline."""

from __future__ import annotations

import json
import os
from typing import Any

from main import run_pipeline


def _format_evidence(evidence: list[dict[str, Any]]) -> str:
    if not evidence:
        return "No evidence was selected."

    sections = []
    for index, item in enumerate(evidence, start=1):
        title = item.get("title") or "Untitled source"
        source = item.get("source") or "Unknown source"
        url = item.get("url") or ""
        link = f" ([source]({url}))" if url else ""
        text = item.get("text") or ""
        sections.append(
            f"### {index}. {title}\n"
            f"**Source:** {source}{link}\n\n"
            f"{text}"
        )
    return "\n\n".join(sections)


def verify_claim(claim: str) -> tuple[str, str, str, str, str]:
    """Run FACTLENS and format its result for the Gradio outputs."""
    if not isinstance(claim, str) or not claim.strip():
        return (
            "Please enter a claim.",
            "",
            "",
            "",
            "",
        )

    try:
        result = run_pipeline(claim.strip())
    except Exception as error:
        return (
            f"Pipeline error: {error}",
            "",
            "",
            "",
            "",
        )

    verification = result["verification"]
    probabilities = verification["probabilities"]
    probability_text = "\n".join(
        f"- **{label}:** {float(probability):.4f}"
        for label, probability in probabilities.items()
    )
    counts = json.dumps(result["counts"], indent=2)
    verdict = (
        f"## {verification['verdict']}\n\n"
        f"**Confidence:** {float(verification['confidence']):.4f}\n\n"
        f"{probability_text}"
    )
    return (
        verdict,
        result.get("explanation") or "Explanation was not generated.",
        _format_evidence(result["evidence"]),
        f"```json\n{counts}\n```",
        "",
    )


def create_demo() -> Any:
    """Build the FACTLENS Gradio application."""
    try:
        import gradio as gr
    except ModuleNotFoundError as error:
        if error.name != "gradio":
            raise
        raise RuntimeError(
            "Gradio is not installed. Install the optional UI dependency "
            "with 'python -m pip install --no-cache-dir "
            "-r requirements-ui.txt'. If disk quota is exhausted, clear "
            "pip's cache first with 'python -m pip cache purge'."
        ) from error

    with gr.Blocks(title="FACTLENS Claim Verification") as demo:
        gr.Markdown(
            "# FACTLENS Claim Verification\n"
            "Enter a claim to retrieve evidence, rank it, verify the claim, "
            "and generate an explanation."
        )

        claim = gr.Textbox(
            label="Claim",
            placeholder="Example: Python is an interpreted programming language.",
            lines=3,
        )
        verify = gr.Button("Verify claim", variant="primary")
        status = gr.Markdown(label="Status")

        with gr.Row():
            verdict = gr.Markdown(label="Verdict")
            explanation = gr.Markdown(label="Explanation")

        evidence = gr.Markdown(label="Selected evidence")
        counts = gr.Markdown(label="Pipeline counts")

        verify.click(
            fn=verify_claim,
            inputs=claim,
            outputs=[verdict, explanation, evidence, counts, status],
        )
        claim.submit(
            fn=verify_claim,
            inputs=claim,
            outputs=[verdict, explanation, evidence, counts, status],
        )

    return demo


if __name__ == "__main__":
    create_demo().launch(
        server_name=os.getenv("GRADIO_SERVER_NAME", "127.0.0.1"),
        server_port=int(os.getenv("GRADIO_SERVER_PORT", "7860")),
        share=os.getenv("GRADIO_SHARE", "").lower() == "true",
    )

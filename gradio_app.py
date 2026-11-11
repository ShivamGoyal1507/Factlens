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
            "## Ready when you are\nEnter a claim above to start verification.",
            "",
            "",
            "",
            "Please enter a claim to verify.",
        )

    try:
        result = run_pipeline(claim.strip())
    except Exception as error:
        return (
            "## Verification unavailable",
            "",
            "",
            "",
            f"Pipeline error: {error}",
        )

    verification = result["verification"]
    probabilities = verification["probabilities"]
    probability_text = "\n".join(
        f"- **{label}:** {float(probability):.4f}"
        for label, probability in probabilities.items()
    )
    counts = json.dumps(result["counts"], indent=2)
    verdict = (
        f"### {verification['verdict']}\n"
        f"**Confidence** · {float(verification['confidence']):.1%}\n\n"
        f"{probability_text}"
    )
    return (
        verdict,
        result.get("explanation") or "Explanation was not generated.",
        _format_evidence(result["evidence"]),
        f"```json\n{counts}\n```",
        "Verification complete · Evidence ranked and analyzed.",
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

    css = """
    :root {
        --factlens-ink: #173332;
        --factlens-muted: #647a77;
        --factlens-blue: #0f766e;
        --factlens-surface: #ffffff;
        --factlens-border: #dce8e5;
    }
    .gradio-container {
        max-width: 1160px !important;
        margin: 0 auto !important;
        background: #f3f8f6;
        color: var(--factlens-ink) !important;
    }
    .gradio-container, .gradio-container * { color-scheme: light; }
    .gradio-container .gr-block,
    .gradio-container .gr-box,
    .gradio-container .gr-panel,
    .gradio-container .gr-form {
        background: var(--factlens-surface);
        border-color: var(--factlens-border);
    }
    .gradio-container label,
    .gradio-container .label-wrap,
    .gradio-container .block-label,
    .gradio-container .wrap,
    .gradio-container .prose,
    .gradio-container .markdown {
        color: var(--factlens-ink) !important;
    }
    .gradio-container textarea,
    .gradio-container input {
        color: var(--factlens-ink) !important;
        background: #ffffff !important;
        border-color: #b9cfca !important;
    }
    .gradio-container textarea::placeholder,
    .gradio-container input::placeholder {
        color: #718783 !important;
        opacity: 1;
    }
    .gradio-container .gr-button {
        min-height: 44px;
        border: 1px solid #c3d8d2 !important;
        color: #173332 !important;
        background: #eaf3f0 !important;
        font-weight: 700;
    }
    .gradio-container .gr-button:hover {
        background: #dcece7 !important;
        border-color: #9bbdb4 !important;
    }
    .gradio-container .gr-button.primary {
        color: #ffffff !important;
        background: #0f766e !important;
        border-color: #0f766e !important;
    }
    .gradio-container .gr-button.primary:hover {
        background: #0b5f59 !important;
        border-color: #0b5f59 !important;
    }
    .gradio-container .secondary {
        color: #173332 !important;
    }
    .gradio-container .examples,
    .gradio-container .examples-table {
        color: var(--factlens-ink) !important;
    }
    .gradio-container .examples-table button {
        color: #0f766e !important;
        background: #edf7f3 !important;
        border-color: #c8e2d9 !important;
    }
    .gradio-container .accordion,
    .gradio-container details,
    .gradio-container details > summary {
        color: var(--factlens-ink) !important;
        background: #ffffff !important;
        border-color: var(--factlens-border) !important;
    }
    .gradio-container .icon-button {
        color: #526d68 !important;
    }
    .app-shell { gap: 22px; }
    .gradio-container .hero {
        padding: 38px 42px 34px;
        border-radius: 24px;
        color: #ffffff !important;
        background: linear-gradient(120deg, #123331 0%, #155e59 56%, #0f766e 100%) !important;
        border: 0 !important;
        box-shadow: 0 16px 38px rgba(15, 91, 82, 0.18);
    }
    .gradio-container .hero .prose,
    .gradio-container .hero h1,
    .gradio-container .hero p {
        color: #ffffff !important;
    }
    .gradio-container .hero h1 { margin: 0 0 10px; font-size: 40px; letter-spacing: -1px; }
    .gradio-container .hero p { margin: 0; color: #d8f0e9 !important; font-size: 16px; max-width: 680px; }
    .gradio-container .hero .eyebrow {
        color: #9fe0cf;
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 1.8px;
        text-transform: uppercase;
    }
    .gradio-container .input-card,
    .gradio-container .result-card {
        background: var(--factlens-surface);
        border: 1px solid var(--factlens-border);
        border-radius: 18px;
        padding: 22px;
        box-shadow: 0 8px 24px rgba(23, 51, 50, 0.06);
    }
    .gradio-container .input-card textarea { font-size: 16px !important; }
    .section-label {
        color: var(--factlens-ink);
        font-size: 17px;
        font-weight: 700;
        margin-bottom: 4px;
    }
    .section-help { color: var(--factlens-muted); font-size: 13px; }
    .status {
        color: var(--factlens-muted) !important;
        min-height: 24px;
        opacity: 1 !important;
    }
    .gradio-container .status,
    .gradio-container .status .prose,
    .gradio-container .status p,
    .gradio-container .status span {
        color: var(--factlens-muted) !important;
        opacity: 1 !important;
    }
    .gradio-container .result-card strong,
    .gradio-container .result-card b {
        color: var(--factlens-ink) !important;
        opacity: 1 !important;
    }
    .gradio-container .result-card h3 { margin-top: 0; color: var(--factlens-ink); }
    .gradio-container h2,
    .gradio-container h3,
    .gradio-container .prose h2,
    .gradio-container .prose h3,
    .gradio-container .result-card h2,
    .gradio-container .result-card h3 {
        color: var(--factlens-ink) !important;
    }
    .gradio-container .result-card p,
    .gradio-container .result-card li { color: #526d68; line-height: 1.65; }
    .gradio-container .evidence-card { padding-top: 8px; }
    .gradio-container .evidence-card h3 { color: var(--factlens-ink); }
    .gradio-container .footer { color: var(--factlens-muted); text-align: center; font-size: 12px; }
    @media (max-width: 700px) {
        .gradio-container .hero {
            padding: 28px 26px 26px;
        }
        .gradio-container .hero h1 {
            font-size: 34px;
        }
    }
    """

    with gr.Blocks(
        title="FACTLENS · Claim verification",
        theme=gr.themes.Soft(
            primary_hue="teal",
            secondary_hue="green",
            neutral_hue="slate",
            font=gr.themes.GoogleFont("DM Sans"),
        ),
        css=css,
    ) as demo:
        with gr.Column(elem_classes="app-shell"):
            gr.Markdown(
                '<div class="hero">'
                '<div class="eyebrow">Evidence intelligence</div>'
                "<h1>Know what to believe.</h1>"
                "<p>FACTLENS checks a claim against ranked evidence and "
                "explains the reasoning behind its verdict.</p>"
                "</div>",
                elem_classes="hero",
            )

            with gr.Column(elem_classes="input-card"):
                gr.Markdown(
                    '<div class="section-label">What would you like to verify?</div>'
                    '<div class="section-help">Paste a factual claim and we’ll '
                    "search, rank, and analyze the strongest available sources.</div>"
                )
                claim = gr.Textbox(
                    show_label=False,
                    placeholder="e.g. Rahul Gandhi has been chief justice of india in the past.",
                    lines=3,
                    max_lines=6,
                    autofocus=True,
                )
                with gr.Row():
                    verify = gr.Button("Verify claim", variant="primary", scale=3)
                    clear = gr.Button("Clear", variant="secondary", scale=1)
                gr.Examples(
                    examples=[
                        ["Rahul Gandhi has been chief justice of india in the past."],
                        ["Rahul Gandhi is the leader of opposition."],
                        ["Jensen Huang cheated on his wife."],
                    ],
                    inputs=claim,
                    label="Try an example",
                )
                status = gr.Markdown(
                    "Ready to investigate a claim.",
                    elem_classes="status",
                )

            gr.Markdown("## Verification report")
            with gr.Row():
                with gr.Column(elem_classes="result-card"):
                    gr.Markdown("### Verdict")
                    verdict = gr.Markdown(
                        "## Ready when you are\nEnter a claim above to start verification."
                    )
                with gr.Column(elem_classes="result-card"):
                    gr.Markdown("### Explanation")
                    explanation = gr.Markdown(
                        "The reasoning behind the verdict will appear here."
                    )

            with gr.Column(elem_classes=["result-card", "evidence-card"]):
                gr.Markdown("### Selected evidence")
                evidence = gr.Markdown(
                    "The highest-ranked sources supporting the analysis will appear here."
                )

            with gr.Accordion("Pipeline details", open=False):
                counts = gr.Markdown(
                    "Retrieval and ranking counts will appear here after verification."
                )

            gr.Markdown(
                "FACTLENS · Retrieval, ranking, verification, and explanation in one place.",
                elem_classes="footer",
            )

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
        clear.click(
            fn=lambda: (
                "",
                "## Ready when you are\nEnter a claim above to start verification.",
                "The reasoning behind the verdict will appear here.",
                "The highest-ranked sources supporting the analysis will appear here.",
                "Retrieval and ranking counts will appear here after verification.",
                "Ready to investigate a claim.",
            ),
            outputs=[claim, verdict, explanation, evidence, counts, status],
        )

    return demo


if __name__ == "__main__":
    create_demo().launch(
        server_name=os.getenv("GRADIO_SERVER_NAME", "127.0.0.1"),
        server_port=int(os.getenv("GRADIO_SERVER_PORT", "7860")),
        share=os.getenv("GRADIO_SHARE", "").lower() == "true",
    )

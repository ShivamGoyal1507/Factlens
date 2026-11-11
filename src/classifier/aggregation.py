"""
Combine per-passage verifier predictions into one claim verdict.

The FEVER classifier was trained on (claim, short evidence) pairs, so the
pipeline verifies the claim against each selected passage separately
instead of concatenating them (which truncates at 256 tokens and hides the
decisive passage). This module turns those per-passage predictions into a
single verdict.

Rules
-----
* Each passage i gets a weight w_i by its evidence rank (better-ranked
  passages count more).
* A verdict needs at least one clear passage: max_i w_i * P_i(label) must
  reach `decision_threshold`, otherwise -> NEI. Generic background passages
  (NEI) do not dilute a clear one, so "Narendra Modi is the current prime
  minister" alone refutes "Rahul Gandhi is prime minister".
* Weighted vote: mass(label) = sum_i w_i * P_i(label). The label with more
  mass wins. If both sides have a clear passage and neither has at least
  `conflict_ratio` times the other's mass -> NEI (conflicting evidence).
  This stops one off-topic passage (e.g. a biography the FEVER model reads
  as REFUTES because it doesn't mention the claim) from outvoting several
  passages that state the fact directly.
"""

from __future__ import annotations

from typing import Any, Sequence

LABELS = ("SUPPORTS", "REFUTES", "NEI")


def rank_weights(count: int, decay: float = 0.1, floor: float = 0.5) -> list[float]:
    """1.0 for the top passage, decreasing by `decay` per rank, never below `floor`."""
    return [max(floor, 1.0 - decay * index) for index in range(count)]


def aggregate_predictions(
    predictions: Sequence[dict[str, Any]],
    weights: Sequence[float] | None = None,
    decision_threshold: float = 0.5,
    conflict_ratio: float = 1.5,
) -> dict[str, Any]:
    """
    Args:
        predictions: per-passage results, each with "probabilities"
            {"SUPPORTS": p, "REFUTES": p, "NEI": p}, ordered by evidence rank.
        weights: optional per-passage weights (default: rank_weights).

    Returns:
        {"verdict", "confidence", "probabilities", "decisive_evidence"}
        where decisive_evidence lists the passage indices behind the verdict.
    """
    if not predictions:
        return {
            "verdict": "NEI",
            "confidence": 1.0,
            "probabilities": {"SUPPORTS": 0.0, "REFUTES": 0.0, "NEI": 1.0},
            "decisive_evidence": [],
        }

    if weights is None:
        weights = rank_weights(len(predictions))

    peak = {}
    mass = {}
    for label in ("SUPPORTS", "REFUTES"):
        weighted = [
            weights[i] * float(p["probabilities"].get(label, 0.0))
            for i, p in enumerate(predictions)
        ]
        peak[label] = max(weighted)
        mass[label] = sum(weighted)

    clear = {label for label in peak if peak[label] >= decision_threshold}

    if not clear:
        verdict = "NEI"
    elif len(clear) == 1:
        verdict = next(iter(clear))
    else:
        winner = "SUPPORTS" if mass["SUPPORTS"] >= mass["REFUTES"] else "REFUTES"
        loser = "REFUTES" if winner == "SUPPORTS" else "SUPPORTS"
        if mass[winner] >= conflict_ratio * mass[loser]:
            verdict = winner
        else:
            verdict = "NEI"  # strong evidence on both sides

    total_mass = sum(weights[: len(predictions)]) or 1.0
    support = mass["SUPPORTS"] / total_mass
    refute = mass["REFUTES"] / total_mass
    nei = max(0.0, 1.0 - support - refute)
    total = support + refute + nei or 1.0
    probabilities = {
        "SUPPORTS": support / total,
        "REFUTES": refute / total,
        "NEI": nei / total,
    }

    if verdict == "NEI":
        decisive = []
    else:
        decisive = [
            i for i, p in enumerate(predictions)
            if p.get("verdict") == verdict
            and float(p["probabilities"].get(verdict, 0.0)) >= decision_threshold
        ]

    # Confidence = how sure the model is in the strongest passage behind the
    # verdict (the mass shares above are diluted by background passages).
    if decisive:
        confidence = max(
            float(predictions[i]["probabilities"][verdict]) for i in decisive
        )
    else:
        confidence = 1.0 - max(peak.values()) if not clear else 0.5

    return {
        "verdict": verdict,
        "confidence": confidence,
        "probabilities": probabilities,
        "decisive_evidence": decisive,
    }


__all__ = ["aggregate_predictions", "rank_weights", "LABELS"]

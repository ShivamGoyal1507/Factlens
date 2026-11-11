"""
Benchmark the live retrieval + ranking pipeline on FEVER validation claims.

For a stratified sample of claims from the verifier's validation set
(experiments/classifier/output/fever_deberta_lora_final/validation_predictions.csv)
this runs main.run_pipeline (no explanation) and reports:

* gold recall@k  - share of SUPPORTS/REFUTES claims for which one of the
                   selected evidence passages contains a gold FEVER sentence
                   (fuzzy token match; live Wikipedia text differs from the
                   2017 FEVER dump, so exact matching is impossible);
* accuracy       - end-to-end verdict accuracy, overall and per label.

Usage (project root, factlens env):

    python scripts/eval_fever_pipeline.py --per-label 20
    python scripts/eval_fever_pipeline.py --per-label 20 --with-query-generation
    python scripts/eval_fever_pipeline.py --per-label 20 --out results.jsonl
"""

from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import torch  # noqa: E402,F401  (must precede spacy / sentence-transformers)

import argparse  # noqa: E402
import csv  # noqa: E402
import json  # noqa: E402
import random  # noqa: E402
import re  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from collections import Counter, defaultdict  # noqa: E402
from pathlib import Path  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from main import run_pipeline  # noqa: E402

VALIDATION_CSV = (
    PROJECT_ROOT
    / "experiments/classifier/output/fever_deberta_lora_final/validation_predictions.csv"
)

_FEVER_TOKENS = {
    "-LRB-": "(", "-RRB-": ")", "-LSB-": "[", "-RSB-": "]",
    "-LCB-": "{", "-RCB-": "}", "-COLON-": ":", "``": '"', "''": '"',
}
_STOP = {
    "the", "a", "an", "of", "in", "and", "is", "was", "to", "for", "on",
    "by", "as", "at", "with", "it", "he", "she", "his", "her", "its", "be",
    "are", "were", "from", "that", "which", "or", "has", "had", "this",
}


def detokenize_fever(text: str) -> str:
    for token, char in _FEVER_TOKENS.items():
        text = text.replace(token, char)
    return text


def content_tokens(text: str) -> set[str]:
    return {
        t for t in re.findall(r"[a-z0-9]+", detokenize_fever(text).lower())
        if t not in _STOP
    }


def gold_match(gold: str, passage: str, threshold: float) -> bool:
    """True when any sentence of the gold group is (fuzzily) in the passage."""
    passage_tokens = content_tokens(passage)
    for sentence in re.split(r"(?<= \.) ", gold):
        gold_tokens = content_tokens(sentence)
        if len(gold_tokens) >= 3 and (
            len(gold_tokens & passage_tokens) / len(gold_tokens) >= threshold
        ):
            return True
    return False


def load_claims(per_label: int, seed: int) -> list[dict]:
    by_id: dict[str, dict] = {}
    with VALIDATION_CSV.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            item = by_id.setdefault(
                row["id"],
                {"id": row["id"], "claim": row["claim"],
                 "label": row["gold_label"], "gold": []},
            )
            if row["evidence_source"] == "gold":
                item["gold"].append(row["evidence"])

    groups = defaultdict(list)
    for item in by_id.values():
        groups[item["label"]].append(item)

    rng = random.Random(seed)
    sample = []
    for label in ("SUPPORTS", "REFUTES", "NEI"):
        items = sorted(groups[label], key=lambda x: int(x["id"]))
        sample.extend(rng.sample(items, min(per_label, len(items))))
    rng.shuffle(sample)
    return sample


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--per-label", type=int, default=20)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--match-threshold", type=float, default=0.7)
    parser.add_argument("--with-query-generation", action="store_true")
    parser.add_argument("--out", default=None, help="Write per-claim JSONL here")
    args = parser.parse_args()

    claims = load_claims(args.per_label, args.seed)
    out = open(args.out, "w", encoding="utf-8") if args.out else None

    correct = Counter()
    total = Counter()
    recall_hits = 0
    positives = 0
    confusion = Counter()
    started = time.perf_counter()

    for index, item in enumerate(claims, 1):
        try:
            result = run_pipeline(
                item["claim"],
                evidence_top_k=args.top_k,
                include_explanation=False,
                use_query_generation=args.with_query_generation,
            )
            evidence = result["evidence"]
            predicted = result["verification"]["verdict"]
        except Exception as error:  # count as NEI, keep going
            print(f"[eval] pipeline error: {error}")
            evidence, predicted = [], "NEI"

        texts = [e.get("text", "") for e in evidence]
        hit = any(
            gold_match(g, t, args.match_threshold)
            for g in item["gold"] for t in texts
        )
        if item["label"] != "NEI":
            positives += 1
            recall_hits += hit

        total[item["label"]] += 1
        correct[item["label"]] += predicted == item["label"]
        confusion[(item["label"], predicted)] += 1

        print(
            f"[eval {index}/{len(claims)}] {item['label']:8s} -> {predicted:8s} "
            f"gold_hit={hit!s:5s} | {item['claim']}"
        )
        if out:
            out.write(json.dumps({
                **item, "predicted": predicted, "gold_hit": hit,
                "evidence": [
                    {"title": e.get("title"), "text": e.get("text"),
                     "verdict": e.get("verifier_verdict")}
                    for e in evidence
                ],
            }, ensure_ascii=False) + "\n")
            out.flush()

    if out:
        out.close()

    n = sum(total.values())
    print("\n===== FEVER PIPELINE BENCHMARK =====")
    print(f"claims: {n}  time: {time.perf_counter() - started:.0f}s")
    print(f"gold sentence recall@{args.top_k}: {recall_hits}/{positives} "
          f"= {recall_hits / max(positives, 1):.1%}")
    print(f"accuracy: {sum(correct.values())}/{n} = {sum(correct.values()) / max(n, 1):.1%}")
    for label in ("SUPPORTS", "REFUTES", "NEI"):
        if total[label]:
            print(f"  {label:8s}: {correct[label]}/{total[label]} = "
                  f"{correct[label] / total[label]:.1%}")
    print("confusion (gold -> predicted):")
    for (gold, pred), count in sorted(confusion.items()):
        print(f"  {gold:8s} -> {pred:8s}: {count}")


if __name__ == "__main__":
    main()

"""
Evaluate the FACTLENS system on the FEVER dataset (Kaggle: lk1920/fever-data).

The dataset holds the prepared classifier files

    train_classifier.jsonl  dev_classifier.jsonl
    paper_dev_classifier.jsonl  paper_test_classifier.jsonl

one row per (claim, gold evidence group):
    {"id", "claim", "label", "evidence_group_id",
     "evidence": [{"page_id", "sentence_id", "text"}], "evidence_source"}

Modes
-----
pipeline (default)  the real system: claim -> retrieval (live Wikipedia,
                    optionally NewsAPI) -> ranking -> verifier. The FEVER
                    gold evidence is NOT given to the model; it is only used
                    to score retrieval.
gold                the verifier alone on the dataset's gold evidence
                    (upper bound; this is the setting the verifier was
                    trained and validated in).
both                both of the above on the same claims, side by side.

All models (verifier, dense encoder, cross-encoder, spaCy and, with
--with-query-generation, the Qwen query model) are loaded ONCE before the
loop and reused for every claim. Predictions are written after every claim,
so an interrupted run continues where it stopped (--resume).

Metrics: accuracy, macro / micro / weighted F1, per-class precision /
recall / F1, confusion matrix, and for the pipeline also retrieval metrics
(gold page retrieved, gold page / gold sentence in the selected evidence)
and an approximate FEVER score (correct label + gold evidence in top-k).

Kaggle notebook (Settings: Internet ON, Accelerator GPU; add the
lk1920/fever-data dataset as input):

    !git clone https://github.com/ShivamGoyal1507/Factlens.git
    %cd Factlens
    # Only what Kaggle lacks: requirements.txt pins torch and would replace
    # Kaggle's CUDA build.
    !pip install -q rank-bm25 peft python-dotenv
    !python -m spacy download en_core_web_sm
    !python scripts/evaluate_fever_kaggle.py --split paper_test --num-samples 300 --mode both --workers 4

Locally:

    python scripts/evaluate_fever_kaggle.py --data-dir /path/to/fever-data --num-samples 50

Speed (pipeline mode):
  * `--workers N` (default 4) threads retrieve evidence for upcoming claims
    while the main thread runs ranking + verification on the GPU, so network
    waits and model inference overlap. The GPU stage stays single-threaded.
  * Wikipedia requests from all threads share one rate limiter (one request
    per 0.35 s, at most 3 in flight, per Wikimedia's etiquette), so extra
    workers cannot get the IP blocked. That limit (~170 requests/min,
    ~8-10 requests per claim) is the ceiling: about 15-20 claims per minute.
    More than ~4 workers gives no further gain.
  * Progress lines show retrieval vs model seconds and claims/min, so you can
    see which stage is the bottleneck.
"""

from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import torch  # noqa: E402,F401  (must precede spacy / sentence-transformers)

import argparse  # noqa: E402
import itertools  # noqa: E402
import json  # noqa: E402
import random  # noqa: E402
import re  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from collections import Counter, OrderedDict, deque  # noqa: E402
from concurrent.futures import ThreadPoolExecutor  # noqa: E402
from pathlib import Path  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

LABELS = ["SUPPORTS", "REFUTES", "NEI"]
SPLIT_FILES = {
    "paper_test": "paper_test_classifier.jsonl",
    "paper_dev": "paper_dev_classifier.jsonl",
    "dev": "dev_classifier.jsonl",
    "train": "train_classifier.jsonl",
}
DATA_DIR_CANDIDATES = [
    "/kaggle/input/datasets/lk1920/fever-data",
    "/kaggle/input/fever-data",
    str(PROJECT_ROOT / "experiments/classifier/fever-data/processed"),
    str(PROJECT_ROOT / "fever-data/processed"),
]


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def find_data_file(data_dir: str | None, split: str) -> Path:
    filename = SPLIT_FILES[split]
    candidates = [data_dir] if data_dir else DATA_DIR_CANDIDATES
    for candidate in candidates:
        path = Path(candidate) / filename
        if path.is_file():
            return path
    kaggle = Path("/kaggle/input")
    if not data_dir and kaggle.is_dir():
        found = sorted(kaggle.rglob(filename))
        if found:
            return found[0]
    raise FileNotFoundError(
        f"Could not find {filename}. Looked in: {', '.join(c for c in candidates if c)}"
        " (and /kaggle/input/**). Pass --data-dir."
    )


def load_claims(path: Path) -> list[dict]:
    """Group the per-evidence-group rows into one record per claim."""
    claims: "OrderedDict[str, dict]" = OrderedDict()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            label = row["label"]
            if label == "NOT ENOUGH INFO":
                label = "NEI"
            claim_id = str(row["id"])
            record = claims.setdefault(claim_id, {
                "id": claim_id,
                "claim": row["claim"],
                "label": label,
                "gold_groups": [],
                "first_evidence": [e.get("text", "") for e in row.get("evidence") or []],
            })
            if label != "NEI" and row.get("evidence_source", "gold") == "gold":
                group = [
                    {"page_id": e.get("page_id"), "sentence_id": e.get("sentence_id"),
                     "text": e.get("text", "")}
                    for e in row.get("evidence") or []
                ]
                if group:
                    record["gold_groups"].append(group)
    return list(claims.values())


def sample_claims(claims, num_samples, seed, balanced):
    rng = random.Random(seed)
    if not num_samples or num_samples >= len(claims):
        sample = list(claims)
        rng.shuffle(sample)
        return sample
    if not balanced:
        return rng.sample(claims, num_samples)
    by_label = {label: [c for c in claims if c["label"] == label] for label in LABELS}
    per_label = num_samples // len(LABELS)
    sample = []
    for label in LABELS:
        sample.extend(rng.sample(by_label[label], min(per_label, len(by_label[label]))))
    chosen = {c["id"] for c in sample}
    rest = [c for c in claims if c["id"] not in chosen]
    sample.extend(rng.sample(rest, num_samples - len(sample)))
    rng.shuffle(sample)
    return sample


# ---------------------------------------------------------------------------
# Evidence matching (FEVER text is tokenized: "-LRB-", "word ,")
# ---------------------------------------------------------------------------

_FEVER_TOKENS = {
    "-LRB-": "(", "-RRB-": ")", "-LSB-": "[", "-RSB-": "]", "-LCB-": "{",
    "-RCB-": "}", "-COLON-": ":", "``": '"', "''": '"',
}
_STOP = {
    "the", "a", "an", "of", "in", "and", "is", "was", "to", "for", "on", "by",
    "as", "at", "with", "it", "he", "she", "his", "her", "its", "be", "are",
    "were", "from", "that", "which", "or", "has", "had", "this",
}


def detokenize(text: str) -> str:
    for token, char in _FEVER_TOKENS.items():
        text = text.replace(token, char)
    return text


def page_title(page_id: str) -> str:
    """FEVER page id -> Wikipedia title: 'Vedam_-LRB-film-RRB-' -> 'vedam (film)'."""
    return re.sub(r"\s+", " ", detokenize((page_id or "").replace("_", " "))).strip().lower()


def _tokens(text: str) -> set[str]:
    return {
        t for t in re.findall(r"[a-z0-9]+", detokenize(text).lower()) if t not in _STOP
    }


def sentence_match(gold: str, passage: str, threshold: float = 0.6) -> bool:
    """Fuzzy match: live Wikipedia wording differs from the 2017 FEVER dump."""
    gold_tokens = _tokens(gold)
    if len(gold_tokens) < 3:
        return False
    return len(gold_tokens & _tokens(passage)) / len(gold_tokens) >= threshold


def retrieval_scores(record, evidence, retrieved_titles):
    """Per-claim retrieval metrics against the FEVER gold evidence."""
    gold_pages = {
        page_title(e["page_id"]) for group in record["gold_groups"] for e in group
    }
    evidence_titles = {(e.get("title") or "").lower() for e in evidence}
    texts = [e.get("text", "") for e in evidence]
    sentence_hit = any(
        sentence_match(e["text"], t)
        for group in record["gold_groups"] for e in group for t in texts
    )
    # A complete gold group: every sentence of one group is in the evidence.
    group_hit = any(
        all(any(sentence_match(e["text"], t) for t in texts) for e in group)
        for group in record["gold_groups"]
    )
    return {
        "gold_page_retrieved": bool(gold_pages & {t.lower() for t in retrieved_titles}),
        "gold_page_in_evidence": bool(gold_pages & evidence_titles),
        "gold_sentence_in_evidence": sentence_hit,
        "gold_group_in_evidence": group_hit,
    }


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def classification_metrics(gold, pred):
    from sklearn.metrics import (
        accuracy_score, classification_report, confusion_matrix, f1_score,
        precision_recall_fscore_support,
    )

    precision, recall, f1, support = precision_recall_fscore_support(
        gold, pred, labels=LABELS, zero_division=0
    )
    return {
        "n": len(gold),
        "accuracy": accuracy_score(gold, pred),
        "macro_f1": f1_score(gold, pred, labels=LABELS, average="macro", zero_division=0),
        "micro_f1": f1_score(gold, pred, labels=LABELS, average="micro", zero_division=0),
        "weighted_f1": f1_score(gold, pred, labels=LABELS, average="weighted", zero_division=0),
        "macro_precision": float(precision.mean()),
        "macro_recall": float(recall.mean()),
        "per_class": {
            label: {"precision": float(p), "recall": float(r), "f1": float(f), "support": int(s)}
            for label, p, r, f, s in zip(LABELS, precision, recall, f1, support)
        },
        "confusion_matrix": {
            "labels": LABELS,
            "matrix (rows=gold, cols=predicted)": confusion_matrix(
                gold, pred, labels=LABELS
            ).tolist(),
        },
        "report": classification_report(
            gold, pred, labels=LABELS, digits=4, zero_division=0
        ),
    }


def pipeline_retrieval_metrics(rows):
    positives = [r for r in rows if r["label"] != "NEI"]
    n = max(len(positives), 1)

    def share(key):
        return sum(bool(r["retrieval"][key]) for r in positives) / n

    fever_hits = sum(
        r["predicted"] == r["label"]
        and (r["label"] == "NEI" or r["retrieval"]["gold_group_in_evidence"])
        for r in rows
    )
    return {
        "claims_with_gold_evidence": len(positives),
        "gold_page_retrieved": share("gold_page_retrieved"),
        "gold_page_in_topk_evidence": share("gold_page_in_evidence"),
        "gold_sentence_in_topk_evidence": share("gold_sentence_in_evidence"),
        "gold_group_in_topk_evidence": share("gold_group_in_evidence"),
        "approx_fever_score": fever_hits / max(len(rows), 1),
        "errors": sum(1 for r in rows if r.get("error")),
    }


def print_metrics(title, metrics, retrieval=None, timing=None):
    print(f"\n{'=' * 64}\n{title}\n{'=' * 64}")
    print(f"claims            : {metrics['n']}")
    print(f"accuracy          : {metrics['accuracy']:.4f}")
    print(f"macro F1          : {metrics['macro_f1']:.4f}")
    print(f"micro F1          : {metrics['micro_f1']:.4f}  (= accuracy for single-label 3-class)")
    print(f"weighted F1       : {metrics['weighted_f1']:.4f}")
    print(f"macro precision   : {metrics['macro_precision']:.4f}")
    print(f"macro recall      : {metrics['macro_recall']:.4f}")
    print("\nper class:")
    print(metrics["report"])
    print("confusion matrix (rows = gold, cols = predicted):")
    print("            " + "".join(f"{label:>10s}" for label in LABELS))
    for label, row in zip(LABELS, metrics["confusion_matrix"]["matrix (rows=gold, cols=predicted)"]):
        print(f"{label:>10s}  " + "".join(f"{value:>10d}" for value in row))
    if retrieval:
        print("\nretrieval (SUPPORTS/REFUTES claims, vs FEVER gold evidence):")
        print(f"  gold page retrieved           : {retrieval['gold_page_retrieved']:.4f}")
        print(f"  gold page in top-k evidence   : {retrieval['gold_page_in_topk_evidence']:.4f}")
        print(f"  gold sentence in top-k (fuzzy): {retrieval['gold_sentence_in_topk_evidence']:.4f}")
        print(f"  full gold group in top-k      : {retrieval['gold_group_in_topk_evidence']:.4f}")
        print(f"  approx. FEVER score           : {retrieval['approx_fever_score']:.4f}")
        print(f"  claims that raised errors     : {retrieval['errors']}")
    if timing:
        workers = f", {timing['workers']} retrieval threads" if timing.get("workers") else ""
        print(f"\ntime: {timing['total_s']:.0f}s total, {timing['per_claim_s']:.2f}s per claim"
              f" (model loading {timing['model_load_s']:.0f}s, done once{workers})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def load_models(with_query_generation):
    """Load every model once; run_pipeline reuses these cached instances."""
    import main
    from src.ranking.passage_splitter import _load_spacy

    started = time.perf_counter()
    print("Loading models (once)...")
    main._verifier()
    main._dense_model("sentence-transformers/all-MiniLM-L6-v2")
    main._cross_encoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    _load_spacy()
    from src.evidence_retrieval.sources.wikipedia import _default_nlp

    _default_nlp()  # load before worker threads start
    if with_query_generation:
        from src.query_generation.slm_client import load_slm

        load_slm()
    elapsed = time.perf_counter() - started
    print(f"Models loaded in {elapsed:.0f}s on device: {main._verifier().device_name}")
    return main, elapsed


def iterate_pipeline(main, records, args):
    """
    Yield (record, result, error, timing) for every claim, in input order.

    Two-stage pipeline:
      * `args.workers` threads run the network stage (main.prepare_claim:
        query plan + Wikipedia/NewsAPI retrieval) for upcoming claims;
      * the calling thread runs the model stage (main.finish_claim: ranking +
        verification) on the GPU, one claim at a time.
    So retrieval of the next claims overlaps model inference of the current
    one, and several claims' Wikipedia requests are in flight together. The
    Wikipedia client still enforces the global rate limit and at most 3
    concurrent requests, so more workers cannot get the IP blocked.
    """
    def prepare(record):
        started = time.perf_counter()
        try:
            prepared = main.prepare_claim(
                record["claim"],
                use_query_generation=args.with_query_generation,
                verbose=False,
            )
            return prepared, None, time.perf_counter() - started
        except Exception as exc:
            return None, f"{type(exc).__name__}: {exc}", time.perf_counter() - started

    prefetch = max(2, args.workers * 2)  # bounded read-ahead keeps memory flat
    with ThreadPoolExecutor(max_workers=max(1, args.workers),
                            thread_name_prefix="retrieval") as pool:
        queue = deque()
        records_iter = iter(records)
        for record in itertools.islice(records_iter, prefetch):
            queue.append((record, pool.submit(prepare, record)))

        while queue:
            record, future = queue.popleft()
            next_record = next(records_iter, None)
            if next_record is not None:
                queue.append((next_record, pool.submit(prepare, next_record)))

            prepared, error, retrieval_s = future.result()
            result, model_started = None, time.perf_counter()
            if prepared is not None:
                try:
                    result = main.finish_claim(
                        prepared,
                        evidence_top_k=args.evidence_top_k,
                        include_explanation=False,
                    )
                except Exception as exc:
                    error = f"{type(exc).__name__}: {exc}"
            timing = {
                "retrieval_s": round(retrieval_s, 2),
                "model_s": round(time.perf_counter() - model_started, 2),
            }
            yield record, result, error, timing


def run_gold_mode(verifier, claims, batch_size=64):
    """Verifier on each claim's first dataset evidence group (as in training)."""
    items = [
        {"claim": c["claim"],
         "evidence": [t for t in c["first_evidence"] if t.strip()] or ["No evidence."]}
        for c in claims
    ]
    predictions = []
    for start in range(0, len(items), batch_size):
        batch = items[start:start + batch_size]
        predictions.extend(p["verdict"] for p in verifier.verify_batch(batch))
    return predictions


def main_cli() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate FACTLENS on FEVER (Kaggle lk1920/fever-data).",
    )
    parser.add_argument("--data-dir", default=None,
                        help="Folder with the *_classifier.jsonl files (auto-detected on Kaggle)")
    parser.add_argument("--split", choices=sorted(SPLIT_FILES), default="paper_test")
    parser.add_argument("--num-samples", type=int, default=200,
                        help="Number of claims to evaluate; 0 = the whole split")
    parser.add_argument("--balanced", action="store_true",
                        help="Sample the same number of SUPPORTS / REFUTES / NEI claims")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mode", choices=["pipeline", "gold", "both"], default="pipeline")
    parser.add_argument("--evidence-top-k", type=int, default=5)
    parser.add_argument("--with-query-generation", action="store_true",
                        help="Use the Qwen query generator (slower; off = search with the claim)")
    parser.add_argument("--with-news", action="store_true",
                        help="Also query NewsAPI (needs NEWSAPI_API_KEY; FEVER is Wikipedia-based)")
    parser.add_argument("--output-dir", default=None,
                        help="Default: /kaggle/working/fever_eval on Kaggle, ./eval_outputs locally")
    parser.add_argument("--workers", type=int, default=4,
                        help="Retrieval threads running ahead of the GPU stage (default 4)")
    parser.add_argument("--resume", action="store_true",
                        help="Skip claims already in the predictions file of --output-dir")
    args = parser.parse_args()

    if not args.with_news:
        os.environ["FACTLENS_DISABLE_NEWS"] = "1"
    os.environ.setdefault("FACTLENS_QUIET", "1")  # threads would interleave logs

    data_file = find_data_file(args.data_dir, args.split)
    claims = load_claims(data_file)
    print(f"Loaded {len(claims)} claims from {data_file}")
    print("Label distribution:", dict(Counter(c["label"] for c in claims)))
    sample = sample_claims(claims, args.num_samples, args.seed, args.balanced)
    print(f"Evaluating {len(sample)} claims ({args.split}, seed {args.seed}, mode {args.mode})")

    output_dir = Path(
        args.output_dir
        or ("/kaggle/working/fever_eval" if Path("/kaggle/working").is_dir()
            else PROJECT_ROOT / "eval_outputs")
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    run_name = f"{args.split}_n{len(sample)}_seed{args.seed}"

    main, model_load_s = load_models(args.with_query_generation)
    results = {"config": vars(args) | {"data_file": str(data_file)}}

    # ---- gold-evidence mode (verifier only) --------------------------------
    if args.mode in ("gold", "both"):
        started = time.perf_counter()
        gold_pred = run_gold_mode(main._verifier(), sample)
        gold_metrics = classification_metrics([c["label"] for c in sample], gold_pred)
        print_metrics("VERIFIER WITH GOLD EVIDENCE (upper bound)", gold_metrics, timing={
            "total_s": time.perf_counter() - started,
            "per_claim_s": (time.perf_counter() - started) / max(len(sample), 1),
            "model_load_s": model_load_s,
        })
        results["gold_evidence"] = gold_metrics

    # ---- full pipeline -----------------------------------------------------
    if args.mode in ("pipeline", "both"):
        predictions_path = output_dir / f"predictions_{run_name}.jsonl"
        done = {}
        if args.resume and predictions_path.is_file():
            for line in predictions_path.open(encoding="utf-8"):
                if line.strip():
                    row = json.loads(line)
                    done[row["id"]] = row
            print(f"Resuming: {len(done)} claims already evaluated")

        rows = [done[c["id"]] for c in sample if c["id"] in done]
        todo = [c for c in sample if c["id"] not in done]
        started = time.perf_counter()
        with predictions_path.open("a" if args.resume else "w", encoding="utf-8") as out:
            for record, result, error, timing in iterate_pipeline(main, todo, args):
                if result is not None:
                    predicted = result["verification"]["verdict"]
                    evidence = result["evidence"]
                    retrieved_titles = result.get("retrieved_titles", [])
                else:  # keep evaluating; a failed claim counts as NEI
                    predicted, evidence, retrieved_titles = "NEI", [], []

                row = {
                    "id": record["id"],
                    "claim": record["claim"],
                    "label": record["label"],
                    "predicted": predicted,
                    "correct": predicted == record["label"],
                    "retrieval": retrieval_scores(record, evidence, retrieved_titles),
                    "evidence": [
                        {"title": e.get("title"), "text": e.get("text"),
                         "verdict": e.get("verifier_verdict")}
                        for e in evidence
                    ],
                    "retrieval_seconds": timing["retrieval_s"],
                    "model_seconds": timing["model_s"],
                    "error": error,
                }
                rows.append(row)
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                out.flush()

                done_now = len(rows) - len(done)
                accuracy = sum(r["correct"] for r in rows) / len(rows)
                rate = done_now / max(time.perf_counter() - started, 1e-9) * 60
                print(
                    f"[{len(rows)}/{len(sample)}] {record['label']:8s} -> {predicted:8s} "
                    f"(retrieval {timing['retrieval_s']:.1f}s, model {timing['model_s']:.1f}s; "
                    f"{rate:.1f} claims/min, running acc {accuracy:.3f})"
                    f"{'  ERROR ' + error if error else ''} | {record['claim'][:70]}",
                    flush=True,
                )

        elapsed = time.perf_counter() - started
        pipeline_metrics = classification_metrics(
            [r["label"] for r in rows], [r["predicted"] for r in rows]
        )
        retrieval = pipeline_retrieval_metrics(rows)
        print_metrics("FULL PIPELINE (retrieval + ranking + verifier)", pipeline_metrics,
                      retrieval=retrieval, timing={
                          "total_s": elapsed,
                          "per_claim_s": elapsed / max(len(todo), 1),
                          "model_load_s": model_load_s,
                          "workers": args.workers,
                      })
        results["pipeline"] = pipeline_metrics | {"retrieval": retrieval}
        print(f"\nper-claim predictions: {predictions_path}")

    metrics_path = output_dir / f"metrics_{run_name}_{args.mode}.json"
    for section in ("gold_evidence", "pipeline"):
        if section in results:
            results[section].pop("report", None)
    metrics_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"metrics: {metrics_path}")


if __name__ == "__main__":
    main_cli()

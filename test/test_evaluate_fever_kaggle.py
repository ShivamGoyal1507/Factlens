import importlib.util
import json
import time
from pathlib import Path
from types import SimpleNamespace

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "evaluate_fever_kaggle.py"
spec = importlib.util.spec_from_file_location("evaluate_fever_kaggle", SCRIPT)
evaluate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluate)


class _FakeMain:
    """Network stage sleeps (simulated retrieval); model stage is instant."""

    def prepare_claim(self, claim, use_query_generation=False, verbose=True):
        time.sleep(0.1)
        if claim == "boom":
            raise RuntimeError("retrieval failed")
        return {"claim": claim}

    def finish_claim(self, prepared, evidence_top_k=5, include_explanation=False):
        return {"verification": {"verdict": "SUPPORTS"}, "evidence": [],
                "retrieved_titles": [prepared["claim"]]}


def test_iterate_pipeline_overlaps_retrieval_and_keeps_order():
    records = [{"id": str(i), "claim": "boom" if i == 3 else f"claim {i}"} for i in range(8)]
    args = SimpleNamespace(workers=4, with_query_generation=False, evidence_top_k=5)

    started = time.perf_counter()
    out = list(evaluate.iterate_pipeline(_FakeMain(), records, args))
    elapsed = time.perf_counter() - started

    assert [record["id"] for record, *_ in out] == [r["id"] for r in records]
    assert out[3][1] is None and "retrieval failed" in out[3][2]
    assert all(result is not None for i, (_, result, _, _) in enumerate(out) if i != 3)
    assert elapsed < 8 * 0.1 * 0.6  # sequential would take 0.8s


def test_load_claims_groups_rows_and_metrics(tmp_path):
    rows = [
        {"id": 1, "claim": "A is B.", "label": "SUPPORTS", "evidence_group_id": 0,
         "evidence": [{"page_id": "A_-LRB-film-RRB-", "sentence_id": 0, "text": "A is a B film ."}],
         "evidence_source": "gold"},
        {"id": 1, "claim": "A is B.", "label": "SUPPORTS", "evidence_group_id": 1,
         "evidence": [{"page_id": "A", "sentence_id": 2, "text": "A was B ."}],
         "evidence_source": "gold"},
        {"id": 2, "claim": "C is D.", "label": "NEI", "evidence_group_id": None,
         "evidence": [{"page_id": "X", "sentence_id": 0, "text": "Unrelated ."}],
         "evidence_source": "sampled_corpus"},
    ]
    path = tmp_path / "paper_test_classifier.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows))

    claims = evaluate.load_claims(path)
    assert [c["id"] for c in claims] == ["1", "2"]
    assert len(claims[0]["gold_groups"]) == 2 and claims[1]["gold_groups"] == []
    assert evaluate.page_title("A_-LRB-film-RRB-") == "a (film)"

    scores = evaluate.retrieval_scores(
        claims[0], [{"title": "A (film)", "text": "A is a B film."}], ["A (film)", "Other"]
    )
    assert scores["gold_page_retrieved"] and scores["gold_page_in_evidence"]

    metrics = evaluate.classification_metrics(
        ["SUPPORTS", "REFUTES", "NEI", "NEI"], ["SUPPORTS", "NEI", "NEI", "NEI"]
    )
    assert metrics["accuracy"] == 0.75 and metrics["micro_f1"] == 0.75

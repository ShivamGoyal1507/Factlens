from src.classifier.aggregation import aggregate_predictions, rank_weights
from src.query_generation.generator import normalize_claim
from src.ranking.entity_scorer import EntityScorer


def _pred(s, r, n):
    probs = {"SUPPORTS": s, "REFUTES": r, "NEI": n}
    return {"verdict": max(probs, key=probs.get), "probabilities": probs}


def test_one_clear_refuting_passage_beats_generic_background():
    # generic definition (NEI), then "Modi is the current PM" (REFUTES)
    result = aggregate_predictions([_pred(0.0, 0.0, 1.0), _pred(0.0, 1.0, 0.0)])
    assert result["verdict"] == "REFUTES"
    assert result["decisive_evidence"] == [1]


def test_only_irrelevant_passages_give_nei():
    result = aggregate_predictions([_pred(0.1, 0.1, 0.8), _pred(0.2, 0.1, 0.7)])
    assert result["verdict"] == "NEI"
    assert result["decisive_evidence"] == []


def test_one_vs_one_conflict_is_nei():
    result = aggregate_predictions([_pred(0.0, 0.99, 0.01), _pred(0.99, 0.0, 0.01)])
    assert result["verdict"] == "NEI"


def test_several_supporting_passages_outvote_one_offtopic_refute():
    # Jensen Huang case: biography read as REFUTES at rank 1, three news
    # passages stating the fact as SUPPORTS.
    result = aggregate_predictions([
        _pred(0.01, 0.99, 0.0),
        _pred(1.0, 0.0, 0.0),
        _pred(1.0, 0.0, 0.0),
        _pred(1.0, 0.0, 0.0),
        _pred(0.0, 0.0, 1.0),
    ])
    assert result["verdict"] == "SUPPORTS"
    assert result["decisive_evidence"] == [1, 2, 3]


def test_empty_predictions_are_nei():
    assert aggregate_predictions([])["verdict"] == "NEI"


def test_rank_weights_decrease_with_floor():
    assert rank_weights(7) == [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.5]


def test_normalize_claim_restores_entity_case_and_period():
    assert (
        normalize_claim("rahul gandhi is prime minister of india", ["Rahul Gandhi", "India"])
        == "Rahul Gandhi is prime minister of India."
    )
    assert normalize_claim("ISRO launched Chandrayaan-3 in 2023.", ["ISRO"]) == (
        "ISRO launched Chandrayaan-3 in 2023."
    )
    # partial-word matches are left alone
    assert normalize_claim("indian economy grew", ["India"]) == "Indian economy grew."


def test_generational_suffix_is_a_namesake():
    candidates = [
        {"title": "Barack Obama Sr.", "text": "Barack Obama Sr. was a Kenyan economist.", "rerank_score": 9.0},
        {"title": "Barack Obama", "text": "Barack Obama was born in Honolulu, Hawaii.", "rerank_score": 8.0},
        {"title": "Early life and career of Barack Obama", "text": "Obama was born in Honolulu.", "rerank_score": 7.0},
    ]
    ranked = EntityScorer().score(candidates, ["Barack Obama", "Kenya"])
    assert ranked[-1]["title"] == "Barack Obama Sr."

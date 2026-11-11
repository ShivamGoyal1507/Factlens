import json
import re

from src.query_generation.complexity import claim_complexity_score, estimate_claim_complexity
from src.query_generation.validator import is_valid_query, remove_duplicate_queries

try:
    from src.evidence_retrieval.retriever import retrieve_evidence_for_claim
except Exception:  # pragma: no cover - import guard for optional environments
    def retrieve_evidence_for_claim(claim, queries=None, claim_made_at=None):
        return []


MODEL_NAME = "google/flan-t5-base"

_MODEL_CACHE = {
    "tokenizer": None,
    "model": None,
    "device": None,
}

INVALID_QUERY_PATTERNS = (
    "key events/actions",
    "important relationships",
    "named entities",
    "important keywords",
    "original claim",
    "claim content dates",
    "search for subqueries",
    "output only a json array",
    "requirements:",
    "not specified",
    "here are",
    "subqueries:",
    "important events",
    "important information",
    "output:",
    "key points",
    "evidence request",
)


def _normalize_query(query):
    if not isinstance(query, str):
        return ""
    cleaned = " ".join(query.strip().split())
    cleaned = cleaned.strip("\"'[](){}")
    return cleaned


def _deduplicate_queries(queries):
    seen = set()
    unique = []
    for query in queries:
        normalized = _normalize_query(query)
        if not normalized:
            continue
        key = normalized.lower()
        if key in seen:
            continue
        seen.add(key)
        unique.append(normalized)
    return unique


def _near_duplicate(a, b):
    a_text = _normalize_query(a).lower()
    b_text = _normalize_query(b).lower()
    if not a_text or not b_text:
        return False
    if a_text == b_text:
        return True

    a_tokens = set(re.findall(r"[A-Za-z0-9]+", a_text))
    b_tokens = set(re.findall(r"[A-Za-z0-9]+", b_text))
    if not a_tokens or not b_tokens:
        return False

    union = a_tokens | b_tokens
    overlap = len(a_tokens & b_tokens) / len(union) if union else 0
    return overlap >= 0.75


def _looks_like_prompt_fragment(query):
    lower = query.lower()
    if any(pattern in lower for pattern in INVALID_QUERY_PATTERNS):
        return True
    if re.match(r"^(?:key|important|named|original|search|output|evidence|requirements)\b", lower):
        return True
    return False


def _is_valid_query(candidate):
    return is_valid_query(candidate)


def _filter_and_prune_queries(queries):
    valid = []
    seen = set()

    for item in queries:
        normalized = _normalize_query(item)
        if not normalized or not is_valid_query(normalized):
            continue

        if any(_near_duplicate(normalized, existing) for existing in valid):
            continue

        key = normalized.lower()
        if key in seen:
            continue
        seen.add(key)
        valid.append(normalized)

    return remove_duplicate_queries(valid)[:5]


def _extract_years(claim):
    if not isinstance(claim, str):
        return []
    return re.findall(r"\b(19\d{2}|20\d{2}|21\d{2})\b", claim)


def _extract_keywords(claim):
    tokens = re.findall(r"[A-Za-z0-9][A-Za-z0-9-]*", claim)
    stopwords = {
        "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
        "has", "have", "in", "is", "it", "its", "of", "on", "or", "that",
        "the", "this", "to", "was", "were", "with", "about", "into", "after",
        "before", "during", "under", "over", "against", "between", "without",
        "within", "through", "while", "claim", "made", "news", "evidence",
        "first", "launch", "launched", "using", "used", "powered",
        "artificial", "intelligence", "ai"
    }
    keywords = []
    for token in tokens:
        lowered = token.lower()
        if lowered in stopwords:
            continue
        if lowered.isdigit() and len(token) <= 4:
            keywords.append(token)
            continue
        if len(token) >= 2:
            keywords.append(token)
    return keywords


def _extract_named_entities(claim):
    if not isinstance(claim, str):
        return []

    entities = []
    for pattern in [
        r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b",
        r"\b(?:India|China|United States|United Kingdom|Russia|Japan|France|Germany|Brazil|Australia|Canada|Pakistan|Iran|Israel|South Korea|Nepal|Sri Lanka|Singapore|Mexico|Europe|Asia|Africa)\b",
    ]:
        for item in re.findall(pattern, claim):
            clean = " ".join(item.split())
            if clean and clean.lower() not in {"the", "a", "an"}:
                entities.append(clean)

    for token in claim.replace("-", " ").split():
        if token.isupper() and len(token) >= 2:
            entities.append(token)

    unique = []
    seen = set()
    for entity in entities:
        key = entity.lower()
        if key in seen:
            continue
        seen.add(key)
        unique.append(entity)
    return unique[:12]


def extract_claim_features(claim):
    if not isinstance(claim, str):
        return {"keywords": [], "entities": [], "dates": [], "events": [], "relationships": []}

    cleaned = _normalize_query(claim)
    keywords = _extract_keywords(cleaned)
    entities = _extract_named_entities(cleaned)
    years = _extract_years(cleaned)
    action_terms = []
    for match in re.finditer(
        r"\b(?:launched|launch|announced|opened|signed|tested|developed|reported|approved|built|released|created|used|introduced|said|claim|fired)\b",
        cleaned,
        flags=re.IGNORECASE,
    ):
        action_terms.append(match.group(0).lower())
    relationships = []
    if "first" in cleaned.lower():
        relationships.append("first occurrence or first launch")
    if "AI" in cleaned or "artificial intelligence" in cleaned.lower():
        relationships.append("AI or artificial intelligence involved")
    if "satellite" in cleaned.lower():
        relationships.append("satellite or space program context")

    return {
        "keywords": keywords[:12],
        "entities": entities[:12],
        "dates": years,
        "events": list(dict.fromkeys(action_terms))[:8],
        "relationships": relationships[:8],
    }


def _build_model_prompt(claim):
    from src.query_generation.prompts import QUERY_GENERATION_PROMPT

    features = extract_claim_features(claim)
    keywords = ", ".join(features["keywords"][:12]) if features["keywords"] else "not specified"
    entities = ", ".join(features["entities"][:12]) if features["entities"] else "not specified"
    dates = ", ".join(features["dates"]) if features["dates"] else "not specified"
    events = ", ".join(features["events"]) if features["events"] else "not specified"
    relationships = "; ".join(features["relationships"]) if features["relationships"] else "not specified"
    return QUERY_GENERATION_PROMPT.format(
        claim=claim,
        keywords=keywords,
        entities=entities,
        dates=dates,
        events=events,
        relationships=relationships,
    )


def _extract_json_queries(payload):
    if isinstance(payload, list):
        return [str(item) for item in payload if str(item).strip()]

    if isinstance(payload, str):
        cleaned = payload.strip()
        if not cleaned:
            return []

        try:
            loaded = json.loads(cleaned)
            if isinstance(loaded, list):
                return [str(item) for item in loaded if str(item).strip()]
        except (TypeError, ValueError, json.JSONDecodeError):
            pass

        lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
        parsed = []
        for line in lines:
            candidate = line.strip("[]\" '")
            candidate = re.sub(r"^\d+[\.)\-\s]*", "", candidate)
            candidate = _normalize_query(candidate)
            if candidate:
                parsed.append(candidate)
        return parsed

    return []


def _load_flant5_model():
    import torch
    from transformers import (
        AutoModelForSeq2SeqLM,
        AutoTokenizer,
    )

    if (
        _MODEL_CACHE["model"] is not None
        and _MODEL_CACHE["tokenizer"] is not None
    ):
        return _MODEL_CACHE["tokenizer"], _MODEL_CACHE["model"]

    print(f"[FLAN] Loading {MODEL_NAME}")

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    dtype = (
        torch.float16
        if device.type == "cuda"
        else torch.float32
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=dtype,
    )

    model.to(device)
    model.eval()

    _MODEL_CACHE["tokenizer"] = tokenizer
    _MODEL_CACHE["model"] = model
    _MODEL_CACHE["device"] = device

    print(f"[FLAN] Model loaded successfully on {device}")

    return tokenizer, model



def generate_queries_with_llm(claim):
    if not isinstance(claim, str) or not claim.strip():
        return []

    loaded = _load_flant5_model()
    if loaded is None:
        return []

    tokenizer, model = loaded
    prompt = _build_model_prompt(claim)

    try:
        import torch

        device = _MODEL_CACHE.get("device")
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        if device is not None and str(device) != "cpu":
            inputs = {key: value.to(device) for key, value in inputs.items()}

        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=64,
                do_sample=False,
                num_beams=4,
                num_return_sequences=4,
                repetition_penalty=1.15,
                length_penalty=1.0,
            )

        generated = []
        for output in outputs:
            text = tokenizer.decode(output, skip_special_tokens=True)
            generated.extend(_extract_json_queries(text))

        return _filter_and_prune_queries(generated)[:5]
    except Exception:
        return []

def _is_multi_component_claim(claim, features):
    text = claim.lower()

    events = features.get("events", [])
    relationships = features.get("relationships", [])

    # Multiple actions indicate multiple factual components.
    if len(set(events)) >= 2:
        return True

    # Look for separately stated clauses.
    clause_patterns = [
        r"\band\b",
        r"\bbut\b",
        r"\bwhereas\b",
        r"\bwhile\b",
        r"\balthough\b",
        r"\bwhich was\b",
        r"\bwho was\b",
        r"\bdeveloped by\b",
        r"\bin collaboration with\b",
        r"\baccording to\b",
    ]

    if any(re.search(pattern, text) for pattern in clause_patterns):
        return True

    # Certain qualifiers introduce independently verifiable dimensions.
    if relationships:
        return True

    if re.search(r"\b(first|only|largest|smallest|oldest|latest)\b", text):
        return True

    return False


def _build_diverse_fallback_queries(claim, features):
    text = _normalize_query(claim)

    if not text:
        return []

    lower = text.lower()
    candidates = []

    # Extract the main subject by removing the opening location/entity
    # only when useful. Keep the original claim for direct verification.
    entities = features.get("entities", [])
    events = features.get("events", [])
    years = features.get("dates", [])

    entity = entities[0] if entities else ""
    year = years[0] if years else ""

    # 1. Verify the complete claim.
    candidates.append(
        f"Did {text.rstrip('.!?')}?"
    )

    # 2. Investigate a "first", "only", or similar superlative claim.
    if re.search(r"\b(first|only|largest|smallest|oldest|latest)\b", lower):
        candidates.append(
            f"Was this the first such development? {text}"
        )

    # 3. Investigate the technology or capability.
    technology_terms = [
        term for term in ("AI", "artificial intelligence", "machine learning")
        if term.lower() in lower
    ]

    if technology_terms:
        technology = technology_terms[0]
        candidates.append(
            f"What role did {technology} play? {text}"
        )

    # 4. Investigate the organisation responsible for development.
    developed_match = re.search(
        r"\bdeveloped by\s+([^,.]+)",
        text,
        flags=re.IGNORECASE,
    )

    if developed_match:
        organisation = developed_match.group(1).strip()
        candidates.append(
            f"Did {organisation} develop the project described here? {text}"
        )

    # 5. Investigate a collaboration or partnership.
    collaboration_match = re.search(
        r"\b(?:in collaboration with|in partnership with|jointly with)\s+([^,.]+)",
        text,
        flags=re.IGNORECASE,
    )

    if collaboration_match:
        partner = collaboration_match.group(1).strip()
        candidates.append(
            f"Which organisation collaborated with {entity or 'the organisation'} "
            f"on this project? {partner} {text}"
        )

    # 6. Investigate the date or timeline when available.
    if year and events:
        candidates.append(
            f"{entity} {events[0]} timeline {year}".strip()
        )

    return _filter_and_prune_queries(candidates)[:5]


def _keyword_fallback_queries(claim):
    text = _normalize_query(claim)
    if not text:
        return []

    features = extract_claim_features(text)
    return _build_diverse_fallback_queries(text, features)


def _analysis_to_queries(claim, analysis):
    if not isinstance(analysis, dict):
        return []

    components = analysis.get("components", [])
    if not isinstance(components, list) or not components:
        return []

    queries = []
    for component in components:
        normalized = _normalize_query(component)
        if not normalized:
            continue
        if "?" in normalized:
            candidate = normalized
        else:
            candidate = f"Did {normalized.rstrip('.!?')}?"
        if is_valid_query(candidate):
            queries.append(candidate)

    if not queries:
        entities = analysis.get("entities", [])
        if isinstance(entities, list) and entities:
            subject = _normalize_query(entities[0])
            if subject:
                queries.append(f"Did {subject} feature in this claim?")

    queries = remove_duplicate_queries(queries)
    if 3 <= len(queries) <= 5:
        return queries[:5]
    return []



def generate_queries(claim, claim_made_at=None):
    """
    Main query-generation pipeline:
    1. Analyze the original claim.
    2. Estimate claim complexity.
    3. Generate subqueries using FLAN-T5-Base.
    4. Validate, ground and deduplicate the queries.
    5. Use deterministic fallback queries if generation fails.
    """
    if not isinstance(claim, str) or not claim.strip():
        return []

    claim = _normalize_query(claim)

    # Step 1: Determine claim complexity.
    complexity = estimate_claim_complexity(claim)

    # Step 2: Analyze the claim using FLAN-T5-Base.
    try:
        from src.query_generation.analyzer import analyze_claim

        analysis = analyze_claim(claim)

    except Exception as error:
        print(f"[QUERY GENERATION] Claim analysis failed: {error}")
        analysis = None

    # Step 3: Generate subqueries using the structured analysis.
    if isinstance(analysis, dict):
        try:
            from src.query_generation.subquery_generator import (
                generate_subqueries
            )

            queries = generate_subqueries(
                claim=claim,
                analysis=analysis,
                complexity=complexity,
            )

            queries = _filter_and_prune_queries(queries)

            if queries:
                print(
                    f"[QUERY GENERATION] Generated {len(queries)} "
                    f"queries using FLAN-T5-Base."
                )
                return queries

        except Exception as error:
            print(f"[QUERY GENERATION] Subquery generation failed: {error}")

    # Step 4: Deterministic fallback; do not invoke SmolLM2.
    fallback_queries = _keyword_fallback_queries(claim)

    if fallback_queries:
        print("[QUERY GENERATION] Using deterministic fallback.")
        return fallback_queries[:5]

    # Last-resort fallback.
    return [claim]


def generate_queries_with_metadata(claim, claim_made_at=None):
    queries = generate_queries(claim, claim_made_at=claim_made_at)
    return [{"query": query, "claim_made_at": claim_made_at} for query in queries]


def generate_evidence_for_claim(claim, claim_made_at=None):
    if not isinstance(claim, str) or not claim.strip():
        return []

    queries = generate_queries(claim, claim_made_at=claim_made_at)
    if not queries:
        return []

    return retrieve_evidence_for_claim(claim, queries=queries, claim_made_at=claim_made_at)


def _query_has_claim_signal(query, claim):
    text = _normalize_query(query).lower()
    if not text:
        return False
    if not isinstance(claim, str):
        return False

    claim_tokens = set(re.findall(r"[A-Za-z0-9]+", _normalize_query(claim).lower()))
    if not claim_tokens:
        return True

    query_tokens = set(re.findall(r"[A-Za-z0-9]+", text))
    if not query_tokens:
        return False

    overlap = query_tokens & claim_tokens
    if overlap:
        return True

    if "?" in text:
        return True

    return len(query_tokens) >= 3


def _parse_smol_output(payload, claim=None):
    if isinstance(payload, str):
        raw = payload
    elif isinstance(payload, dict):
        raw = payload.get("text", "")
    else:
        raw = ""

    claim_tokens = set()
    if isinstance(claim, str):
        claim_tokens = set(re.findall(r"[A-Za-z0-9]+", _normalize_query(claim).lower()))

    extracted = []
    for match in re.findall(r'"([^"]+)"|\'([^\']+)\'', raw):
        for fragment in match:
            if fragment:
                extracted.append(fragment)
    extracted.extend(line.strip() for line in raw.splitlines())

    items = []
    for text in extracted:
        candidate = text.strip()
        if not candidate:
            continue
        candidate = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", candidate)
        candidate = candidate.strip("\"' ")
        candidate = _normalize_query(candidate)
        if not candidate:
            continue
        lower = candidate.lower()
        if lower in {"search queries:", "search queries", "queries:", "queries"}:
            continue
        if re.match(r'^[A-Za-z_]+\s*:\s*', lower):
            continue
        if re.match(r'^[A-Za-z_]+\s*"?\s*$', lower):
            continue
        if any(token in lower for token in ("first occurrence", "important relationships", "named entities", "output only", "key events", "search for subqueries")):
            continue
        if " or " in lower and "?" not in candidate:
            continue
        if len(re.findall(r"[A-Za-z0-9]+", candidate)) < 3 and "?" not in candidate:
            continue
        if claim_tokens:
            query_tokens = set(re.findall(r"[A-Zaz0-9]+", candidate.lower()))
            if not query_tokens:
                continue
            if not (query_tokens & claim_tokens) and "?" not in candidate:
                continue
        items.append(candidate)

    return remove_duplicate_queries(items)


def generate_queries_with_smol(claim, max_new_tokens=180):
    if not isinstance(claim, str) or not claim.strip():
        return []

    try:
        from src.query_generation.smol_client import generate_smol_text
    except Exception:
        return []

    try:
        prompt = _build_model_prompt(claim)
        result = generate_smol_text(prompt, max_new_tokens=max_new_tokens)
        parsed = _parse_smol_output(result, claim)
        if parsed:
            return parsed[:5]
    except Exception:
        pass

    return _keyword_fallback_queries(claim)[:5]

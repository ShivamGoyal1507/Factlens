"""
Try the query-generation model on your own claims.

Usage (from the project root, in the `factlens` conda env):

    python scripts/try_query_generation.py                  # runs the CLAIMS list below
    python scripts/try_query_generation.py "Your claim here" # runs one claim
    python scripts/try_query_generation.py -i               # type claims interactively

Change the model with:  FACTLENS_QUERY_MODEL=Qwen/Qwen2.5-3B-Instruct python scripts/...
"""

import os
import sys
import time

# Avoid the macOS "libomp already initialized" crash with torch.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# Make `src` importable when run as a script.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.query_generation import generate_queries  # noqa: E402
from src.query_generation.complexity import (  # noqa: E402
    estimate_claim_complexity,
    get_max_queries,
)

# ---------------------------------------------------------------------------
# Write your own claims here
# ---------------------------------------------------------------------------
CLAIMS = [
    # Simple claims
    "The Earth revolves around the Sun.",
    "India won the ICC Men's T20 World Cup in 2024.",

    # Moderate claims
    "India became the world's fifth-largest economy in 2022, surpassing the United Kingdom.",
    "WHO declared COVID-19 a global pandemic on 11 March 2020.",

    # Complex claims
    "India's Chandrayaan-3 mission became the first mission to successfully land near the Moon's south pole in August 2023, making India the fourth country to achieve a soft landing on the Moon.",

    "A 2024 study published by Stanford University found that artificial intelligence systems consistently outperform human doctors in diagnosing diseases across all medical specialties, including radiology, oncology, and cardiology."
]


def run(claim):
    complexity = estimate_claim_complexity(claim)
    start = time.perf_counter()
    queries = generate_queries(claim)
    elapsed = time.perf_counter() - start

    print(f"\nClaim      : {claim}")
    print(f"Complexity : {complexity} (max {get_max_queries(complexity)} queries)")
    print(f"Time       : {elapsed:.1f}s")
    for i, query in enumerate(queries, 1):
        print(f"  {i}. {query}")


def main():
    args = sys.argv[1:]

    if args and args[0] in ("-i", "--interactive"):
        print("Enter a claim (empty line to quit).")
        while True:
            claim = input("\nclaim> ").strip()
            if not claim:
                break
            run(claim)
        return

    for claim in (args or CLAIMS):
        run(claim)


if __name__ == "__main__":
    main()

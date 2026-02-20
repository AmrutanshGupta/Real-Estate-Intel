#!/usr/bin/env python3
"""
eval/evaluate.py — Automated retrieval quality evaluation.

Generates 20 test questions, queries the running API, and reports:
  - Top-1 accuracy  (correct answer in position 1)
  - Top-3 accuracy  (correct answer in top 3)
  - Average + P95 latency

Usage:
    python eval/evaluate.py
    python eval/evaluate.py --host http://localhost:5000 --k 5
"""

import argparse
import json
import statistics
import sys
import time
from dataclasses import dataclass, field

import requests


# ── Test data ────────────────────────────────────────────────────────────────
# Each question has:
#   - query:    natural-language question
#   - keywords: list of strings that MUST appear in a correct result's text
#               (case-insensitive substring match — no exact string required)
TEST_QUESTIONS = [
    # Property fundamentals
    {"query": "What are the nearby landmarks or attractions?",
     "keywords": ["landmark", "near", "located", "proximity", "adjacent", "nearby"]},
    {"query": "What is the total area or size of the property?",
     "keywords": ["sq ft", "sqft", "square feet", "area", "carpet", "super built", "acre"]},
    {"query": "What is the price or cost of the property?",
     "keywords": ["price", "cost", "lakh", "crore", "₹", "rs.", "amount", "value"]},
    {"query": "How many bedrooms or BHK configuration does the property have?",
     "keywords": ["bhk", "bedroom", "2bhk", "3bhk", "4bhk", "bed room"]},
    {"query": "What floor is the apartment on?",
     "keywords": ["floor", "storey", "level", "ground", "first", "second", "top"]},

    # Location and connectivity
    {"query": "What is the address or location of the project?",
     "keywords": ["sector", "noida", "delhi", "gurgaon", "mumbai", "bangalore", "road", "street", "plot"]},
    {"query": "What is the distance from the nearest metro station or railway station?",
     "keywords": ["metro", "railway", "station", "km", "minutes", "drive", "transit"]},
    {"query": "What schools or educational institutions are nearby?",
     "keywords": ["school", "college", "university", "education", "institute", "academy"]},
    {"query": "What hospitals or medical facilities are nearby?",
     "keywords": ["hospital", "medical", "clinic", "health", "doctor", "pharmacy"]},
    {"query": "What shopping malls or commercial areas are close?",
     "keywords": ["mall", "market", "shop", "commercial", "retail", "store"]},

    # Amenities
    {"query": "What amenities does the project offer?",
     "keywords": ["amenities", "swimming", "gym", "club", "park", "parking", "security", "power"]},
    {"query": "Is there parking available?",
     "keywords": ["parking", "car", "vehicle", "garage", "basement", "stilt"]},
    {"query": "What security features are provided?",
     "keywords": ["security", "guard", "cctv", "surveillance", "gated", "intercom", "24"]},
    {"query": "Is there a swimming pool or recreational facilities?",
     "keywords": ["swimming", "pool", "gym", "recreation", "sport", "club", "fitness"]},
    {"query": "What green or environment-friendly features are present?",
     "keywords": ["green", "garden", "landscape", "tree", "park", "environment", "rain water"]},

    # Legal and financial
    {"query": "What is the possession or handover date?",
     "keywords": ["possession", "ready", "handover", "completion", "deliver", "2024", "2025", "2026"]},
    {"query": "What is the RERA registration number?",
     "keywords": ["rera", "registration", "number", "approved", "authority"]},
    {"query": "Who is the developer or builder of the project?",
     "keywords": ["developer", "builder", "group", "pvt", "limited", "ltd", "promoter", "realty"]},
    {"query": "What are the payment plan or financing options?",
     "keywords": ["payment", "plan", "emi", "loan", "finance", "instalment", "scheme", "bank"]},
    {"query": "What is the maintenance charge or society fee?",
     "keywords": ["maintenance", "charge", "society", "fee", "monthly", "annual", "corpus"]},
]


# ── Helpers ──────────────────────────────────────────────────────────────────

@dataclass
class Result:
    query:      str
    keywords:   list[str]
    hits:       list[dict]
    latency_ms: float
    top1_pass:  bool = field(default=False, init=False)
    top3_pass:  bool = field(default=False, init=False)

    def __post_init__(self):
        def _match(text: str) -> bool:
            t = text.lower()
            return any(kw.lower() in t for kw in self.keywords)

        hits_text = [h.get("text", "") for h in self.hits]
        self.top1_pass = bool(hits_text) and _match(hits_text[0])
        self.top3_pass = any(_match(t) for t in hits_text[:3])


def _run_query(host: str, query: str, k: int) -> tuple[list[dict], float]:
    t0 = time.perf_counter()
    r  = requests.post(
        f"{host}/api/search",
        json={"query": query, "k": k},
        timeout=30,
    )
    r.raise_for_status()
    data = r.json()
    return data.get("results", []), data.get("latency_ms", (time.perf_counter() - t0) * 1000)


# ── Main ─────────────────────────────────────────────────────────────────────

def main(host: str, k: int, output: str | None) -> None:
    print("\n" + "━" * 62)
    print(f"  Real Estate Intel — Retrieval Evaluation")
    print(f"  Host: {host}   K: {k}   Questions: {len(TEST_QUESTIONS)}")
    print("━" * 62)

    # Health check
    try:
        r = requests.get(f"{host}/api/health", timeout=10)
        h = r.json()
        if not h.get("ready"):
            print(f"\n  ✗  API is up but index is empty.  Run ingest.py first.\n")
            sys.exit(1)
        print(f"\n  Index: {h['vectors']} vectors  |  {h['documents']} document(s)")
    except Exception as exc:
        print(f"\n  ✗  Cannot reach {host}: {exc}\n")
        sys.exit(1)

    results: list[Result] = []
    latencies: list[float] = []

    print()
    for i, q in enumerate(TEST_QUESTIONS, 1):
        try:
            hits, ms = _run_query(host, q["query"], k)
            r = Result(
                query=q["query"],
                keywords=q["keywords"],
                hits=hits,
                latency_ms=ms,
            )
            results.append(r)
            latencies.append(ms)

            status = "✓" if r.top1_pass else ("~" if r.top3_pass else "✗")
            print(f"  [{status}] Q{i:02d}  {ms:6.1f} ms  {q['query'][:55]}")

        except Exception as exc:
            print(f"  [!] Q{i:02d}  ERROR: {exc}")

    if not results:
        print("\n  No results to evaluate.\n")
        sys.exit(1)

    # ── Summary ──────────────────────────────────────────────────────────────
    n      = len(results)
    top1   = sum(r.top1_pass for r in results)
    top3   = sum(r.top3_pass for r in results)
    avg_ms = statistics.mean(latencies)
    p95_ms = sorted(latencies)[int(0.95 * len(latencies)) - 1]

    print("\n" + "─" * 62)
    print(f"  Top-1 Accuracy : {top1}/{n}  ({top1/n*100:.1f}%)")
    print(f"  Top-3 Accuracy : {top3}/{n}  ({top3/n*100:.1f}%)")
    print(f"  Avg Latency    : {avg_ms:.1f} ms")
    print(f"  P95 Latency    : {p95_ms:.1f} ms")
    print("─" * 62)
    print("  Legend: ✓ Top-1 hit  ~  Top-3 hit  ✗ Miss")
    print("━" * 62 + "\n")

    # ── Optional JSON output ──────────────────────────────────────────────────
    report = {
        "summary": {
            "questions":   n,
            "top1":        top1,
            "top3":        top3,
            "top1_pct":    round(top1 / n * 100, 1),
            "top3_pct":    round(top3 / n * 100, 1),
            "avg_latency_ms": round(avg_ms, 1),
            "p95_latency_ms": round(p95_ms, 1),
        },
        "questions": [
            {
                "query":      r.query,
                "top1_pass":  r.top1_pass,
                "top3_pass":  r.top3_pass,
                "latency_ms": round(r.latency_ms, 1),
                "top_result": r.hits[0].get("text", "")[:200] if r.hits else "",
            }
            for r in results
        ],
    }

    if output:
        with open(output, "w") as fh:
            json.dump(report, fh, indent=2)
        print(f"  Report saved → {output}\n")

    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--host",   default="http://localhost:5000")
    p.add_argument("--k",      default=5, type=int)
    p.add_argument("--output", default="eval/report.json")
    args = p.parse_args()
    main(args.host, args.k, args.output)

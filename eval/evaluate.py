#!/usr/bin/env python3
"""
eval/evaluate.py — Automated retrieval quality evaluation.

Generates test queries against the API and reports mandatory Agmentis metrics:
  - Recall@1 / Recall@3
  - MRR (Mean Reciprocal Rank)
  - nDCG (Normalized Discounted Cumulative Gain)
  - Entity Coverage Score
  - False Positive Rate
  - Stage-wise Latency Breakdown
"""

import argparse
import json
import math
import statistics
import sys
import time
from dataclasses import dataclass, field

import requests

# Exact questions provided by the user, mapped with evaluation heuristics
TEST_QUESTIONS = [
    # SECTION A: 222 RAJPUR, DEHRADUN
    {"query": "For 222 Rajpur, Dehradun, how many total residences are planned and over how many acres is the project spread?", "keywords": ["residences", "acres", "spread", "total"], "entities": ["rajpur"]},
    {"query": "At 222 Rajpur, what types of residences are available?", "keywords": ["types", "residences", "villa", "townhouse"], "entities": ["rajpur"]},
    {"query": "Is 222 Rajpur adjacent to any forest area? If yes, which one?", "keywords": ["adjacent", "forest", "malsi"], "entities": ["rajpur"]},
    {"query": "What are the views offered from residences at 222 Rajpur, Dehradun?", "keywords": ["views", "scenic", "mountains", "forest"], "entities": ["rajpur"]},
    {"query": "How far is Jolly Grant Airport from 222 Rajpur, Dehradun?", "keywords": ["jolly grant", "airport", "distance", "km", "minutes"], "entities": ["rajpur"]},
    {"query": "What is the distance between 222 Rajpur and The Doon School?", "keywords": ["doon school", "distance", "km"], "entities": ["rajpur"]},
    {"query": "How long does it take to reach Pacific Mall from 222 Rajpur?", "keywords": ["pacific mall", "reach", "minutes", "time"], "entities": ["rajpur"]},
    {"query": "Is 222 Rajpur close to Max Super Specialty Hospital?", "keywords": ["max super specialty", "hospital", "close", "distance"], "entities": ["rajpur"]},
    {"query": "How many Townhouse units are available at 222 Rajpur?", "keywords": ["townhouse", "units", "how many"], "entities": ["rajpur"]},
    {"query": "What is the built-up area and carpet area of a Townhouse at 222 Rajpur?", "keywords": ["built-up", "carpet", "area", "sq", "ft"], "entities": ["rajpur"]},
    {"query": "Does the Townhouse at 222 Rajpur include a sky court?", "keywords": ["sky court", "townhouse"], "entities": ["rajpur"]},
    {"query": "What is the ceiling height in the Townhouse units at 222 Rajpur?", "keywords": ["ceiling height", "feet", "meters"], "entities": ["rajpur"]},
    {"query": "How many parking spaces are provided with each Townhouse at 222 Rajpur?", "keywords": ["parking", "spaces", "car"], "entities": ["rajpur"]},
    {"query": "How many Courtyard Villas are available at 222 Rajpur?", "keywords": ["courtyard", "villas", "how many"], "entities": ["rajpur"]},
    {"query": "What is the plot size range for Courtyard Villas at 222 Rajpur?", "keywords": ["plot size", "range", "sq", "yds"], "entities": ["rajpur"]},
    {"query": "Do the Courtyard Villas at 222 Rajpur include staff accommodation?", "keywords": ["staff", "accommodation", "servant", "room"], "entities": ["rajpur"]},
    {"query": "What is the terrace size of a Courtyard Villa at 222 Rajpur?", "keywords": ["terrace", "size", "area"], "entities": ["rajpur"]},
    {"query": "How many Forest Villas are available at 222 Rajpur?", "keywords": ["forest villas", "how many"], "entities": ["rajpur"]},
    {"query": "What is the built-up area of a Forest Villa at 222 Rajpur?", "keywords": ["built-up", "area", "sq", "ft"], "entities": ["rajpur"]},
    {"query": "Do Forest Villas at 222 Rajpur have private elevators?", "keywords": ["private", "elevators", "lift"], "entities": ["rajpur"]},
    {"query": "What special landscape feature is included in the lower ground floor of Forest Villas at 222 Rajpur?", "keywords": ["landscape", "lower ground", "feature"], "entities": ["rajpur"]},
    {"query": "Does 222 Rajpur provide round-the-clock security?", "keywords": ["security", "round-the-clock", "24/7", "cctv"], "entities": ["rajpur"]},
    {"query": "What wellness or nature-focused amenities are offered at 222 Rajpur?", "keywords": ["wellness", "nature", "amenities", "pool", "gym", "spa"], "entities": ["rajpur"]},
    {"query": "Does 222 Rajpur offer power backup and uninterrupted water supply?", "keywords": ["power backup", "water supply", "uninterrupted"], "entities": ["rajpur"]},
    {"query": "Is there a private orchard at 222 Rajpur?", "keywords": ["private", "orchard"], "entities": ["rajpur"]},

    # SECTION B: MAX TOWERS, NOIDA
    {"query": "What is the total super built-up area of Max Towers, Noida?", "keywords": ["super built-up", "area", "sq", "ft"], "entities": ["towers"]},
    {"query": "How many office floors and amenity floors are there in Max Towers?", "keywords": ["office floors", "amenity floors"], "entities": ["towers"]},
    {"query": "What is the typical floor plate size at Max Towers?", "keywords": ["typical", "floor plate", "size", "sq", "ft"], "entities": ["towers"]},
    {"query": "What is the floor-to-floor height at Max Towers?", "keywords": ["floor-to-floor", "height", "meters", "feet"], "entities": ["towers"]},
    {"query": "What green rating has Max Towers achieved?", "keywords": ["green rating", "leed", "platinum", "gold"], "entities": ["towers"]},
    {"query": "Does Max Towers offer on-site wastewater treatment?", "keywords": ["wastewater", "treatment", "stp"], "entities": ["towers"]},
    {"query": "What is the coefficient of performance (COP) of the chiller system at Max Towers?", "keywords": ["coefficient", "performance", "cop", "chiller"], "entities": ["towers"]},
    {"query": "Does Max Towers support electric vehicle parking?", "keywords": ["electric vehicle", "ev", "parking", "charging"], "entities": ["towers"]},
    {"query": "Does Max Towers have a swimming pool?", "keywords": ["swimming pool", "pool"], "entities": ["towers"]},
    {"query": "What kind of fitness facilities are available at Max Towers?", "keywords": ["fitness", "gym", "health"], "entities": ["towers"]},
    {"query": "Does Max Towers provide daycare facilities?", "keywords": ["daycare", "creche", "child"], "entities": ["towers"]},
    {"query": "What air treatment system is used in Max Towers?", "keywords": ["air treatment", "hvac", "filtration", "purification"], "entities": ["towers"]},
    {"query": "Where is Max Towers located?", "keywords": ["located", "address", "sector", "noida"], "entities": ["towers"]},
    {"query": "Is Max Towers within walking distance of a metro station?", "keywords": ["walking distance", "metro", "station", "sector 16"], "entities": ["towers"]},
    {"query": "Does Max Towers have direct access to the DND Flyway?", "keywords": ["direct access", "dnd", "flyway"], "entities": ["towers"]},
    {"query": "What type of façade glass is used in Max Towers?", "keywords": ["façade", "glass", "glazing", "dgu"], "entities": ["towers"]},
    {"query": "What is the solar heat gain coefficient of the façade at Max Towers?", "keywords": ["solar heat gain", "coefficient", "shgc"], "entities": ["towers"]},
    {"query": "What percentage of regular occupied space at Max Towers gets line-of-sight to the outside?", "keywords": ["percentage", "occupied space", "line-of-sight", "outside"], "entities": ["towers"]},

    # SECTION C: MAX HOUSE, OKHLA
    {"query": "What is the total super built-up area of Max House, Okhla?", "keywords": ["super built-up", "area", "sq", "ft"], "entities": ["house"]},
    {"query": "How many tenant floors are there in Max House?", "keywords": ["tenant floors", "floors"], "entities": ["house"]},
    {"query": "What is the typical floor plate size at Max House?", "keywords": ["typical", "floor plate", "size", "sq", "ft"], "entities": ["house"]},
    {"query": "What is the green rating of Max House?", "keywords": ["green rating", "leed", "gold", "platinum"], "entities": ["house"]},
    {"query": "How far is Max House, Okhla from the Okhla NSIC Metro Station?", "keywords": ["distance", "okhla nsic", "metro", "station"], "entities": ["house"]},
    {"query": "How far is Max House from IGI Airport?", "keywords": ["distance", "igi", "airport", "km"], "entities": ["house"]},
    {"query": "Is Max House within walking distance of a metro station?", "keywords": ["walking distance", "metro", "station"], "entities": ["house"]},
    {"query": "What façade material is used in Max House?", "keywords": ["façade", "material", "brick", "glass"], "entities": ["house"]},
    {"query": "What is the floor-to-ceiling height at Max House?", "keywords": ["floor-to-ceiling", "height", "meters", "feet"], "entities": ["house"]},
    {"query": "Does Max House use double-glazed windows?", "keywords": ["double-glazed", "windows", "dgu"], "entities": ["house"]},
    {"query": "What air treatment technology is used in Max House?", "keywords": ["air treatment", "filtration", "merv"], "entities": ["house"]},
    {"query": "Is Max House LEED certified?", "keywords": ["leed", "certified", "gold"], "entities": ["house"]},
    {"query": "Does Max House incorporate biophilic design principles?", "keywords": ["biophilic", "design", "nature"], "entities": ["house"]},

    # SECTION D: CROSS-PROPERTY COMPARISON QUESTIONS
    {"query": "Which property among 222 Rajpur, Max Towers, and Max House is purely residential?", "keywords": ["residential", "rajpur", "towers", "house"], "entities": ["rajpur", "towers", "house"]},
    {"query": "Which property is located in Dehradun: 222 Rajpur or Max Towers?", "keywords": ["dehradun", "rajpur", "towers"], "entities": ["rajpur", "towers"]},
    {"query": "Between Max Towers and Max House, which one has a higher LEED certification?", "keywords": ["higher", "leed", "platinum", "gold"], "entities": ["towers", "house"]},
    {"query": "Compare the typical floor plate size of Max Towers and Max House.", "keywords": ["floor plate", "size", "sq", "ft"], "entities": ["towers", "house"]},
    {"query": "Which property has a larger total built-up area: Max Towers or Max House?", "keywords": ["larger", "built-up", "area"], "entities": ["towers", "house"]},
    {"query": "Which property has more tenant floors: Max Towers or Max House?", "keywords": ["more", "tenant floors", "floors"], "entities": ["towers", "house"]},
    {"query": "Which property is closer to a metro station: Max House or Max Towers?", "keywords": ["closer", "metro", "station", "distance"], "entities": ["towers", "house"]},
    {"query": "Between 222 Rajpur and Max House, which property is closer to an airport?", "keywords": ["closer", "airport", "distance", "km"], "entities": ["rajpur", "house"]},
    {"query": "Which property offers direct access to the DND Flyway: Max Towers or Max House?", "keywords": ["direct access", "dnd", "flyway"], "entities": ["towers", "house"]},
    {"query": "Which property has LEED Platinum certification: Max Towers or Max House?", "keywords": ["leed platinum", "certification"], "entities": ["towers", "house"]},
    {"query": "Which properties use advanced air treatment systems: Max Towers, Max House, or both?", "keywords": ["advanced", "air treatment", "system", "both"], "entities": ["towers", "house"]},
    {"query": "Which property offers on-site wastewater treatment: Max Towers or Max House?", "keywords": ["on-site", "wastewater", "treatment", "stp"], "entities": ["towers", "house"]},
    {"query": "Which property offers a swimming pool: Max Towers or Max House?", "keywords": ["swimming pool", "pool"], "entities": ["towers", "house"]},
    {"query": "Does 222 Rajpur offer wellness amenities comparable to Max Towers?", "keywords": ["wellness", "amenities", "compare"], "entities": ["rajpur", "towers"]},
    {"query": "Which property explicitly mentions decompression spaces: Max Towers or Max House?", "keywords": ["decompression", "spaces"], "entities": ["towers", "house"]},

    # SECTION E: Real-World Client Simulation Questions
    {"query": "I’m looking for a 4-bedroom villa with staff accommodation in 222 Rajpur — which unit type should I consider?", "keywords": ["4-bedroom", "villa", "staff", "accommodation", "courtyard"], "entities": ["rajpur"]},
    {"query": "My company needs a 25,000 sq. ft. office in Noida — can Max Towers accommodate this on a single floor?", "keywords": ["25,000", "sq. ft.", "office", "noida", "single floor", "floor plate"], "entities": ["towers"]},
    {"query": "We want an office in Delhi with LEED Gold certification — is Max House, Okhla suitable?", "keywords": ["office", "delhi", "leed gold", "okhla"], "entities": ["house"]},
    {"query": "I want a residential property near the forest with private garden space — does 222 Rajpur offer this?", "keywords": ["residential", "forest", "private garden"], "entities": ["rajpur"]},
    {"query": "We are a wellness-focused company — between Max Towers and Max House, which better supports employee wellbeing?", "keywords": ["wellness", "employee wellbeing", "amenities"], "entities": ["towers", "house"]},
    {"query": "I need an office within walking distance of the metro in Delhi — is Max House a good option?", "keywords": ["office", "walking distance", "metro", "delhi", "okhla nsic"], "entities": ["house"]},
    {"query": "Which property among 222 Rajpur, Max Towers, and Max House offers private elevators?", "keywords": ["private elevators", "lift"], "entities": ["rajpur", "towers", "house"]},
    {"query": "If sustainability is a top priority, should I choose Max Towers or Max House?", "keywords": ["sustainability", "leed", "green"], "entities": ["towers", "house"]},
    {"query": "I need a property with daycare facilities — which of these three properties provides that?", "keywords": ["daycare", "creche"], "entities": ["rajpur", "towers", "house"]},

    # SECTION F: PARAPHRASE ROBUSTNESS
    {"query": "Which of the three developments is a housing project rather than an office building?", "keywords": ["housing", "office", "residential"], "entities": ["rajpur", "towers", "house"]},
    {"query": "Identify the project that is not meant for commercial office use.", "keywords": ["not commercial", "office", "residential"], "entities": ["rajpur", "towers", "house"]},
    {"query": "Among the three properties, which one is exclusively residential in nature?", "keywords": ["exclusively", "residential"], "entities": ["rajpur", "towers", "house"]},
    {"query": "Between Max Towers and Max House, which holds the higher level of LEED certification?", "keywords": ["higher level", "leed", "platinum", "gold"], "entities": ["towers", "house"]},
    {"query": "If sustainability certification level is the deciding factor, which property ranks highest?", "keywords": ["sustainability", "highest", "leed", "platinum"], "entities": ["towers", "house"]},
    {"query": "Which development has achieved Platinum-level green certification?", "keywords": ["platinum", "green certification"], "entities": ["towers", "house"]},
    {"query": "Which office property can employees walk to from a metro station?", "keywords": ["walk", "metro station", "employees"], "entities": ["towers", "house"]},
    {"query": "Identify the development located within walking distance of a metro stop.", "keywords": ["walking distance", "metro stop"], "entities": ["towers", "house"]},
    {"query": "Between the Noida and Okhla projects, which one offers closer metro access?", "keywords": ["noida", "okhla", "closer", "metro"], "entities": ["towers", "house"]},
    {"query": "Which project is larger in overall constructed area: Max Towers or Max House?", "keywords": ["larger", "constructed area", "built-up"], "entities": ["towers", "house"]},
    {"query": "Between the Delhi and Noida office developments, which spans more total square footage?", "keywords": ["delhi", "noida", "more", "square footage"], "entities": ["towers", "house"]},
    {"query": "Which property has the greater overall scale in terms of built-up space?", "keywords": ["greater", "scale", "built-up"], "entities": ["towers", "house"]},
    {"query": "Which property includes an indoor swimming facility?", "keywords": ["indoor", "swimming", "pool"], "entities": ["towers", "house"]},
    {"query": "Identify the development that provides decompression or relaxation spaces.", "keywords": ["decompression", "relaxation", "spaces"], "entities": ["towers", "house"]},
    {"query": "If employee wellness is a priority, which property explicitly supports it through facilities?", "keywords": ["employee wellness", "supports", "facilities"], "entities": ["towers", "house"]},

    # SECTION G: NEGATIVE / ADVERSARIAL QUESTIONS
    {"query": "Which property among the three includes a helipad?", "keywords": [], "entities": [], "is_negative": True},
    {"query": "Which development offers a golf course within the premises?", "keywords": [], "entities": [], "is_negative": True},
    {"query": "Is any of the properties located in Mumbai?", "keywords": [], "entities": [], "is_negative": True},
    {"query": "Which project provides co-living or serviced apartments?", "keywords": [], "entities": [], "is_negative": True},
    {"query": "What is the rental yield percentage of Max Towers?", "keywords": [], "entities": [], "is_negative": True},
    {"query": "Which property has a shopping mall attached to it?", "keywords": [], "entities": [], "is_negative": True},
    {"query": "Do any of the properties include a five-star hotel?", "keywords": [], "entities": [], "is_negative": True},
    {"query": "Which development offers beachfront views?", "keywords": [], "entities": [], "is_negative": True},
    {"query": "Is there a data center facility mentioned in any of the properties?", "keywords": [], "entities": [], "is_negative": True},
    {"query": "Which property includes an amusement park or entertainment zone?", "keywords": [], "entities": [], "is_negative": True},

    # SECTION H: AMBIGUOUS / CLARIFICATION TEST QUESTIONS
    {"query": "What is the total area?", "keywords": ["total area", "sq ft", "acre"], "entities": []},
    {"query": "How many floors does it have?", "keywords": ["floors", "levels", "storey"], "entities": []},
    {"query": "What certification does it hold?", "keywords": ["certification", "leed", "rating", "green"], "entities": []},
    {"query": "How far is it from the airport?", "keywords": ["far", "airport", "distance", "km"], "entities": []},
    {"query": "Does it offer parking?", "keywords": ["parking", "spaces", "car"], "entities": []}
]

@dataclass
class Result:
    query: str
    keywords: list[str]
    entities: list[str]
    is_negative: bool
    hits: list[dict]
    latency: dict
    
    # Metrics
    recall_at_1: bool = field(default=False, init=False)
    recall_at_3: bool = field(default=False, init=False)
    mrr: float = field(default=0.0, init=False)
    ndcg: float = field(default=0.0, init=False)
    entity_coverage: float = field(default=0.0, init=False)
    false_positive: bool = field(default=False, init=False)

    def __post_init__(self):
        if self.is_negative:
            self.false_positive = len(self.hits) > 0
            return

        hits_text = [h.get("text", "").lower() for h in self.hits]
        relevance = [1 if any(kw.lower() in t for kw in self.keywords) else 0 for t in hits_text]

        if relevance:
            self.recall_at_1 = bool(relevance[0])
            self.recall_at_3 = any(relevance[:3])
            
            try:
                first_hit_idx = relevance.index(1)
                self.mrr = 1.0 / (first_hit_idx + 1)
            except ValueError:
                self.mrr = 0.0

            dcg = sum(rel / math.log2(idx + 2) for idx, rel in enumerate(relevance))
            idcg = sum(1.0 / math.log2(idx + 2) for idx, rel in enumerate(sorted(relevance, reverse=True)))
            self.ndcg = dcg / idcg if idcg > 0 else 0.0

            # Evaluate Entity Coverage: Checks if chunks from BOTH expected properties were returned
            if hits_text and self.entities:
                sources_returned = [h.get("source", "").lower() for h in self.hits]
                found = sum(1 for e in self.entities if any(e.lower() in src for src in sources_returned))
                self.entity_coverage = found / len(self.entities)

def _run_query(host: str, query: str, k: int) -> tuple[list[dict], dict]:
    t0 = time.perf_counter()
    r = requests.post(
        f"{host}/api/search",
        json={"query": query, "k": k},
        timeout=30,
    )
    r.raise_for_status()
    data = r.json()
    
    latency = data.get("latency", {})
    if not latency:
        # Fallback for older API format
        latency = {"total": (time.perf_counter() - t0) * 1000}
        
    return data.get("results", []), latency

def main(host: str, k: int, output: str | None) -> None:
    print("\n" + "━" * 65)
    print(f"  Agmentis Benchmark Evaluation")
    print(f"  Host: {host}   K: {k}   Questions: {len(TEST_QUESTIONS)}")
    print("━" * 65)

    try:
        r = requests.get(f"{host}/api/health", timeout=10)
        h = r.json()
        if not h.get("ready"):
            print(f"\n  [!] API is up but index is empty. Run ingestion first.\n")
            sys.exit(1)
        print(f"\n  Index: {h.get('vectors', 0)} vectors | {h.get('documents', 0)} document(s)")
    except Exception as exc:
        print(f"\n  [!] Cannot reach {host}: {exc}\n")
        sys.exit(1)

    results: list[Result] = []

    print()
    for i, q in enumerate(TEST_QUESTIONS, 1):
        try:
            hits, latency = _run_query(host, q["query"], k)
            r = Result(
                query=q["query"],
                keywords=q.get("keywords", []),
                entities=q.get("entities", []),
                is_negative=q.get("is_negative", False),
                hits=hits,
                latency=latency,
            )
            results.append(r)

            if r.is_negative:
                status = "FP" if r.false_positive else "TN"
                print(f"  [{status}] Q{i:02d}  {latency.get('total', 0):6.1f} ms  {q['query'][:50]}...")
            else:
                status = "✓" if r.recall_at_1 else ("~" if r.recall_at_3 else "✗")
                print(f"  [{status}] Q{i:02d}  {latency.get('total', 0):6.1f} ms  {q['query'][:50]}...")

        except Exception as exc:
            print(f"  [!] Q{i:02d}  ERROR: {exc}")

    if not results:
        print("\n  No results to evaluate.\n")
        sys.exit(1)

    # Calculate Aggregates
    standard_results = [r for r in results if not r.is_negative]
    negative_results = [r for r in results if r.is_negative]
    
    n = max(1, len(standard_results))
    neg_n = max(1, len(negative_results))

    metrics = {
        "recall_1": sum(r.recall_at_1 for r in standard_results) / n * 100,
        "recall_3": sum(r.recall_at_3 for r in standard_results) / n * 100,
        "mrr": sum(r.mrr for r in standard_results) / n,
        "ndcg": sum(r.ndcg for r in standard_results) / n,
        "entity_coverage": sum(r.entity_coverage for r in standard_results) / n * 100,
        "false_positive_rate": sum(r.false_positive for r in negative_results) / neg_n * 100,
    }

    # Latency Aggregates
    totals = [r.latency.get("total", 0) for r in results]
    embeds = [r.latency.get("embedding", 0) for r in results]
    retrievals = [r.latency.get("retrieval", 0) for r in results]
    reranks = [r.latency.get("reranking", 0) for r in results]

    print("\n" + "─" * 65)
    print("  RETRIEVAL METRICS")
    print(f"  Recall@1            : {metrics['recall_1']:.1f}%")
    print(f"  Recall@3            : {metrics['recall_3']:.1f}%")
    print(f"  MRR                 : {metrics['mrr']:.4f}")
    print(f"  nDCG                : {metrics['ndcg']:.4f}")
    print(f"  Entity Coverage     : {metrics['entity_coverage']:.1f}%")
    print(f"  False Positive Rate : {metrics['false_positive_rate']:.1f}%")
    
    print("\n  LATENCY BREAKDOWN (Avg)")
    print(f"  Embedding           : {statistics.mean(embeds):.1f} ms")
    print(f"  Retrieval           : {statistics.mean(retrievals):.1f} ms")
    print(f"  Re-ranking          : {statistics.mean(reranks):.1f} ms")
    print(f"  Total (P95)         : {sorted(totals)[int(0.95 * len(totals)) - 1]:.1f} ms")
    print("─" * 65)

    report = {
        "summary": {
            "total_queries": len(results),
            "metrics": {k: round(v, 4) for k, v in metrics.items()},
            "latency_ms": {
                "avg_embedding": round(statistics.mean(embeds), 1),
                "avg_retrieval": round(statistics.mean(retrievals), 1),
                "avg_reranking": round(statistics.mean(reranks), 1),
                "avg_total": round(statistics.mean(totals), 1),
                "p95_total": round(sorted(totals)[int(0.95 * len(totals)) - 1], 1)
            }
        },
        "questions": [
            {
                "query": r.query,
                "is_negative": r.is_negative,
                "recall_at_1": r.recall_at_1,
                "recall_at_3": r.recall_at_3,
                "mrr": round(r.mrr, 4),
                "ndcg": round(r.ndcg, 4),
                "latency_ms": r.latency,
                "top_result": r.hits[0].get("text", "")[:200] if r.hits else "",
            }
            for r in results
        ],
    }

    if output:
        with open(output, "w") as fh:
            json.dump(report, fh, indent=2)
        print(f"  Report saved → {output}\n")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="http://localhost:5000")
    p.add_argument("--k", default=5, type=int)
    p.add_argument("--output", default="eval/report.json")
    args = p.parse_args()
    main(args.host, args.k, args.output)
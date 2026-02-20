#!/usr/bin/env python3
"""
benchmark.py — Latency profiler for the search API.

Runs QUERIES × RUNS_PER_QUERY and prints per-query and overall stats.
"""

import asyncio
import statistics
import sys
import time

QUERIES = [
    "What are the nearby landmarks?",
    "What is the total carpet area?",
    "What is the possession date?",
    "Who is the developer of the project?",
    "What amenities does the project provide?",
    "Is parking available?",
    "What floor is the property on?",
    "What is the RERA registration number?",
    "What is the price of the property?",
    "What hospitals are nearby?",
]

RUNS_PER_QUERY = 5


def run() -> None:
    import requests

    host = "http://localhost:5000"
    print("\n" + "━" * 56)
    print(f"  Real Estate Intel — Latency Benchmark")
    print("━" * 56)

    try:
        requests.get(f"{host}/api/health", timeout=5).raise_for_status()
    except Exception as e:
        print(f"\n  ✗  API not reachable: {e}")
        print("     Start the server first:  python run.py\n")
        sys.exit(1)

    all_ms: list[float] = []

    for query in QUERIES:
        times: list[float] = []
        for _ in range(RUNS_PER_QUERY):
            t0 = time.perf_counter()
            r  = requests.post(
                f"{host}/api/search",
                json={"query": query, "k": 5},
                timeout=30,
            )
            times.append((time.perf_counter() - t0) * 1000)

        mean = statistics.mean(times)
        p95  = sorted(times)[int(0.95 * len(times)) - 1]
        all_ms.extend(times)
        print(f"  {mean:6.1f} ms  (p95 {p95:5.1f} ms)  {query[:50]}")

    print("\n" + "─" * 56)
    p95_all = sorted(all_ms)[int(0.95 * len(all_ms)) - 1]
    print(f"  Overall mean : {statistics.mean(all_ms):.1f} ms")
    print(f"  Overall P95  : {p95_all:.1f} ms")
    print("━" * 56 + "\n")


if __name__ == "__main__":
    run()

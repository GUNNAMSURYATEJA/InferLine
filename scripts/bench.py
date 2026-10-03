"""InferLine benchmark suite — runs fully in-process (no manual server startup).

Uses httpx ASGI transport against the FastAPI app with the model pre-loaded,
so measured latencies are pure request-handling + inference. Phases:

  1. warm-up (20 requests, discarded)
  2. single-request latency, n=200  -> mean / p50 / p95 / p99
  3. batch-size sweep (1, 8, 32), n=256 texts each -> latency + throughput
  4. concurrency sweep (1, 4, 16 workers), n=200 each -> throughput + p99

Prints a comparison table and writes results/bench_results.json.
"""

from __future__ import annotations

import json
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient  # noqa: E402

from inferline.app import app, load_model  # noqa: E402

# ~8 short realistic review sentences, cycled through the benchmarks.
SAMPLE_TEXTS = [
    "This movie was absolutely fantastic, I loved every minute of it.",
    "Terrible film. Boring plot, bad acting, I want my money back.",
    "An okay watch — decent performances but nothing memorable.",
    "Brilliant cinematography and a gripping story from start to finish.",
    "I fell asleep halfway through. Completely forgettable.",
    "The cast was great and the jokes actually landed. Highly recommend.",
    "Worst two hours of my life. Avoid at all costs.",
    "Surprisingly good! Much better than I expected it to be.",
]

RESULTS_PATH = Path(__file__).resolve().parent.parent / "results" / "bench_results.json"


def percentile(values: list[float], pct: float) -> float:
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    rank = (pct / 100.0) * (len(s) - 1)
    low = int(rank)
    high = min(low + 1, len(s) - 1)
    return s[low] * (1 - (rank - low)) + s[high] * (rank - low)


def summarize(latencies: list[float]) -> dict:
    return {
        "n": len(latencies),
        "mean_ms": round(statistics.fmean(latencies) * 1000, 2),
        "p50_ms": round(percentile(latencies, 50) * 1000, 2),
        "p95_ms": round(percentile(latencies, 95) * 1000, 2),
        "p99_ms": round(percentile(latencies, 99) * 1000, 2),
    }


def make_client() -> TestClient:
    # Model is pre-loaded by load_model(); no lifespan run needed.
    return TestClient(app, raise_server_exceptions=False)


def phase_single_latency(n: int = 200) -> dict:
    print(f"\n[1/3] single-request latency (n={n}) ...")
    client = make_client()
    latencies = []
    for i in range(n):
        text = SAMPLE_TEXTS[i % len(SAMPLE_TEXTS)]
        t0 = time.perf_counter()
        r = client.post("/predict", json={"text": text})
        latencies.append(time.perf_counter() - t0)
        assert r.status_code == 200, r.text
    client.close()
    return summarize(latencies)


def phase_batch_sweep(batch_sizes=(1, 8, 32), n_texts: int = 256) -> dict:
    print(f"\n[2/3] batch-size sweep (n={n_texts} texts each) ...")
    client = make_client()
    out = {}
    for bs in batch_sizes:
        texts = [SAMPLE_TEXTS[i % len(SAMPLE_TEXTS)] for i in range(n_texts)]
        latencies = []
        t_start = time.perf_counter()
        for i in range(0, n_texts, bs):
            chunk = texts[i : i + bs]
            t0 = time.perf_counter()
            r = client.post("/predict/batch", json={"texts": chunk})
            latencies.append(time.perf_counter() - t0)
            assert r.status_code == 200, r.text
            assert len(r.json()) == len(chunk)
        total = time.perf_counter() - t_start
        s = summarize(latencies)
        s["throughput_texts_per_s"] = round(n_texts / total, 1)
        s["batch_size"] = bs
        out[str(bs)] = s
        print(f"    batch={bs:<3} p50={s['p50_ms']:.1f}ms  throughput={s['throughput_texts_per_s']:.1f} texts/s")
    client.close()
    return out


def _worker(n: int, worker_id: int) -> list[float]:
    client = make_client()
    latencies = []
    for i in range(n):
        text = SAMPLE_TEXTS[(i + worker_id) % len(SAMPLE_TEXTS)]
        t0 = time.perf_counter()
        r = client.post("/predict", json={"text": text})
        latencies.append(time.perf_counter() - t0)
        assert r.status_code == 200, r.text
    client.close()
    return latencies


def phase_concurrency_sweep(workers=(1, 4, 16), n_per_worker: int = 200) -> dict:
    print(f"\n[3/3] concurrency sweep (n={n_per_worker} requests/worker) ...")
    out = {}
    for w in workers:
        t_start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=w) as ex:
            results = list(ex.map(_worker, [n_per_worker] * w, range(w)))
        total = time.perf_counter() - t_start
        latencies = [t for sub in results for t in sub]
        s = summarize(latencies)
        total_reqs = w * n_per_worker
        s["throughput_req_per_s"] = round(total_reqs / total, 1)
        s["workers"] = w
        out[str(w)] = s
        print(
            f"    workers={w:<3} p50={s['p50_ms']:.1f}ms p99={s['p99_ms']:.1f}ms "
            f"throughput={s['throughput_req_per_s']:.1f} req/s"
        )
    return out


def print_table(results: dict) -> None:
    print("\n================ InferLine benchmark ================")
    s = results["single_latency"]
    print(f"model: {results['model']}   threads: {results['torch_threads']}")
    print("\n-- single-request latency (n=200) --")
    print(f"  mean {s['mean_ms']:.2f} ms | p50 {s['p50_ms']:.2f} ms | "
          f"p95 {s['p95_ms']:.2f} ms | p99 {s['p99_ms']:.2f} ms")
    print("\n-- batch sweep (256 texts) --")
    print(f"  {'batch':<6}{'p50 ms':<10}{'throughput':<14}")
    for bs, r in results["batch_sweep"].items():
        print(f"  {bs:<6}{r['p50_ms']:<10.1f}{r['throughput_texts_per_s']:.1f} texts/s")
    print("\n-- concurrency sweep (200 req/worker) --")
    print(f"  {'workers':<8}{'p99 ms':<10}{'throughput':<12}")
    for w, r in results["concurrency_sweep"].items():
        print(f"  {w:<8}{r['p99_ms']:<10.1f}{r['throughput_req_per_s']:.1f} req/s")
    print("=====================================================\n")


def main() -> None:
    import torch

    t0 = time.time()
    print("Loading model ...")
    load_model()
    torch_threads = torch.get_num_threads()

    # Warm-up (discarded).
    client = make_client()
    for i in range(20):
        r = client.post("/predict", json={"text": SAMPLE_TEXTS[i % len(SAMPLE_TEXTS)]})
        assert r.status_code == 200
    client.close()

    results = {
        "model": "distilbert-base-uncased-finetuned-sst-2-english",
        "device": "cpu",
        "torch_threads": torch_threads,
        "single_latency": phase_single_latency(),
        "batch_sweep": phase_batch_sweep(),
        "concurrency_sweep": phase_concurrency_sweep(),
    }
    wall_s = time.time() - t0
    results["wall_clock_s"] = round(wall_s, 1)

    print_table(results)
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved results to {RESULTS_PATH}")
    print(f"Total benchmark wall-clock: {wall_s / 60:.1f} minutes")


if __name__ == "__main__":
    main()

"""InferLine demo: quick end-to-end smoke (<2 min).

Loads the model in-process, runs a few single + batch predictions through the
FastAPI app, and prints the /metrics snapshot. No server startup needed.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient  # noqa: E402

from inferline.app import app, load_model  # noqa: E402


def main() -> None:
    t0 = time.time()
    print("=== InferLine demo ===\nLoading model ...")
    load_model()
    client = TestClient(app)

    texts = [
        "This movie was absolutely fantastic, I loved every minute of it.",
        "Terrible film. Boring plot, bad acting, I want my money back.",
        "An okay watch — decent performances but nothing memorable.",
    ]

    print("\n-- single predictions --")
    for text in texts:
        r = client.post("/predict", json={"text": text})
        r.raise_for_status()
        p = r.json()
        print(f"  [{p['label']:<8} {p['score']:.3f}] {text[:60]}")

    print("\n-- batch prediction --")
    r = client.post("/predict/batch", json={"texts": texts})
    r.raise_for_status()
    for text, p in zip(texts, r.json()):
        print(f"  [{p['label']:<8} {p['score']:.3f}] {text[:60]}")

    print("\n-- validation (expect 422) --")
    r = client.post("/predict", json={"text": "x" * 600})
    print(f"  too-long text -> HTTP {r.status_code}")
    r = client.post("/predict", json={"text": "   "})
    print(f"  blank text     -> HTTP {r.status_code}")

    print("\n-- /metrics snapshot --")
    m = client.get("/metrics").json()["endpoints"]
    for ep, s in sorted(m.items()):
        print(f"  {ep:<16} count={s['count']:<3} p50={s['p50_ms']:.2f}ms p95={s['p95_ms']:.2f}ms")

    client.close()
    print(f"\nDemo finished in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()

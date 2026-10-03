# InferLine

![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)
![CPU-only](https://img.shields.io/badge/inference-CPU--only-lightgrey.svg)

A **production-style inference server** for a small transformer on CPU:
[DistilBERT](https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english)
(sentiment classifier) behind a FastAPI API with request batching, per-endpoint
latency metrics, and an honest benchmark suite. No GPU required.

## Architecture

```
                ┌─────────────────────────────────────────────┐
  client ──►    │  FastAPI                                    │
  POST          │   ├─ RequestIdMiddleware (X-Request-ID)     │
  /predict      │   ├─ LatencyMiddleware (per-endpoint stats) │
  /predict/     │   └─ routes                                │
  batch         │        │                                    │
  /health       │        ▼                                    │
  /metrics      │   batching: one forward pass per request   │
                │        │                                    │
                │        ▼                                    │
                │   DistilBERT (SST-2) ──► CPU (torch)        │
                │        │                                    │
                │        ▼                                    │
                │   {label, score}  +  /metrics JSON          │
                └─────────────────────────────────────────────┘
```

## Quickstart

```bash
make setup   # torch CPU wheel, then pinned requirements
make test    # pytest (TestClient, model loaded once)
make demo    # quick end-to-end smoke: predictions + /metrics (<2 min)
make bench   # full benchmark suite (see below)
```

Or with Docker (CPU):

```bash
docker build -t inferline .
docker run -p 8000:8000 inferline
curl -X POST localhost:8000/predict -H 'Content-Type: application/json' \
  -d '{"text": "This movie was fantastic!"}'
```

## API

| Endpoint | Description |
|---|---|
| `POST /predict` | Single text → `{label, score}`. Text: 1–512 chars (else 422). |
| `POST /predict/batch` | Up to 64 texts → list of `{label, score}`. True model-level batching (one forward pass), not a Python loop. |
| `GET /health` | `{status, model, ready}` — readiness flips once the model is loaded. |
| `GET /metrics` | Per-endpoint `{count, mean_ms, p50_ms, p95_ms, p99_ms}` (in-memory). |

## Benchmarks

`make bench` runs fully in-process (starlette `TestClient` against the app, no
manual server startup): warm-up, single-request latency (n=200), a batch-size
sweep (1/8/32, n=256 texts), and a concurrency sweep (1/4/16 workers, n=200 each).
Results are saved to `results/bench_results.json`.

Measured on this machine (CPU-only) on 2026-10-03 — no numbers below are invented,
they are the literal output of the benchmark run:

### Single-request latency (n=200)

| mean | p50 | p95 | p99 |
|---|---|---|---|
| 59.11 ms | 59.32 ms | 83.03 ms | 148.73 ms |

### Batch-size sweep (256 texts)

| batch size | p50 / request | throughput |
|---|---|---|
| 1 | 60.2 ms | 16.6 texts/s |
| 8 | 224.1 ms | 34.5 texts/s |
| 32 | 758.5 ms | 41.4 texts/s |

### Concurrency sweep (200 requests/worker)

| workers | p99 | throughput |
|---|---|---|
| 1 | 140.5 ms | 17.6 req/s |
| 4 | 405.2 ms | 20.7 req/s |
| 16 | 1264.5 ms | 18.9 req/s |

## Limitations (read before judging the numbers)

- **CPU-only, single model.** DistilBERT SST-2 is the only supported model;
  there is no model registry or A/B setup.
- **One worker.** The Docker/uvicorn setup runs a single worker; the
  concurrency sweep measures thread-level request handling in-process, not a
  multi-worker deployment.
- **In-memory metrics** reset on every restart — there is no persistence or
  Prometheus export.
- **No production hardening claimed:** no Kubernetes, no autoscaling, no
  auth, no rate limiting. This is a clean, well-instrumented baseline, not a
  hardened serving platform.

## Project layout

```
InferLine/
├── inferline/
│   ├── app.py         # FastAPI: routes, validation, model lifecycle
│   └── middleware.py  # request-id + per-endpoint latency stats
├── scripts/
│   ├── bench.py       # benchmark suite (latency / batch / concurrency)
│   └── demo.py        # quick end-to-end smoke
├── tests/             # pytest: API + middleware
├── Dockerfile         # CPU-only image (torch CPU wheel first)
└── results/           # bench_results.json (git-ignored)
```

## License

All rights reserved.

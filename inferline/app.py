"""InferLine FastAPI app: sentiment inference over DistilBERT on CPU.

Endpoints:
  POST /predict        single text -> {label, score}
  POST /predict/batch list of texts -> [{label, score}, ...]  (model-level batching)
  GET  /health        liveness + readiness (model loaded?)
  GET  /metrics       per-endpoint latency stats (count, mean, p50/p95/p99)
"""

from __future__ import annotations

from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from inferline import __version__
from inferline.middleware import LatencyMiddleware, LatencyStats, RequestIdMiddleware

MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"
MAX_TEXT_CHARS = 512
MAX_BATCH_SIZE = 64

stats = LatencyStats()
_model = None
_tokenizer = None
_device = torch.device("cpu")


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=MAX_TEXT_CHARS)


class BatchPredictRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1, max_length=MAX_BATCH_SIZE)


class Prediction(BaseModel):
    label: str
    score: float


@asynccontextmanager
async def lifespan(app: FastAPI):  # noqa: ANN001, ANN202
    load_model()
    yield
    global _model, _tokenizer
    _model = None
    _tokenizer = None


def load_model() -> None:
    """Load tokenizer + model onto CPU (idempotent). Used by lifespan and by
    in-process benchmarks that bypass lifespan."""
    global _model, _tokenizer
    if _model is not None:
        return
    _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    _model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    _model.to(_device)
    _model.eval()


app = FastAPI(title="InferLine", version=__version__, lifespan=lifespan)
app.add_middleware(RequestIdMiddleware)
app.add_middleware(LatencyMiddleware, stats=stats)


def _predict_texts(texts: list[str]) -> list[Prediction]:
    """Run one batched forward pass over `texts` (no Python loop over the model)."""
    inputs = _tokenizer(
        texts, padding=True, truncation=True, max_length=512, return_tensors="pt"
    )
    inputs = {k: v.to(_device) for k, v in inputs.items()}
    with torch.no_grad():
        logits = _model(**inputs).logits
    probs = torch.softmax(logits, dim=-1)
    id2label = _model.config.id2label
    out = []
    for i in range(len(texts)):
        p = probs[i]
        idx = int(torch.argmax(p))
        out.append(Prediction(label=id2label[idx], score=round(float(p[idx]), 4)))
    return out


@app.get("/health")
def health() -> dict:
    ready = _model is not None and _tokenizer is not None
    return {"status": "ok" if ready else "starting", "model": MODEL_NAME, "ready": ready}


@app.get("/metrics")
def metrics() -> dict:
    return {"endpoints": stats.snapshot()}


@app.post("/predict", response_model=Prediction)
def predict(req: PredictRequest) -> Prediction:
    if _model is None:
        raise HTTPException(status_code=503, detail="model not loaded yet")
    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=422, detail="text must not be blank")
    return _predict_texts([text])[0]


@app.post("/predict/batch", response_model=list[Prediction])
def predict_batch(req: BatchPredictRequest) -> list[Prediction]:
    if _model is None:
        raise HTTPException(status_code=503, detail="model not loaded yet")
    texts = [t.strip() for t in req.texts]
    for t in texts:
        if not t:
            raise HTTPException(status_code=422, detail="texts must not contain blank entries")
        if len(t) > MAX_TEXT_CHARS:
            raise HTTPException(
                status_code=422,
                detail=f"each text must be <= {MAX_TEXT_CHARS} characters",
            )
    return _predict_texts(texts)

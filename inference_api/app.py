from pathlib import Path
from typing import List

import numpy as np
import tensorflow as tf
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator


BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "artifacts" / "attempt2_cnn" / "cnn_model.keras"
NORM_PATH = BASE_DIR / "artifacts" / "attempt2_cnn" / "normalization.npz"
LABELS_PATH = BASE_DIR / "artifacts" / "attempt2_merged" / "merged_dataset_labels.json"

SEQUENCE_LENGTH = 16
FEATURES_PER_FRAME = 126


class PredictRequest(BaseModel):
    sequence: List[List[float]] = Field(
        ...,
        description=f"Sequence of {SEQUENCE_LENGTH} frames, each with {FEATURES_PER_FRAME} features.",
    )

    @field_validator("sequence")
    @classmethod
    def validate_sequence(cls, value: List[List[float]]):
        if not value:
            raise ValueError("sequence cannot be empty")
        if len(value) != SEQUENCE_LENGTH:
            raise ValueError(f"sequence must contain exactly {SEQUENCE_LENGTH} frames")
        for index, frame in enumerate(value):
            if len(frame) != FEATURES_PER_FRAME:
                raise ValueError(
                    f"frame {index} must contain exactly {FEATURES_PER_FRAME} features"
                )
        return value


class PredictionItem(BaseModel):
    label: str
    score: float


class PredictResponse(BaseModel):
    label: str
    score: float
    top3: List[PredictionItem]


app = FastAPI(
    title="Sign Inference API",
    version="1.0.0",
    description="Predicts a sign label from a sequence of hand-landmark features.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


MODEL = None
LABELS = None
MEAN = None
STD = None


def load_assets():
    global MODEL, LABELS, MEAN, STD
    if MODEL is not None:
        return

    if not MODEL_PATH.exists():
        raise RuntimeError(f"Model not found: {MODEL_PATH}")
    if not NORM_PATH.exists():
        raise RuntimeError(f"Normalization file not found: {NORM_PATH}")
    if not LABELS_PATH.exists():
        raise RuntimeError(f"Labels file not found: {LABELS_PATH}")

    MODEL = tf.keras.models.load_model(MODEL_PATH)
    norm = np.load(NORM_PATH)
    MEAN = norm["mean"].astype(np.float32)
    STD = norm["std"].astype(np.float32)

    import json

    with LABELS_PATH.open("r", encoding="utf-8") as f:
        LABELS = json.load(f)["labels"]


@app.on_event("startup")
def startup_event():
    load_assets()


@app.get("/health")
def health():
    load_assets()
    return {
        "status": "ok",
        "model_loaded": MODEL is not None,
        "sequence_length": SEQUENCE_LENGTH,
        "features_per_frame": FEATURES_PER_FRAME,
        "num_labels": len(LABELS),
    }


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest):
    load_assets()

    try:
        sequence = np.asarray(payload.sequence, dtype=np.float32)
        sequence = sequence.reshape(1, SEQUENCE_LENGTH, FEATURES_PER_FRAME)
        normalized = (sequence - MEAN) / STD
        probabilities = MODEL.predict(normalized, verbose=0)[0]
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inference error: {exc}") from exc

    top_ids = np.argsort(probabilities)[-3:][::-1]
    top3 = [
        PredictionItem(label=LABELS[idx], score=float(probabilities[idx]))
        for idx in top_ids
    ]

    return PredictResponse(
        label=top3[0].label,
        score=top3[0].score,
        top3=top3,
    )

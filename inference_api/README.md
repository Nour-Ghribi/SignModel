# Sign Inference API

This service exposes the trained 1D-CNN sign model as an HTTP API.

## Files used

- `artifacts/attempt2_cnn/cnn_model.keras`
- `artifacts/attempt2_cnn/normalization.npz`
- `artifacts/attempt2_merged/merged_dataset_labels.json`

## Install

```bash
pip install -r inference_api/requirements.txt
```

## Run

```bash
python -m uvicorn inference_api.app:app --host 0.0.0.0 --port 8000 --reload
```

Swagger docs:

- `http://localhost:8000/docs`

Health:

- `GET /health`

## Request format

`POST /predict`

```json
{
  "sequence": [
    [0.0, 0.1, 0.2],
    [0.0, 0.1, 0.2]
  ]
}
```

Rules:

- `sequence.length` must be exactly `16`
- each frame must contain exactly `126` features

## Response format

```json
{
  "label": "drink",
  "score": 0.94,
  "top3": [
    { "label": "drink", "score": 0.94 },
    { "label": "call", "score": 0.03 },
    { "label": "cool", "score": 0.01 }
  ]
}
```

## What your colleague needs

Your colleague keeps:

- webcam
- UI
- MediaPipe / hand landmarks extraction

Your colleague sends only:

- a `16 x 126` sequence of landmarks features

Your API returns:

- predicted sign
- score
- top 3 labels

## Frontend usage

See `inference_api/client_example.js`.

## Send to colleague

Share this folder:

- `inference_api/`

And make sure they also receive these model artifacts:

- `artifacts/attempt2_cnn/cnn_model.keras`
- `artifacts/attempt2_cnn/normalization.npz`
- `artifacts/attempt2_merged/merged_dataset_labels.json`

If they run the API from a different project, they should either:

1. keep the same relative folder structure, or
2. edit the paths at the top of `inference_api/app.py`.

import argparse
import csv
import json
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python import vision


NUM_HANDS = 2
LANDMARKS_PER_HAND = 21
COORDS_PER_LANDMARK = 3
FEATURES_PER_FRAME = NUM_HANDS * LANDMARKS_PER_HAND * COORDS_PER_LANDMARK


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract hand-landmark sequences in small batches."
    )
    parser.add_argument("--csv", required=True, help="Metadata CSV with Filepath,Label")
    parser.add_argument("--data-root", required=True, help="Root folder containing the videos")
    parser.add_argument("--out-dir", required=True, help="Output directory")
    parser.add_argument(
        "--model",
        default="C:\\Users\\nourg\\Desktop\\challenges\\public\\wasm\\hand_landmarker.task",
        help="Path to hand_landmarker.task",
    )
    parser.add_argument("--frames", type=int, default=32, help="Frames sampled per video")
    parser.add_argument("--batch-size", type=int, default=50, help="Videos per batch")
    parser.add_argument("--batch-index", type=int, default=0, help="0-based batch index")
    return parser.parse_args()


def resolve_video_path(data_root: Path, rel_path: str) -> Path:
    rel = Path(rel_path)
    parts = rel.parts
    if len(parts) >= 2 and parts[0].lower() == "data":
        return data_root / Path(*parts[1:])
    return data_root / rel


def sample_frame_indices(frame_count: int, frames_to_sample: int):
    if frame_count <= 0:
        return []
    if frame_count <= frames_to_sample:
        return list(range(frame_count))
    return np.linspace(0, frame_count - 1, frames_to_sample).round().astype(int).tolist()


def normalize_hand(landmarks):
    wrist = landmarks[0]
    middle_mcp = landmarks[9]
    scale = np.sqrt(
        (middle_mcp.x - wrist.x) ** 2
        + (middle_mcp.y - wrist.y) ** 2
        + (middle_mcp.z - wrist.z) ** 2
    )
    scale = max(scale, 1e-6)

    values = []
    for lm in landmarks:
        values.extend(
            [
                (lm.x - wrist.x) / scale,
                (lm.y - wrist.y) / scale,
                (lm.z - wrist.z) / scale,
            ]
        )
    return values


def extract_sequence(video_path: Path, detector, frames_to_sample: int):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None, 0, 0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = sample_frame_indices(total_frames, frames_to_sample)
    sequence = []
    detections = 0

    for frame_idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
        ok, frame = cap.read()
        if not ok:
            sequence.append([0.0] * FEATURES_PER_FRAME)
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = detector.detect(mp_image)
        frame_features = []

        if result.hand_landmarks:
            ordered_hands = sorted(
                result.hand_landmarks,
                key=lambda hand: hand[0].x,
            )[:NUM_HANDS]
            detections += 1
            for hand_landmarks in ordered_hands:
                frame_features.extend(normalize_hand(hand_landmarks))

        while len(frame_features) < FEATURES_PER_FRAME:
            frame_features.extend([0.0] * (LANDMARKS_PER_HAND * COORDS_PER_LANDMARK))

        sequence.append(frame_features[:FEATURES_PER_FRAME])

    cap.release()

    if len(sequence) < frames_to_sample:
        missing = frames_to_sample - len(sequence)
        sequence.extend([[0.0] * FEATURES_PER_FRAME for _ in range(missing)])

    return np.asarray(sequence, dtype=np.float32), total_frames, detections


def main():
    args = parse_args()
    csv_path = Path(args.csv)
    data_root = Path(args.data_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    labels = sorted({row["Label"] for row in rows})
    label_to_id = {label: idx for idx, label in enumerate(labels)}

    start = args.batch_index * args.batch_size
    end = min(start + args.batch_size, len(rows))
    batch_rows = rows[start:end]
    if not batch_rows:
        raise SystemExit(f"No rows for batch {args.batch_index}.")

    detector_options = vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(Path(args.model))),
        num_hands=NUM_HANDS,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    X = []
    y = []
    meta = []

    with vision.HandLandmarker.create_from_options(detector_options) as detector:
        for offset, row in enumerate(batch_rows, start=1):
            rel_path = row["Filepath"]
            label = row["Label"]
            video_path = resolve_video_path(data_root, rel_path)

            if not video_path.exists():
                meta.append(
                    {
                        "filepath": rel_path,
                        "resolved_path": str(video_path),
                        "label": label,
                        "status": "missing",
                    }
                )
                continue

            sequence, total_frames, detections = extract_sequence(video_path, detector, args.frames)
            if sequence is None:
                meta.append(
                    {
                        "filepath": rel_path,
                        "resolved_path": str(video_path),
                        "label": label,
                        "status": "unreadable",
                    }
                )
                continue

            X.append(sequence)
            y.append(label_to_id[label])
            meta.append(
                {
                    "filepath": rel_path,
                    "resolved_path": str(video_path),
                    "label": label,
                    "label_id": label_to_id[label],
                    "status": "ok",
                    "total_frames": int(total_frames),
                    "detected_frames": int(detections),
                }
            )
            print(f"Batch {args.batch_index}: {offset}/{len(batch_rows)}")

    batch_prefix = f"batch_{args.batch_index:03d}"

    if X:
        np.savez_compressed(
            out_dir / f"{batch_prefix}_landmarks.npz",
            X=np.stack(X),
            y=np.asarray(y, dtype=np.int64),
        )

    with (out_dir / f"{batch_prefix}_meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=True, indent=2)

    labels_path = out_dir / "labels.json"
    if not labels_path.exists():
        with labels_path.open("w", encoding="utf-8") as f:
            json.dump({"labels": labels, "label_to_id": label_to_id}, f, ensure_ascii=True, indent=2)

    ok_count = sum(1 for item in meta if item["status"] == "ok")
    missing_count = sum(1 for item in meta if item["status"] == "missing")
    unreadable_count = sum(1 for item in meta if item["status"] == "unreadable")

    print(f"Batch rows: {len(batch_rows)}")
    print(f"Usable videos: {ok_count}")
    print(f"Missing videos: {missing_count}")
    print(f"Unreadable videos: {unreadable_count}")


if __name__ == "__main__":
    main()

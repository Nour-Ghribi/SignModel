import argparse
import json
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python import vision


NUM_HANDS = 2
LANDMARKS_PER_HAND = 21
COORDS_PER_LANDMARK = 3
FEATURES_PER_FRAME = NUM_HANDS * LANDMARKS_PER_HAND * COORDS_PER_LANDMARK


def parse_args():
    parser = argparse.ArgumentParser(description="Predict a sign label from one video.")
    parser.add_argument("--video", required=True, help="Path to the video file")
    parser.add_argument(
        "--model",
        default="C:\\Users\\nourg\\Desktop\\challenges\\artifacts\\attempt2_cnn\\cnn_model.keras",
        help="Path to the trained CNN model",
    )
    parser.add_argument(
        "--labels",
        default="C:\\Users\\nourg\\Desktop\\challenges\\artifacts\\attempt2_merged\\merged_dataset_labels.json",
        help="Path to labels json",
    )
    parser.add_argument(
        "--norm",
        default="C:\\Users\\nourg\\Desktop\\challenges\\artifacts\\attempt2_cnn\\normalization.npz",
        help="Path to normalization stats",
    )
    parser.add_argument(
        "--hand-model",
        default="C:\\Users\\nourg\\Desktop\\challenges\\public\\wasm\\hand_landmarker.task",
        help="Path to hand_landmarker.task",
    )
    parser.add_argument("--frames", type=int, default=16, help="Frames sampled per video")
    return parser.parse_args()


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
        raise RuntimeError(f"Cannot open video: {video_path}")

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
            ordered_hands = sorted(result.hand_landmarks, key=lambda hand: hand[0].x)[:NUM_HANDS]
            detections += 1
            for hand_landmarks in ordered_hands:
                frame_features.extend(normalize_hand(hand_landmarks))

        while len(frame_features) < FEATURES_PER_FRAME:
            frame_features.extend([0.0] * (LANDMARKS_PER_HAND * COORDS_PER_LANDMARK))

        sequence.append(frame_features[:FEATURES_PER_FRAME])

    cap.release()

    if len(sequence) < frames_to_sample:
        sequence.extend([[0.0] * FEATURES_PER_FRAME for _ in range(frames_to_sample - len(sequence))])

    return np.asarray(sequence, dtype=np.float32), detections


def main():
    args = parse_args()
    video_path = Path(args.video)
    if not video_path.exists():
        raise SystemExit(f"Video not found: {video_path}")

    with open(args.labels, "r", encoding="utf-8") as f:
        labels = json.load(f)["labels"]

    norm = np.load(args.norm)
    mean = norm["mean"]
    std = norm["std"]

    detector_options = vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(Path(args.hand_model))),
        num_hands=NUM_HANDS,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    with vision.HandLandmarker.create_from_options(detector_options) as detector:
        sequence, detections = extract_sequence(video_path, detector, args.frames)

    model = tf.keras.models.load_model(args.model)
    x = ((sequence[None, ...] - mean) / std).astype(np.float32)
    probs = model.predict(x, verbose=0)[0]
    top_ids = probs.argsort()[-3:][::-1]

    print(f"Video: {video_path}")
    print(f"Detected frames: {detections}/{args.frames}")
    print("Top predictions:")
    for idx in top_ids:
        print(f"- {labels[idx]}: {probs[idx]:.4f}")


if __name__ == "__main__":
    main()

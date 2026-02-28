import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split


def parse_args():
    parser = argparse.ArgumentParser(description="Train a simple baseline model on merged landmark data.")
    parser.add_argument("--dataset", required=True, help="Merged .npz dataset with X and y")
    parser.add_argument("--labels", required=True, help="labels.json file")
    parser.add_argument("--out-dir", required=True, help="Directory to write model artifacts")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test split ratio")
    parser.add_argument("--trees", type=int, default=300, help="RandomForest n_estimators")
    return parser.parse_args()


def summarize_sequence_features(X):
    mean = X.mean(axis=1)
    std = X.std(axis=1)
    maxv = X.max(axis=1)
    minv = X.min(axis=1)
    return np.concatenate([mean, std, maxv, minv], axis=1)


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = np.load(args.dataset)
    X = data["X"]
    y = data["y"]

    with open(args.labels, "r", encoding="utf-8") as f:
        labels_data = json.load(f)
    labels = labels_data["labels"]

    features = summarize_sequence_features(X)
    X_train, X_test, y_train, y_test = train_test_split(
        features,
        y,
        test_size=args.test_size,
        random_state=42,
        stratify=y,
    )

    clf = RandomForestClassifier(
        n_estimators=args.trees,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(
        y_test,
        y_pred,
        labels=list(range(len(labels))),
        target_names=labels,
        zero_division=0,
        output_dict=True,
    )

    joblib.dump(clf, out_dir / "random_forest_baseline.joblib")
    np.savez_compressed(
        out_dir / "baseline_features_test_snapshot.npz",
        X_test=X_test,
        y_test=y_test,
        y_pred=y_pred,
    )

    metrics = {
        "accuracy": acc,
        "num_samples": int(len(y)),
        "num_classes": int(len(labels)),
        "feature_shape": list(features.shape),
        "train_size": int(len(y_train)),
        "test_size": int(len(y_test)),
        "report": report,
    }

    with (out_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=True, indent=2)

    print(f"Accuracy: {acc:.4f}")
    print(f"Samples: {len(y)}")
    print(f"Classes: {len(labels)}")
    print(f"Feature shape: {features.shape}")
    print(f"Saved model: {out_dir / 'random_forest_baseline.joblib'}")
    print(f"Saved metrics: {out_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()

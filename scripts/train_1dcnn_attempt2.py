import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split


def parse_args():
    parser = argparse.ArgumentParser(description="Train a lightweight 1D-CNN on landmark sequences.")
    parser.add_argument("--dataset", required=True, help="Merged .npz dataset with X and y")
    parser.add_argument("--labels", required=True, help="labels json")
    parser.add_argument("--out-dir", required=True, help="Output directory")
    parser.add_argument("--epochs", type=int, default=30, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test split ratio")
    parser.add_argument("--val-size", type=float, default=0.2, help="Validation split ratio inside train")
    return parser.parse_args()


def build_model(input_shape, num_classes):
    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Conv1D(64, 3, padding="same", activation="relu")(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling1D(2)(x)
    x = tf.keras.layers.Dropout(0.2)(x)

    x = tf.keras.layers.Conv1D(128, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling1D(2)(x)
    x = tf.keras.layers.Dropout(0.25)(x)

    x = tf.keras.layers.Conv1D(128, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = np.load(args.dataset)
    X = data["X"].astype(np.float32)
    y = data["y"].astype(np.int64)

    with open(args.labels, "r", encoding="utf-8") as f:
        labels_data = json.load(f)
    labels = labels_data["labels"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=42,
        stratify=y,
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train,
        y_train,
        test_size=args.val_size,
        random_state=42,
        stratify=y_train,
    )

    mean = X_train.mean(axis=(0, 1), keepdims=True)
    std = X_train.std(axis=(0, 1), keepdims=True) + 1e-6
    X_train = (X_train - mean) / std
    X_val = (X_val - mean) / std
    X_test = (X_test - mean) / std

    model = build_model((X.shape[1], X.shape[2]), len(labels))

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=6,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-5,
        ),
    ]

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        verbose=2,
        callbacks=callbacks,
    )

    y_prob = model.predict(X_test, verbose=0)
    y_pred = y_prob.argmax(axis=1)
    test_acc = accuracy_score(y_test, y_pred)
    report = classification_report(
        y_test,
        y_pred,
        labels=list(range(len(labels))),
        target_names=labels,
        zero_division=0,
        output_dict=True,
    )

    model.save(out_dir / "cnn_model.keras")
    np.savez_compressed(
        out_dir / "normalization.npz",
        mean=mean,
        std=std,
    )
    with (out_dir / "history.json").open("w", encoding="utf-8") as f:
        json.dump(history.history, f, ensure_ascii=True, indent=2)
    with (out_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "test_accuracy": test_acc,
                "num_samples": int(len(y)),
                "num_classes": int(len(labels)),
                "train_size": int(len(y_train)),
                "val_size": int(len(y_val)),
                "test_size": int(len(y_test)),
                "input_shape": [int(X.shape[1]), int(X.shape[2])],
                "report": report,
            },
            f,
            ensure_ascii=True,
            indent=2,
        )

    print(f"Test accuracy: {test_acc:.4f}")
    print(f"Saved model: {out_dir / 'cnn_model.keras'}")
    print(f"Saved metrics: {out_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()

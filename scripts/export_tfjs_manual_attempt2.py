import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf


def parse_args():
    parser = argparse.ArgumentParser(description="Export a Keras model to a minimal TFJS layers format.")
    parser.add_argument("--keras-model", required=True, help="Path to .keras model")
    parser.add_argument("--labels", required=True, help="Path to labels json")
    parser.add_argument("--norm", required=True, help="Path to normalization .npz")
    parser.add_argument("--out-dir", required=True, help="Output directory under public/")
    return parser.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = tf.keras.models.load_model(args.keras_model)
    model_topology = json.loads(model.to_json())

    weights_specs = []
    weights_bytes = bytearray()
    for variable in model.weights:
      array = variable.numpy().astype(np.float32)
      name = getattr(variable, "path", variable.name).replace(":0", "")
      weights_specs.append(
          {
              "name": name,
              "shape": list(array.shape),
              "dtype": "float32",
          }
      )
      weights_bytes.extend(array.tobytes(order="C"))

    shard_name = "group1-shard1of1.bin"
    with (out_dir / shard_name).open("wb") as f:
        f.write(weights_bytes)

    model_json = {
        "format": "layers-model",
        "generatedBy": f"Keras v{tf.keras.__version__}",
        "convertedBy": "manual-export-attempt2",
        "modelTopology": model_topology,
        "weightsManifest": [
            {
                "paths": [shard_name],
                "weights": weights_specs,
            }
        ],
    }

    with (out_dir / "model.json").open("w", encoding="utf-8") as f:
        json.dump(model_json, f, ensure_ascii=True)

    labels = json.load(open(args.labels, "r", encoding="utf-8"))
    with (out_dir / "labels.json").open("w", encoding="utf-8") as f:
        json.dump(labels, f, ensure_ascii=True, indent=2)

    norm = np.load(args.norm)
    with (out_dir / "normalization.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "mean": norm["mean"].tolist(),
                "std": norm["std"].tolist(),
            },
            f,
            ensure_ascii=True,
        )

    print(f"Saved TFJS artifacts to: {out_dir}")


if __name__ == "__main__":
    main()

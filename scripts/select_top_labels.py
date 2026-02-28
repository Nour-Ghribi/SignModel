import argparse
import csv
from collections import Counter
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Filter a metadata CSV to keep the top-N most frequent labels."
    )
    parser.add_argument("input_csv", help="Path to the source metadata.csv")
    parser.add_argument("output_csv", help="Path to the filtered CSV to create")
    parser.add_argument(
        "--top-n",
        type=int,
        default=20,
        help="Number of most frequent labels to keep (default: 20)",
    )
    parser.add_argument(
        "--min-count",
        type=int,
        default=0,
        help="Optional minimum number of samples per label",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    input_path = Path(args.input_csv)
    output_path = Path(args.output_csv)

    with input_path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        raise SystemExit("Input CSV is empty.")

    if "Filepath" not in rows[0] or "Label" not in rows[0]:
        raise SystemExit("CSV must contain Filepath and Label columns.")

    counts = Counter(row["Label"] for row in rows)
    selected = [
        label for label, count in counts.most_common() if count >= args.min_count
    ][: args.top_n]
    selected_set = set(selected)

    filtered = [row for row in rows if row["Label"] in selected_set]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["Filepath", "Label"])
        writer.writeheader()
        writer.writerows(filtered)

    print(f"Input rows: {len(rows)}")
    print(f"Unique labels: {len(counts)}")
    print(f"Selected labels: {len(selected)}")
    print(f"Output rows: {len(filtered)}")
    print("Top labels kept:")
    for label in selected:
        print(f"- {label}: {counts[label]}")


if __name__ == "__main__":
    main()

import argparse
import csv
import glob
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Create a clean CSV with only usable videos.")
    parser.add_argument("--source-csv", required=True, help="Original filtered metadata csv")
    parser.add_argument("--meta-dir", required=True, help="Directory with batch_*_meta.json")
    parser.add_argument("--out-csv", required=True, help="Output clean CSV")
    return parser.parse_args()


def main():
    args = parse_args()
    source_rows = list(csv.DictReader(open(args.source_csv, "r", encoding="utf-8-sig", newline="")))

    ok_paths = set()
    for meta_file in glob.glob(str(Path(args.meta_dir) / "batch_*_meta.json")):
        data = json.load(open(meta_file, "r", encoding="utf-8"))
        ok_paths.update(item["filepath"] for item in data if item["status"] == "ok")

    clean_rows = [row for row in source_rows if row["Filepath"] in ok_paths]

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["Filepath", "Label"])
        writer.writeheader()
        writer.writerows(clean_rows)

    print(f"Source rows: {len(source_rows)}")
    print(f"Clean rows: {len(clean_rows)}")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()

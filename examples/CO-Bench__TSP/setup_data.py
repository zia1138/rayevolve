#!/usr/bin/env python3
"""Download CO-Bench TSP data from HuggingFace.

Run this once before using the evaluator:
    uv run --with huggingface_hub python setup_data.py
"""

from huggingface_hub import HfFileSystem
from pathlib import Path

HF_BASE = "datasets/CO-Bench/CO-Bench"
PROBLEM_NAME = "Travelling salesman problem"


def main():
    fs = HfFileSystem()
    data_dir = Path(__file__).resolve().parent / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    src_prefix = f"{HF_BASE}/{PROBLEM_NAME}"
    files = fs.find(src_prefix)

    downloaded = 0
    for path in files:
        fname = path.split("/")[-1]
        if not fname.endswith(".txt"):
            continue

        out_path = data_dir / fname
        with fs.open(path, "rb") as src, open(out_path, "wb") as dst:
            dst.write(src.read())
        downloaded += 1
        print(f"  {out_path}")

    print(f"\nDownloaded {downloaded} files to {data_dir}")


if __name__ == "__main__":
    main()

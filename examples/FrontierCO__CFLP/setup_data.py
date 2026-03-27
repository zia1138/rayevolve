#!/usr/bin/env python3
"""Download CFLP data from FrontierCO on HuggingFace."""

from huggingface_hub import HfFileSystem
from pathlib import Path

HF_BASE = "datasets/CO-Bench/FrontierCO/CFLP"
SUBDIRS = ["valid_instances", "easy_test_instances", "hard_test_instances"]
EXTENSIONS = {".txt", ".plc"}


def main():
    fs = HfFileSystem()
    data_dir = Path(__file__).resolve().parent / "data"
    data_dir.mkdir(exist_ok=True)

    for subdir in SUBDIRS:
        src_prefix = f"{HF_BASE}/{subdir}"
        files = fs.find(src_prefix)
        downloaded = 0
        for path in files:
            fname = path.split("/")[-1]
            if not any(fname.endswith(ext) for ext in EXTENSIONS):
                continue
            out_path = data_dir / fname
            with fs.open(path, "rb") as src, open(out_path, "wb") as dst:
                dst.write(src.read())
            downloaded += 1
            print(f"  {out_path}")
        print(f"[{subdir}] Downloaded {downloaded} files")

    print("\nDone. Data is ready for evaluation.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Download CO-Bench data for bin_packing, flow_shop, and tsp examples."""

from huggingface_hub import HfFileSystem
from pathlib import Path

# Map CO-Bench problem names to local directories and file patterns
PROBLEMS = {
    "Bin packing - one-dimensional": {
        "local_dir": "bin_packing/data",
        "extensions": [".txt"],
    },
    "Flow shop scheduling": {
        "local_dir": "flow_shop/data",
        "extensions": [".txt"],
    },
    "Travelling salesman problem": {
        "local_dir": "tsp/data",
        "extensions": [".txt"],
    },
}

HF_BASE = "datasets/CO-Bench/CO-Bench"


def main():
    fs = HfFileSystem()
    base_dir = Path(__file__).resolve().parent

    for problem_name, cfg in PROBLEMS.items():
        local_dir = base_dir / cfg["local_dir"]
        local_dir.mkdir(parents=True, exist_ok=True)

        src_prefix = f"{HF_BASE}/{problem_name}"
        files = fs.find(src_prefix)

        downloaded = 0
        for path in files:
            fname = path.split("/")[-1]

            # Skip config.py and non-matching extensions
            if fname == "config.py":
                continue
            if not any(fname.endswith(ext) for ext in cfg["extensions"]):
                continue

            out_path = local_dir / fname
            with fs.open(path, "rb") as src, open(out_path, "wb") as dst:
                dst.write(src.read())
            downloaded += 1
            print(f"  {out_path}")

        print(f"[{problem_name}] Downloaded {downloaded} files to {local_dir}")

    print("\nDone. Data is ready for evaluation.")


if __name__ == "__main__":
    main()

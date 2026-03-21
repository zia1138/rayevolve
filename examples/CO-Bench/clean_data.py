#!/usr/bin/env python3

from huggingface_hub import HfFileSystem
from pathlib import Path

if __name__ == "__main__":
    fs = HfFileSystem()

    src_dir = "datasets/CO-Bench/CO-Bench"
    dst_dir = Path("./")

    scanned = 0
    skipped_py = 0
    deleted = 0

    for path in fs.find(src_dir):
        scanned += 1

        # keep only Python files
        if path.endswith(".py") or ".git" in path:
            skipped_py += 1
            continue

        # compute relative path
        rel_path = path[len(src_dir):].lstrip("/")
        out_path = dst_dir / rel_path

        if out_path.exists():
            out_path.unlink()
            deleted += 1
            print(f"Removed {out_path}")

    print(
        "Done."
        f" Scanned: {scanned}, kept .py: {skipped_py}, removed: {deleted}."
    )


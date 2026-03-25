#!/usr/bin/env python3

from huggingface_hub import HfFileSystem
from pathlib import Path


if __name__ == "__main__":
    fs = HfFileSystem()

    src_dir = "datasets/CO-Bench/CO-Bench"
    dst_dir = Path("./")

    scanned = 0
    skipped_py = 0
    downloaded = 0

    for path in fs.find(src_dir):
        scanned += 1

        # keep only Python files
        if path.endswith(".py") or ".git" in path or "README.md" in path:
            skipped_py += 1
            continue

        # compute relative path
        rel_path = path[len(src_dir):].lstrip("/")
        out_path = dst_dir / rel_path

        # only download if destination directory already exists
        if not out_path.parent.exists():
            continue

        # copy file
        with fs.open(path, "rb") as fsrc, open(out_path, "wb") as fdst:
            fdst.write(fsrc.read())
        downloaded += 1
        print(f"Downloaded {out_path}")

    print(
        "Done."
        f" Scanned: {scanned}, kept .py: {skipped_py}, downloaded: {downloaded}."
    )


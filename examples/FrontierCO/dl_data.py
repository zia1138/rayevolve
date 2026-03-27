#!/usr/bin/env python3

from huggingface_hub import HfFileSystem
from pathlib import Path
import typer


def main(all_files: bool = typer.Option(False, "--all")) -> None:
    fs = HfFileSystem()

    src_dir = "datasets/CO-Bench/FrontierCO"
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

        if not all_files and "valid_instances" not in path:
            continue

        # compute relative path
        rel_path = path[len(src_dir):].lstrip("/")
        out_path = dst_dir / rel_path

        # only download if parent exists; create destination dir if needed
        parent_dir = out_path.parent
        if not parent_dir.parent.exists():
            continue
        parent_dir.mkdir(parents=True, exist_ok=True)

        # copy file
        with fs.open(path, "rb") as fsrc, open(out_path, "wb") as fdst:
            fdst.write(fsrc.read())
        downloaded += 1
        print(f"Downloaded {out_path}")

    print(
        "Done."
        f" Scanned: {scanned}, kept .py: {skipped_py}, downloaded: {downloaded}."
    )


if __name__ == "__main__":
    typer.run(main)


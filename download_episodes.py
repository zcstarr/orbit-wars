#!/usr/bin/env python3
"""Download and cache orbit-wars episode datasets listed in manifest.csv."""

from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import date
from pathlib import Path


@dataclass(frozen=True)
class DailyDataset:
    date: date
    slug: str
    url: str
    episode_count: int
    total_bytes: int


def parse_date(value: str) -> date:
    return date.fromisoformat(value)


def load_manifest(path: Path) -> list[DailyDataset]:
    rows: list[DailyDataset] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(
                DailyDataset(
                    date=parse_date(row["date"]),
                    slug=row["daily_dataset_slug"],
                    url=row["daily_dataset_url"],
                    episode_count=int(row["episode_count"]),
                    total_bytes=int(row["total_bytes"]),
                )
            )
    return rows


def filter_datasets(
    datasets: list[DailyDataset],
    dates: set[date] | None,
    start: date | None,
    end: date | None,
) -> list[DailyDataset]:
    selected = datasets
    if dates is not None:
        selected = [row for row in selected if row.date in dates]
    if start is not None:
        selected = [row for row in selected if row.date >= start]
    if end is not None:
        selected = [row for row in selected if row.date <= end]
    return selected


def episode_json_files(directory: Path) -> list[Path]:
    return sorted(path for path in directory.glob("*.json") if path.is_file())


def cache_status(
    dest_dir: Path, expected_episodes: int, expected_bytes: int
) -> tuple[bool, str]:
    if not dest_dir.is_dir():
        return False, "missing directory"

    json_files = episode_json_files(dest_dir)
    if len(json_files) < expected_episodes:
        return False, f"found {len(json_files)}/{expected_episodes} episode json files"

    actual_bytes = sum(path.stat().st_size for path in json_files)
    # Dataset archives can differ slightly from manifest totals; allow 5% slack.
    if expected_bytes > 0 and actual_bytes < expected_bytes * 0.95:
        return (
            False,
            f"size mismatch: {actual_bytes} bytes cached, expected ~{expected_bytes}",
        )

    return True, f"{len(json_files)} episodes, {actual_bytes:,} bytes"


def run_kaggle_download(slug: str, download_dir: Path) -> None:
    command = [
        "kaggle",
        "datasets",
        "download",
        "-d",
        f"kaggle/{slug}",
        "-p",
        str(download_dir),
        "--unzip",
    ]
    subprocess.run(command, check=True)


def download_dataset(
    dataset: DailyDataset,
    output_dir: Path,
    *,
    force: bool,
    dry_run: bool,
) -> str:
    dest_dir = output_dir / dataset.date.isoformat()
    cached, detail = cache_status(dest_dir, dataset.episode_count, dataset.total_bytes)
    if cached and not force:
        return f"skip  {dataset.date}  ({detail})"

    if dry_run:
        action = "refresh" if dest_dir.exists() else "download"
        return (
            f"{action} {dataset.date}  "
            f"({dataset.episode_count} episodes, {dataset.total_bytes:,} bytes)"
        )

    if force and dest_dir.exists():
        shutil.rmtree(dest_dir)

    dest_dir.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix=f"orbit-wars-{dataset.date}-") as tmp:
        tmp_dir = Path(tmp)
        run_kaggle_download(dataset.slug, tmp_dir)

        json_files = episode_json_files(tmp_dir)
        if len(json_files) < dataset.episode_count:
            raise RuntimeError(
                f"{dataset.date}: expected {dataset.episode_count} episode json files, "
                f"got {len(json_files)} after download"
            )

        shutil.move(str(tmp_dir), str(dest_dir))

    cached, detail = cache_status(dest_dir, dataset.episode_count, dataset.total_bytes)
    if not cached:
        raise RuntimeError(f"{dataset.date}: cache verification failed ({detail})")

    return f"saved {dataset.date}  ({detail})"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Download orbit-wars daily episode datasets from manifest.csv into "
            "episode_data/<YYYY-MM-DD>/."
        )
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("manifest.csv"),
        help="Path to the daily dataset manifest (default: manifest.csv)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("episode_data"),
        help="Cache root directory (default: episode_data)",
    )
    parser.add_argument(
        "--date",
        action="append",
        dest="dates",
        metavar="YYYY-MM-DD",
        help="Download only this date; repeatable",
    )
    parser.add_argument(
        "--from",
        dest="start_date",
        metavar="YYYY-MM-DD",
        help="Earliest date to download (inclusive)",
    )
    parser.add_argument(
        "--to",
        dest="end_date",
        metavar="YYYY-MM-DD",
        help="Latest date to download (inclusive)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Redownload even when a valid cache already exists",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned downloads without fetching data",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.manifest.is_file():
        parser.error(f"manifest not found: {args.manifest}")

    datasets = load_manifest(args.manifest)
    date_filter = {parse_date(value) for value in args.dates} if args.dates else None
    start = parse_date(args.start_date) if args.start_date else None
    end = parse_date(args.end_date) if args.end_date else None
    selected = filter_datasets(datasets, date_filter, start, end)

    if not selected:
        print("No datasets matched the requested filters.", file=sys.stderr)
        return 1

    failures = 0
    for dataset in selected:
        try:
            message = download_dataset(
                dataset,
                args.output,
                force=args.force,
                dry_run=args.dry_run,
            )
            print(message)
        except (OSError, RuntimeError, subprocess.CalledProcessError) as exc:
            failures += 1
            print(f"error {dataset.date}: {exc}", file=sys.stderr)

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

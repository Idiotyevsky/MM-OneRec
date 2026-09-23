"""Download and validate the first usable product image for each item."""

from __future__ import annotations

import argparse
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm

from multimodal.utils import cache_name, extract_image_urls, iter_metadata

LOGGER = logging.getLogger("mm_onerec.images")


def _fetch(url: str, timeout: float) -> bytes:
    if url.startswith("file://"):
        return Path(urlparse(url).path).read_bytes()
    response = requests.get(url, timeout=timeout, headers={"User-Agent": "MM-OneRec/1.0"})
    response.raise_for_status()
    return response.content


def _validated_jpeg(content: bytes) -> bytes:
    with Image.open(BytesIO(content)) as image:
        image.verify()
    with Image.open(BytesIO(content)) as image:
        converted = image.convert("RGB")
        output = BytesIO()
        converted.save(output, format="JPEG", quality=92)
        return output.getvalue()


def download_images(
    metadata_path: Path,
    cache_dir: Path,
    manifest_path: Path,
    timeout: float = 10.0,
    retries: int = 2,
    max_items: int | None = None,
    workers: int = 8,
) -> dict[str, int]:
    """Download images and write an item-aligned JSONL manifest."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    stats = {"total": 0, "downloaded": 0, "cached": 0, "missing_url": 0, "failed": 0}
    records = list(iter_metadata(metadata_path))
    if max_items is not None:
        records = records[:max_items]

    def process(entry: tuple[str, dict[str, Any]]) -> dict[str, Any]:
        item_id, record = entry
        urls = extract_image_urls(record)
        row: dict[str, Any] = {"item_id": item_id, "image_path": None, "url": None, "status": "missing_url"}
        if not urls:
            return row
        for url in urls:
            destination = cache_dir / cache_name(item_id, url)
            if destination.is_file() and destination.stat().st_size > 0:
                row.update(image_path=str(destination), url=url, status="cached")
                break
            for attempt in range(retries + 1):
                try:
                    content = _validated_jpeg(_fetch(url, timeout))
                    destination.write_bytes(content)
                    row.update(image_path=str(destination), url=url, status="downloaded")
                    break
                except (OSError, requests.RequestException, UnidentifiedImageError, ValueError) as exc:
                    row["error"] = str(exc)[:300]
                    if attempt < retries:
                        time.sleep(0.25 * (2**attempt))
            if row["image_path"]:
                break
        if not row["image_path"]:
            row["status"] = "failed"
        return row

    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        rows = list(tqdm(executor.map(process, records), total=len(records), desc="images"))
    stats["total"] = len(rows)
    for row in rows:
        stats[row["status"]] += 1
    with manifest_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    LOGGER.info("Image download statistics: %s", stats)
    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--max-items", type=int)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42, help="Reserved for reproducible extensions.")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()
    stats = download_images(args.metadata, args.cache_dir, args.manifest, args.timeout, args.retries, args.max_items, args.workers)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()

"""Shared, dependency-light helpers for multimodal preprocessing."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterator


IMAGE_FIELDS = ("image", "images", "image_url", "image_urls", "imUrl", "large", "hi_res")


def iter_metadata(path: Path) -> Iterator[tuple[str, dict[str, Any]]]:
    """Yield ``(item_id, metadata)`` from JSON dict, JSONL, or gzipped JSONL."""
    import gzip

    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        first = handle.read(1)
        handle.seek(0)
        if first == "{":
            try:
                data = json.load(handle)
            except json.JSONDecodeError:
                handle.seek(0)
            else:
                if isinstance(data, dict):
                    for key, value in data.items():
                        if isinstance(value, dict):
                            yield str(key), value
                    return
        for line_number, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc
            item_id = record.get("parent_asin") or record.get("asin") or record.get("item_id")
            if item_id is not None:
                yield str(item_id), record


def extract_image_urls(record: dict[str, Any]) -> list[str]:
    """Extract candidate URLs from common Amazon18/Amazon23 metadata layouts."""
    urls: list[str] = []

    def visit(value: Any) -> None:
        if isinstance(value, str) and value.startswith(("http://", "https://", "file://")):
            urls.append(value)
        elif isinstance(value, list):
            for entry in value:
                visit(entry)
        elif isinstance(value, dict):
            for key in ("large", "hi_res", "thumb", "url"):
                if key in value:
                    visit(value[key])

    for field in IMAGE_FIELDS:
        if field in record:
            visit(record[field])
    return list(dict.fromkeys(urls))


def cache_name(item_id: str, url: str, suffix: str = ".jpg") -> str:
    """Return a stable, filesystem-safe cache filename."""
    digest = hashlib.sha1(url.encode("utf-8")).hexdigest()[:12]
    safe_id = "".join(char if char.isalnum() or char in "-_" else "_" for char in item_id)
    return f"{safe_id}-{digest}{suffix}"

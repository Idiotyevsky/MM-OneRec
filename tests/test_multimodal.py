import json
from pathlib import Path

import numpy as np
from PIL import Image

from multimodal.image_downloader import download_images
from multimodal.image_encoder import encode_images
from multimodal.multimodal_encoder import fuse_embeddings
from multimodal.text_encoder import item_text
from scripts.mm.evaluate_metrics import evaluate
from scripts.mm.sid_stats import collision_stats


def test_offline_multimodal_pipeline(tmp_path: Path) -> None:
    source = tmp_path / "source"; source.mkdir()
    metadata = {}
    for index in range(20):
        image = source / f"{index}.png"
        Image.new("RGB", (16, 16), (index * 10, 40, 200)).save(image)
        metadata[str(index)] = {"title": f"item {index}", "image": [image.as_uri()]}
    metadata["19"]["image"] = []
    metadata_path = tmp_path / "items.json"; metadata_path.write_text(json.dumps(metadata))
    manifest = tmp_path / "manifest.jsonl"
    stats = download_images(metadata_path, tmp_path / "cache", manifest, retries=0)
    assert stats == {"total": 20, "downloaded": 19, "cached": 0, "missing_url": 1, "failed": 0}
    cached = download_images(metadata_path, tmp_path / "cache", manifest, retries=0)
    assert cached["cached"] == 19
    image_emb, mask = encode_images(manifest, tmp_path / "image.npy", tmp_path / "mask.npy", backend="histogram", batch_size=5)
    text_emb = np.random.default_rng(42).normal(size=(20, 32)).astype(np.float32)
    fused = fuse_embeddings(text_emb, image_emb, alpha=0.7, image_mask=mask, target_dim=16)
    assert image_emb.shape == (20, 48); assert fused.shape == (20, 16); assert mask.sum() == 19
    np.testing.assert_allclose(np.linalg.norm(fused, axis=1), 1.0, atol=1e-5)


def test_sid_and_ranking_metrics(tmp_path: Path) -> None:
    index = tmp_path / "index.json"; index.write_text(json.dumps({"a": [1, 2], "b": [1, 2], "c": [2, 3]}))
    assert collision_stats(index)["collision_rate"] == 1 / 3
    rows = [{"target": "a", "predictions": ["a", "x"]}, {"target": "c", "predictions": ["a", "c"]}]
    metrics = evaluate(rows, {"a", "b", "c"}, ["a", "a", "b", "c"], (5, 10, 20))
    assert metrics["hr@10"] == 1.0
    assert metrics["invalid_sid_rate"] == 0.25


def test_collision_group_metrics_ignore_disambiguation_suffix() -> None:
    index = {
        "a": ["<a_1>", "<b_1>", "<c_1>", "<d_1>"],
        "b": ["<a_1>", "<b_1>", "<c_1>", "<d_2>"],
        "c": ["<a_2>", "<b_2>", "<c_2>"],
    }
    a = "<a_1><b_1><c_1><d_1>"
    b = "<a_1><b_1><c_1><d_2>"
    c = "<a_2><b_2><c_2>"
    rows = [
        {"target": a, "predictions": [a]},
        {"target": b, "predictions": ["invalid"]},
        {"target": c, "predictions": [c]},
    ]
    metrics = evaluate(rows, {a, b, c}, [a, b, c], (10,), sid_index=index)
    assert metrics["samples_collision"] == 2
    assert metrics["samples_non_collision"] == 1
    assert metrics["hr@10_collision"] == 0.5
    assert metrics["hr@10_non_collision"] == 1.0


def test_recommendation_concentration_metrics_use_valid_beam_slots() -> None:
    rows = [
        {"target": "<a_0>", "predictions": ["<a_0>", "<a_0>", "<a_1>", "bad"]},
        {"target": "<a_1>", "predictions": ["<a_0>", "<a_2>", "<a_2>", "bad"]},
    ]
    metrics = evaluate(rows, {"<a_0>", "<a_1>", "<a_2>"}, ["<a_0>", "<a_1>", "<a_2>"])
    assert metrics["unique_recommended_items"] == 3
    assert metrics["valid_recommendation_slots"] == 6
    assert metrics["top20_item_count"] == 3
    assert metrics["top20_item_share"] == 1.0
    assert metrics["unique_a_prefixes"] == 3
    assert metrics["top20_a_prefix_share"] == 1.0


def test_item_text_handles_list_description() -> None:
    assert item_text({"title": "Widget", "description": ["red", "small"]}) == "Title: Widget. Description: red small"

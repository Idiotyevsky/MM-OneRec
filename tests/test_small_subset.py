import json
from pathlib import Path

from data.build_small_subset import build_subset


def test_build_subset_remaps_and_splits(tmp_path: Path) -> None:
    interactions = {"0": [0, 1, 2, 0, 1, 2], "1": [0, 1, 0, 1, 3], "2": [4, 4]}
    items = {str(index): {"title": str(index)} for index in range(5)}
    inter_path = tmp_path / "inter.json"; inter_path.write_text(json.dumps(interactions))
    item_path = tmp_path / "item.json"; item_path.write_text(json.dumps(items))
    stats = build_subset(inter_path, item_path, tmp_path / "out", "small", max_items=5, min_user_interactions=5)
    assert stats["users"] == 2
    assert stats["items"] == 4
    assert (tmp_path / "out/small.test.inter").read_text().count("\n") == 3

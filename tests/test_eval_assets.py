import json
from pathlib import Path

from scripts.mm.build_eval_assets import build_assets


def test_build_eval_assets_preserves_frequency_events(tmp_path: Path) -> None:
    index = tmp_path / "index.json"
    items = tmp_path / "items.json"
    train = tmp_path / "train.csv"
    output = tmp_path / "assets"
    index.write_text(json.dumps({"0": ["<a_0>", "<b_0>", "<c_0>"], "1": ["<a_1>", "<b_1>", "<c_1>"]}), encoding="utf-8")
    items.write_text(json.dumps({"0": {"title": "zero"}, "1": {"title": "one"}}), encoding="utf-8")
    train.write_text(
        "user_id,history_item_id,item_id,history_item_sid,item_sid\n"
        "0,\"['0']\",1,\"['<a_0><b_0><c_0>']\",<a_1><b_1><c_1>\n",
        encoding="utf-8",
    )
    stats = build_assets(index, items, train, output)
    assert stats == {"catalog_items": 2, "train_item_events": 2}
    assert (output / "catalog.txt").read_text().count("\n") == 2
    assert "<a_1><b_1><c_1>" in (output / "train-items.txt").read_text()

import csv
import importlib.util
import json
from pathlib import Path

from minionerec.data import SidItemFeatDataset


MODULE_PATH = Path(__file__).parents[1] / "scripts" / "mm" / "build_sid_csv.py"
SPEC = importlib.util.spec_from_file_location("build_sid_csv", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_sid_item_dataset_preserves_collision_suffix(tmp_path: Path) -> None:
    item_path = tmp_path / "items.json"
    index_path = tmp_path / "index.json"
    item_path.write_text(json.dumps({"0": {"title": "item"}}), encoding="utf-8")
    index_path.write_text(json.dumps({"0": ["<a_1>", "<b_2>", "<c_3>", "<d_1>"]}), encoding="utf-8")
    dataset = SidItemFeatDataset(item_path, index_path)
    assert dataset.title2sid["item"] == "<a_1><b_2><c_3><d_1>"


def test_build_sid_csv_uses_all_tokens(tmp_path: Path) -> None:
    interaction_path = tmp_path / "train.inter"
    index_path = tmp_path / "index.json"
    output_path = tmp_path / "train.csv"
    interaction_path.write_text(
        "user_id:token\titem_id_list:token_seq\titem_id:token\n0\t0 1\t2\n",
        encoding="utf-8",
    )
    index_path.write_text(
        json.dumps(
            {
                "0": ["<a_0>", "<b_0>", "<c_0>"],
                "1": ["<a_0>", "<b_0>", "<c_0>", "<d_1>"],
                "2": ["<a_2>", "<b_2>", "<c_2>"],
            }
        ),
        encoding="utf-8",
    )
    stats = MODULE.build_sid_csv(interaction_path, index_path, output_path)
    with output_path.open(encoding="utf-8") as handle:
        row = next(csv.DictReader(handle))
    assert stats["rows"] == 1
    assert json.loads(row["history_item_sid"])[1].endswith("<d_1>")
    assert row["item_sid"] == "<a_2><b_2><c_2>"

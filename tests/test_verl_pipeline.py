import csv
import json
from pathlib import Path

import torch

from rl.verl.grpo_math import clipped_surrogate, group_relative_advantages
from rl.verl.prepare_data import convert_csv_to_parquet
from rl.verl.reward import RecommendationReward, RewardConfig, semantic_sid_parts, sid_hier_reward


def test_sid_hier_reward_order_and_suffix_handling():
    target = "<a_1><b_2><c_3><d_9>"
    exact = sid_hier_reward(target, target)
    same_prefix = sid_hier_reward("<a_1><b_2><c_8>", target)
    weak = sid_hier_reward("<a_1><b_8><c_8>", target)
    unrelated = sid_hier_reward("<a_9><b_8><c_8>", target)
    assert exact > same_prefix > weak > unrelated
    assert semantic_sid_parts(target) == ["<a_1>", "<b_2>", "<c_3>"]
    assert sid_hier_reward("<a_1><b_2><c_3><d_1>", target) == 1.0


def test_reward_modes_make_invalid_zero():
    reward = RecommendationReward(RewardConfig(mode="exact"))
    assert reward.score("<a_1><b_2>", "<a_1><b_2>") == 1.0
    assert reward.score("not-a-sid", "<a_1><b_2>") == 0.0


def test_standard_grpo_ratio_and_group_advantage():
    rewards = torch.tensor([[1.0, 0.0, 0.5], [0.0, 1.0, 0.0]])
    advantages = group_relative_advantages(rewards)
    assert torch.allclose(advantages.mean(dim=1), torch.zeros(2), atol=1e-6)
    current = torch.log(torch.tensor([[1.3, 1.0], [0.8, 1.0]]))
    old = torch.zeros_like(current)
    loss, metrics = clipped_surrogate(current, old, torch.tensor([1.0, -1.0]), clip_ratio=0.2)
    assert torch.isfinite(loss)
    assert 0.0 <= metrics["clip_fraction"] <= 1.0
    assert metrics["ratio_max"] > 1.0


def test_verl_parquet_schema(tmp_path: Path):
    source = tmp_path / "train.csv"
    source.write_text(
        "user_id,history_item_id,item_id,history_item_sid,item_sid\n"
        "u1,\"['i1', 'i2']\",i3,\"['<a_1>', '<b_2>']\",<c_3>\n",
        encoding="utf-8",
    )
    output = tmp_path / "train.parquet"
    metadata = convert_csv_to_parquet(source, output, dataset="Amazon23", category="Test")
    assert metadata["rows"] == 1
    import pyarrow.parquet as pq

    table = pq.read_table(output)
    assert {"prompt", "target_sid", "target_item_id", "history_sid", "history_item_ids", "target_sid_parts", "dataset", "category"}.issubset(table.column_names)
    prompt = table.column("prompt")[0].as_py()
    assert isinstance(prompt, list)
    assert prompt[0]["role"] == "user"
    assert "### Instruction:" in prompt[0]["content"]
    assert "### Response:" in prompt[0]["content"]

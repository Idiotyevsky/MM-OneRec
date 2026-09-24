from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image

from multimodal.qwen3_vl_encoder import (
    Qwen3VLConfig,
    Qwen3VLItemEncoder,
    _project,
    pool_last_token,
    pool_text_mean,
)


class DummyProcessor:
    def __init__(self):
        self.image_calls = 0
        self.tokenizer = SimpleNamespace(convert_tokens_to_ids=lambda _: -1)

    def __call__(self, text, images=None, padding=True, return_tensors="pt"):
        if images is not None:
            self.image_calls += 1
        lengths = [max(2, len(value) % 5 + 2) for value in text]
        width = max(lengths)
        ids = torch.zeros((len(text), width), dtype=torch.long)
        mask = torch.zeros_like(ids)
        for row, length in enumerate(lengths):
            ids[row, :length] = torch.arange(1, length + 1)
            mask[row, :length] = 1
        return {"input_ids": ids, "attention_mask": mask}


class DummyModel(torch.nn.Module):
    def forward(self, input_ids, attention_mask, output_hidden_states=True, return_dict=True, **kwargs):
        hidden = input_ids.float().unsqueeze(-1).repeat(1, 1, 4)
        return SimpleNamespace(hidden_states=(hidden,))


def test_pooling_respects_attention_mask():
    hidden = torch.arange(24, dtype=torch.float32).reshape(2, 4, 3)
    mask = torch.tensor([[1, 1, 1, 0], [0, 1, 1, 1]])
    torch.testing.assert_close(pool_last_token(hidden, mask), torch.stack([hidden[0, 2], hidden[1, 3]]))
    expected = torch.stack([hidden[0, :3].mean(0), hidden[1, 1:].mean(0)])
    torch.testing.assert_close(pool_text_mean(hidden, mask), expected)


def test_qwen3vl_mock_preserves_order_and_missing_image_route(tmp_path: Path):
    image = tmp_path / "item.jpg"
    Image.new("RGB", (8, 8), (10, 20, 30)).save(image)
    processor = DummyProcessor()
    encoder = Qwen3VLItemEncoder(
        Qwen3VLConfig(device="cpu", pooling="last_token", normalize=False, batch_size=2),
        processor=processor,
        model=DummyModel(),
    )
    records = [{"title": "first"}, {"title": "second"}, {"title": "third"}]
    matrix, has_image = encoder.encode_records(records, [str(image), None, str(image)])
    assert matrix.shape == (3, 4)
    assert has_image.tolist() == [True, False, True]
    assert processor.image_calls == 1
    assert not np.allclose(matrix[0], matrix[1])
    assert not np.allclose(matrix[1], matrix[2])


def test_pca_projection_shape_and_metadata():
    matrix = np.arange(30, dtype=np.float32).reshape(6, 5)
    projected, metadata = _project(matrix, "pca", 3, 42)
    assert projected.shape == (6, 3)
    assert metadata["raw_dim"] == 5
    assert metadata["projected_dim"] == 3

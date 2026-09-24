from __future__ import annotations

import numpy as np
import torch

from rq.diagnostics import (
    apply_zscore,
    compute_codebook_diagnostics,
    embedding_statistics,
    fit_zscore,
)
from rq.models.rqvae import RQVAE


class _Dataset(torch.utils.data.Dataset):
    def __init__(self, values: np.ndarray) -> None:
        self.values = torch.from_numpy(values.astype(np.float32))

    def __len__(self) -> int:
        return len(self.values)

    def __getitem__(self, index: int) -> torch.Tensor:
        return self.values[index]


def test_embedding_statistics_and_zscore_are_finite() -> None:
    values = np.arange(24, dtype=np.float32).reshape(6, 4)
    stats = embedding_statistics(values, pairwise_sample_size=20)
    assert stats["num_items"] == 6
    assert stats["non_finite_count"] == 0
    transformed, scaler = fit_zscore(values)
    np.testing.assert_allclose(transformed.mean(axis=0), 0.0, atol=1e-6)
    np.testing.assert_allclose(apply_zscore(values, scaler), transformed, atol=1e-6)


def test_codebook_diagnostics_use_nearest_neighbor_assignment() -> None:
    torch.manual_seed(7)
    values = np.random.default_rng(7).normal(size=(24, 4)).astype(np.float32)
    model = RQVAE(
        in_dim=4,
        num_emb_list=[4, 4, 4],
        e_dim=2,
        layers=[8, 4],
        kmeans_init=False,
        sk_epsilons=[0.01, 0.01, 0.01],
    )
    metrics = compute_codebook_diagnostics(model, _Dataset(values), device="cpu", batch_size=8)
    assert metrics["assignment"] == "nearest_neighbor_use_sk_false"
    assert metrics["num_items"] == 24
    assert metrics["prefix_diversity"]["raw_unique_sid"] <= 24
    assert len(metrics["layers"]) == 3
    assert all(0 <= layer["utilization_rate"] <= 1 for layer in metrics["layers"])
    assert 0 <= metrics["prefix_diversity"]["collision_rate"] <= 1

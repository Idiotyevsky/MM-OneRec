import json
from pathlib import Path

import numpy as np

from multimodal.rec_alignment import AlignmentConfig, recency_pool, train_alignment


def test_recency_pool_and_train_only_alignment(tmp_path: Path):
    embeddings = np.eye(4, dtype=np.float32)
    emb_path = tmp_path / "vl.npy"
    np.save(emb_path, embeddings)
    items_path = tmp_path / "items.json"
    items_path.write_text(json.dumps({str(i): {"title": str(i)} for i in range(4)}), encoding="utf-8")
    train = tmp_path / "train.csv"
    train.write_text(
        "user_id,history_item_id,item_id,history_item_sid,item_sid\n"
        "u1,\"['0', '1']\",2,\"['a', 'b']\",c\n"
        "u2,\"['1', '2']\",3,\"['b', 'c']\",d\n",
        encoding="utf-8",
    )
    pooled = recency_pool(embeddings, ["0", "1"], {str(i): i for i in range(4)})
    assert pooled.shape == (4,)
    output = tmp_path / "aligned.npy"
    metadata = train_alignment(
        emb_path,
        train,
        output,
        AlignmentConfig(output_dim=3, hidden_dim=4, epochs=1, batch_size=2, negatives=2, device="cpu"),
        items_path=items_path,
    )
    assert metadata["behavior_split"] == "train_only"
    assert np.load(output).shape == (4, 3)

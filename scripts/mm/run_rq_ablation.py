#!/usr/bin/env python3
"""Run the controlled Qwen3-VL RQ-VAE ablation matrix.

The first five runs are fixed by the study protocol.  The latent/codebook
capacity runs are selected from the best collision configuration among those
five and are never mixed with a different preprocessing or Sinkhorn setting.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]

BASE = {
    "epochs": 100,
    "batch_size": 512,
    "num_workers": 0,
    "eval_step": 20,
    "lr": 1e-3,
    "warmup_epochs": 50,
    "num_emb_list": [32, 32, 32],
    "e_dim": 64,
    "layers": [512, 256, 128],
    "beta": 0.25,
    "quant_loss_weight": 1.0,
    "sk_iters": 50,
    "seed": 2024,
    "patience": 10,
}

CORE = [
    {"name": "Q0_none_sk0", "preprocess": "none", "sk_epsilons": [0.0, 0.0, 0.0]},
    {"name": "Q1_zscore_sk0", "preprocess": "zscore", "sk_epsilons": [0.0, 0.0, 0.0]},
    {"name": "Q2_none_sk001", "preprocess": "none", "sk_epsilons": [0.001, 0.001, 0.001]},
    {"name": "Q3_none_sk003", "preprocess": "none", "sk_epsilons": [0.003, 0.003, 0.003]},
    {"name": "Q4_zscore_sk003", "preprocess": "zscore", "sk_epsilons": [0.003, 0.003, 0.003]},
]


def _latest_checkpoint(run_root: Path) -> Path:
    candidates = sorted(run_root.glob("*/best_collision_model.pth"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(f"No best_collision_model.pth found under {run_root}")
    return candidates[-1]


def _run_one(args: argparse.Namespace, config: dict[str, Any]) -> dict[str, Any]:
    name = str(config["name"])
    run_dir = args.output_root / name
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_root = run_dir / "checkpoints"
    diagnostics_path = run_dir / "diagnostics.jsonl"
    stats_path = run_dir / "preprocess_stats.json"
    merged = dict(BASE)
    merged.update(config)
    manifest = {
        "name": name,
        "embedding": str(args.embedding),
        "config": merged,
        "seed": int(merged["seed"]),
    }
    (run_dir / "config.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    checkpoint = None
    if args.skip_existing:
        try:
            checkpoint = _latest_checkpoint(ckpt_root)
        except FileNotFoundError:
            pass
    if checkpoint is None:
        command = [
            sys.executable,
            str(ROOT / "rq" / "rqvae.py"),
            "--data_path", str(args.embedding),
            "--ckpt_dir", str(ckpt_root),
            "--lr", str(merged["lr"]),
            "--epochs", str(merged["epochs"]),
            "--batch_size", str(merged["batch_size"]),
            "--num_workers", str(merged["num_workers"]),
            "--eval_step", str(merged["eval_step"]),
            "--warmup_epochs", str(merged["warmup_epochs"]),
            "--device", args.device,
            "--num_emb_list", *map(str, merged["num_emb_list"]),
            "--e_dim", str(merged["e_dim"]),
            "--layers", *map(str, merged["layers"]),
            "--beta", str(merged["beta"]),
            "--quant_loss_weight", str(merged["quant_loss_weight"]),
            "--sk_epsilons", *map(str, merged["sk_epsilons"]),
            "--sk_iters", str(merged["sk_iters"]),
            "--seed", str(merged["seed"]),
            "--preprocess", str(merged["preprocess"]),
            "--preprocess_stats_path", str(stats_path),
            "--diagnostics_path", str(diagnostics_path),
            "--diagnostics_batch_size", str(args.diagnostics_batch_size),
            "--patience", str(merged["patience"]),
        ]
        log_path = run_dir / "train.log"
        print(f"[{name}] training -> {log_path}", flush=True)
        with log_path.open("w", encoding="utf-8") as log:
            log.write("$ " + " ".join(command) + "\n")
            result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, env=os.environ.copy())
        if result.returncode != 0:
            raise RuntimeError(f"{name} failed with exit code {result.returncode}; see {log_path}")
        checkpoint = _latest_checkpoint(ckpt_root)

    analysis_dir = run_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    analysis_metrics = analysis_dir / "metrics.json"
    if not args.skip_existing or not analysis_metrics.exists():
        command = [
            sys.executable,
            str(ROOT / "scripts" / "mm" / "analyze_codebook.py"),
            "--embedding", str(args.embedding),
            "--checkpoint", str(checkpoint),
            "--output-dir", str(analysis_dir),
            "--device", args.device,
            "--batch-size", str(args.diagnostics_batch_size),
            "--preprocess", str(merged["preprocess"]),
            "--item-meta", str(args.item_meta),
            "--item-ids", str(args.item_ids),
            "--prefix-examples", str(args.prefix_examples),
        ]
        if merged["preprocess"] == "zscore":
            command.extend(["--preprocess-stats", str(stats_path)])
        print(f"[{name}] nearest-neighbor analysis", flush=True)
        with (analysis_dir / "analysis.log").open("w", encoding="utf-8") as log:
            result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, env=os.environ.copy())
        if result.returncode != 0:
            raise RuntimeError(f"{name} analysis failed; see {analysis_dir / 'analysis.log'}")
    metrics = json.loads(analysis_metrics.read_text(encoding="utf-8"))
    metrics["experiment"] = name
    metrics["config"] = merged
    metrics["checkpoint"] = str(checkpoint)
    return metrics


def _row(metrics: dict[str, Any], embedding: Path) -> dict[str, Any]:
    config = metrics["config"]
    diversity = metrics["prefix_diversity"]
    row: dict[str, Any] = {
        "experiment": metrics["experiment"],
        "embedding": str(embedding),
        "preprocessing": config["preprocess"],
        "codebooks": "/".join(map(str, config["num_emb_list"])),
        "latent_dim": config["e_dim"],
        "beta": config["beta"],
        "quant_loss_weight": config["quant_loss_weight"],
        "sk_epsilons": "/".join(map(str, config["sk_epsilons"])),
        "best_epoch": metrics["epoch"],
        "reconstruction_loss": metrics["reconstruction_loss"],
        "raw_unique_sid": diversity["raw_unique_sid"],
        "collision_count": diversity["collision_count"],
        "collision_rate": diversity["collision_rate"],
    }
    for layer in metrics["layers"]:
        prefix = f"L{layer['layer']}"
        for key in ("used_codes", "utilization_rate", "entropy", "normalized_entropy", "perplexity", "most_frequent_code_ratio", "top5_code_ratio"):
            row[f"{prefix}_{key}"] = layer[key]
    for depth in range(1, metrics["num_layers"] + 1):
        row[f"prefix{depth}_unique"] = diversity.get(f"unique_prefix@{depth}", 0)
    return row


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embedding", type=Path, required=True)
    parser.add_argument("--item-meta", type=Path, required=True)
    parser.add_argument("--item-ids", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("outputs/qwen3vl/rq_ablation"))
    parser.add_argument("--results-csv", type=Path, default=Path("results/rq_ablation_qwen3vl.csv"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--diagnostics-batch-size", type=int, default=1024)
    parser.add_argument("--prefix-examples", type=int, default=5)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)

    metrics: list[dict[str, Any]] = []
    for config in CORE:
        metrics.append(_run_one(args, config))
        _write_csv([_row(item, args.embedding) for item in metrics], args.results_csv)

    best = min(metrics, key=lambda item: (item["prefix_diversity"]["collision_rate"], item["reconstruction_loss"]))
    best_config = dict(best["config"])
    capacity = [
        {
            "name": "Q5_best_latent128",
            "preprocess": best_config["preprocess"],
            "sk_epsilons": best_config["sk_epsilons"],
            "e_dim": 128,
            "layers": [512, 256],
            "num_emb_list": [32, 32, 32],
        },
        {
            "name": "Q6_best_latent128_codebook64",
            "preprocess": best_config["preprocess"],
            "sk_epsilons": best_config["sk_epsilons"],
            "e_dim": 128,
            "layers": [512, 256],
            "num_emb_list": [64, 64, 64],
        },
    ]
    for config in capacity:
        metrics.append(_run_one(args, config))
        _write_csv([_row(item, args.embedding) for item in metrics], args.results_csv)

    manifest = {
        "embedding": str(args.embedding),
        "core_experiments": [item["experiment"] for item in metrics[: len(CORE)]],
        "capacity_experiments": [item["experiment"] for item in metrics[len(CORE) :]],
        "best_core_experiment": best["experiment"],
        "best_core_config": best_config,
        "results_csv": str(args.results_csv),
    }
    (args.output_root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

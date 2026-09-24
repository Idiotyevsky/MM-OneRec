"""Launch native verl GRPO with MM-OneRec recommendation rewards.

The launcher intentionally delegates rollout, old-policy log probabilities,
clipped loss and reference KL to ``verl.trainer.main_ppo``.  It only builds
Hydra overrides and records the resolved MM-OneRec configuration.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "rl" / "verl" / "configs" / "grpo_small.yaml"


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - environment-specific
        raise RuntimeError("PyYAML is required to read a verl config") from exc
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected a mapping in {path}")
    return data


def _override(key: str, value: Any) -> str:
    if isinstance(value, bool):
        rendered = "true" if value else "false"
    elif isinstance(value, (int, float)):
        rendered = str(value)
    else:
        rendered = str(value)
    return f"{key}={rendered}"


def build_overrides(config: dict[str, Any], reward_path: Path, reward_config_path: Path | None = None) -> list[str]:
    """Translate the project config to current native verl config keys."""
    model_path = Path(str(config["model_path"])).expanduser()
    train_file = Path(str(config["train_file"])).expanduser()
    val_file = Path(str(config["val_file"])).expanduser()
    overrides = [
        _override("data.train_files", json.dumps([str(train_file.resolve())])),
        _override("data.val_files", json.dumps([str(val_file.resolve())])),
        _override("data.train_batch_size", config.get("train_batch_size", 64)),
        _override("data.max_prompt_length", config.get("max_prompt_length", 512)),
        _override("data.max_response_length", config.get("rollout_response_length", 16)),
        _override("actor_rollout_ref.model.path", str(model_path.resolve())),
        _override("+actor_rollout_ref.model.override_config.attn_implementation", config.get("attn_implementation", "sdpa")),
        _override("actor_rollout_ref.rollout.name", "vllm"),
        _override("actor_rollout_ref.rollout.tensor_model_parallel_size", config.get("tensor_model_parallel_size", 1)),
        _override("actor_rollout_ref.rollout.n", config.get("num_generations", 8)),
        _override("actor_rollout_ref.rollout.response_length", config.get("rollout_response_length", 16)),
        _override(
            "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu",
            config.get(
                "rollout_log_prob_micro_batch_size_per_gpu",
                config.get("ppo_micro_batch_size_per_gpu", 2),
            ),
        ),
        _override("actor_rollout_ref.actor.use_torch_compile", config.get("use_torch_compile", False)),
        _override("actor_rollout_ref.actor.ppo_epochs", config.get("ppo_epochs", 1)),
        _override("actor_rollout_ref.actor.ppo_mini_batch_size", config.get("ppo_mini_batch_size", 64)),
        _override("actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu", config.get("ppo_micro_batch_size_per_gpu", 2)),
        # Current verl releases validate reference-policy log-prob batching
        # independently from the actor micro-batch size.
        _override(
            "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu",
            config.get(
                "ref_log_prob_micro_batch_size_per_gpu",
                config.get("ppo_micro_batch_size_per_gpu", 2),
            ),
        ),
        _override("actor_rollout_ref.actor.clip_ratio", config.get("clip_ratio", 0.2)),
        _override("actor_rollout_ref.actor.use_kl_loss", True),
        _override("actor_rollout_ref.actor.kl_loss_coef", config.get("kl_loss_coef", 1e-3)),
        _override("algorithm.adv_estimator", "grpo"),
        _override("algorithm.use_kl_in_reward", False),
        _override("custom_reward_function.path", str(reward_path.resolve())),
        _override("custom_reward_function.name", "compute_reward"),
        _override("trainer.project_name", "MM-OneRec"),
        _override("trainer.experiment_name", config.get("experiment_name", "verl_grpo")),
        _override("trainer.default_local_dir", str(Path(str(config.get("output_dir", "outputs/verl_grpo"))).resolve())),
        _override("trainer.total_epochs", config.get("total_epochs", 1)),
        _override("trainer.n_gpus_per_node", config.get("gpus_per_node", 1)),
        _override("trainer.nnodes", config.get("nodes", 1)),
        _override("data.seed", config.get("seed", 42)),
        _override("trainer.logger", "['console']"),
        # Ray 2.58 + the cluster's OpenTelemetry package can fail while
        # starting the optional dashboard; GRPO does not depend on it.
        _override("+ray_kwargs.ray_init.include_dashboard", False),
    ]
    if config.get("save_freq") is not None:
        overrides.append(_override("trainer.save_freq", config.get("save_freq")))
    rollout_data_dir = config.get("rollout_data_dir")
    if rollout_data_dir:
        overrides.append(_override("trainer.rollout_data_dir", str(Path(str(rollout_data_dir)).resolve())))
    chat_template = config.get("chat_template")
    if chat_template:
        overrides.append(
            _override(
                "+data.apply_chat_template_kwargs.chat_template",
                json.dumps(str(chat_template)),
            )
        )
    max_steps = int(config.get("max_steps", -1))
    if max_steps > 0:
        overrides.append(_override("trainer.total_training_steps", max_steps))
    if reward_config_path is not None:
        overrides.append(_override("+custom_reward_function.reward_kwargs.config_path", str(reward_config_path.resolve())))
    return overrides


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--reward-config", type=Path, help="JSON RewardConfig with index/embedding paths")
    parser.add_argument("--dry-run", action="store_true", help="Print the native verl command without starting training")
    parser.add_argument("--constrained-rollout", action="store_true", help="Fail explicitly: training-time Trie guidance is not enabled in this version")
    args = parser.parse_args()
    config = _load_yaml(args.config)
    if args.constrained_rollout or config.get("training_rollout_constraint") not in {None, "unconstrained_invalid_zero"}:
        raise SystemExit("Training rollout Trie guidance is not implemented for this verl version; use unconstrained rollouts and invalid-SID reward=0. Evaluation remains Trie-constrained.")
    reward_path = ROOT / "rl" / "verl" / "reward.py"
    output_dir = Path(str(config.get("output_dir", "outputs/verl_grpo"))).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    configured_reward_config = config.get("reward_config")
    if args.reward_config:
        effective_reward_config = args.reward_config.resolve()
    elif configured_reward_config:
        effective_reward_config = (ROOT / str(configured_reward_config)).resolve()
    else:
        effective_reward_config = output_dir / "reward_config.json"
    if args.reward_config is None and not configured_reward_config:
        reward_payload = {
            "mode": config.get("reward_mode", "hybrid"),
            "lambda_sid": config.get("lambda_sid", 0.6),
            "sid_weights": tuple(config.get("sid_weights", [0.5, 0.3, 0.2])),
            "index_path": config.get("index_path"),
            "embedding_path": config.get("embedding_path"),
            "item_ids_path": config.get("item_ids_path"),
        }
        effective_reward_config.write_text(json.dumps(reward_payload, indent=2) + "\n", encoding="utf-8")
    overrides = build_overrides(config, reward_path, effective_reward_config)
    command = [sys.executable, "-m", "verl.trainer.main_ppo", *overrides]
    run_config = {**config, "launcher": "verl.trainer.main_ppo", "algorithm": "grpo", "reward_path": str(reward_path), "reward_config": str(effective_reward_config), "training_rollout_constraint": "unconstrained_invalid_zero", "overrides": overrides}
    (output_dir / "config.json").write_text(json.dumps(run_config, indent=2) + "\n", encoding="utf-8")
    print("Native verl command:")
    print(" ".join(shlex.quote(part) for part in command))
    if args.dry_run:
        return
    if importlib.util.find_spec("verl") is None:
        raise SystemExit("verl is not installed in this environment. Install a compatible verl release, then rerun this command; no legacy trainer fallback is performed.")
    env = os.environ.copy()
    env["MM_ONEREC_REWARD_CONFIG"] = str(effective_reward_config)
    subprocess.run(command, cwd=ROOT, env=env, check=True)


if __name__ == "__main__":
    main()

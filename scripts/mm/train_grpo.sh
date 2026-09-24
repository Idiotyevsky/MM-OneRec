#!/usr/bin/env bash
set -euo pipefail

# Standard GRPO entry point. The historical TRL/ReReTrainer launcher is kept
# as scripts/mm/train_legacy_grpo.sh and scripts/rl.sh.
python -m rl.verl.run_grpo "$@"

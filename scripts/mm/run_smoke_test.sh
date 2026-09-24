#!/usr/bin/env bash
set -euo pipefail
python -m pytest tests/test_multimodal.py tests/test_qwen3vl_encoder.py tests/test_verl_pipeline.py tests/test_rec_alignment.py -q

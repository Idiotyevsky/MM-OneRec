import os
import sys
import re
from collections import Counter
from typing import List
from pathlib import Path

# Support the documented `python scripts/sft.py ...` invocation from repo root.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np 
import fire
import torch
import torch.nn.functional as F
import transformers
from datasets import load_dataset, concatenate_datasets
from transformers import EarlyStoppingCallback, AutoConfig, TrainerCallback
from typing import TYPE_CHECKING, Any, Dict, List, NamedTuple, Optional, Sequence, Tuple, Union
from dataclasses import dataclass
import torch.nn as nn
import math
import warnings
from functools import partial
import numpy as np 
import fire
import transformers
from torch.optim.lr_scheduler import LambdaLR
import json
import torch.nn as nn
try:
    import bitsandbytes as bnb
except ImportError:
    bnb = None
from transformers import AutoModelForCausalLM, AutoTokenizer
from minionerec.data import D3Dataset, SFTData, SidSFTDataset, SidItemFeatDataset, FusionSeqRecDataset, PreferenceSFTDataset, UserPreference2sidSFTDataset, TitleHistory2SidSFTDataset
import random
from datasets import Dataset as HFDataset
from torch.utils.data import ConcatDataset


class TokenExtender:
    def __init__(self, data_path=None, dataset=None, index_file=".index.json", index_path=None):
        self.data_path = data_path
        self.dataset = dataset
        self.index_file = index_file
        self.index_path = index_path
        self.indices = None
        self.new_tokens = None
        
    def _load_data(self):
        path = self.index_path or os.path.join(self.data_path, self.dataset + self.index_file)
        with open(path, 'r') as f:
            self.indices = json.load(f)
    
    def get_new_tokens(self):
        if self.new_tokens is not None:
            return self.new_tokens
            
        if self.indices is None:
            self._load_data()
        
        self.new_tokens = set()
        for index in self.indices.values():
            for token in index:
                self.new_tokens.add(token)
        self.new_tokens = sorted(list(self.new_tokens))
        
        return self.new_tokens

    def get_token_stats(self):
        """Return the exact SID vocabulary and collision-suffix statistics."""
        tokens = self.get_new_tokens()
        by_prefix = Counter()
        suffix_numbers = []
        for token in tokens:
            match = re.fullmatch(r"<([a-z])_(\d+)>", token)
            if match:
                prefix, number = match.groups()
                by_prefix[prefix] += 1
                if prefix == "d":
                    suffix_numbers.append(int(number))
        return {
            "token_count": len(tokens),
            "tokens": tokens,
            "tokens_by_prefix": dict(sorted(by_prefix.items())),
            "max_collision_suffix": max(suffix_numbers, default=0),
            "collision_suffix_count": len(suffix_numbers),
        }


class TrainingProgressCallback(TrainerCallback):
    log_interval = 100

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not state.is_local_process_zero:
            return
        if not logs:
            return
        max_steps = state.max_steps if state.max_steps and state.max_steps > 0 else None
        step = state.global_step
        if step <= 0:
            return
        if max_steps:
            if step % self.log_interval != 0 and step != max_steps:
                return
        else:
            if step % self.log_interval != 0:
                return
        if max_steps:
            pct = min(100.0, step * 100.0 / max_steps)
            loss_val = logs.get("loss")
            if loss_val is not None:
                print(f"[SFT Progress] step {step}/{max_steps} ({pct:.2f}%) | loss={loss_val:.6f}")
            else:
                print(f"[SFT Progress] step {step}/{max_steps} ({pct:.2f}%)")
        else:
            loss_val = logs.get("loss")
            if loss_val is not None:
                print(f"[SFT Progress] step {step} | loss={loss_val:.6f}")
            else:
                print(f"[SFT Progress] step {step}")



_FIRST_PREFIX_RE = re.compile(r"<a_\d+>")


def _first_sid_prefix_from_labels(label_ids, tokenizer):
    """Decode the supervised target and return its first-level SID token."""
    target_ids = [int(token_id) for token_id in label_ids if int(token_id) != -100]
    if not target_ids:
        return None
    target_text = tokenizer.decode(target_ids, skip_special_tokens=False)
    match = _FIRST_PREFIX_RE.search(target_text)
    return match.group(0) if match else None


def _build_frequency_weights(rec_dataset, tokenizer):
    """Build first-level popularity weights from recommendation targets only."""
    counts = Counter()
    missing = 0
    for idx in range(len(rec_dataset)):
        prefix = _first_sid_prefix_from_labels(rec_dataset[idx]["labels"], tokenizer)
        if prefix is None:
            missing += 1
            continue
        counts[prefix] += 1
    if missing:
        raise ValueError(
            f"Could not extract first-level SID prefix from {missing}/{len(rec_dataset)} "
            "recommendation targets; refusing to train with incomplete frequency weights."
        )
    if not counts:
        raise ValueError("No recommendation targets were available for frequency weighting.")

    epsilon = 1e-8
    clip_min, clip_max = 0.5, 2.0
    raw = {prefix: 1.0 / math.sqrt(float(freq) + epsilon) for prefix, freq in counts.items()}
    raw_mean = float(np.mean(list(raw.values())))
    normalized = {prefix: value / raw_mean for prefix, value in raw.items()}
    weights = {
        prefix: float(np.clip(value, clip_min, clip_max))
        for prefix, value in normalized.items()
    }
    sorted_by_freq = sorted(counts.items(), key=lambda pair: (-pair[1], pair[0]))
    sorted_by_rare = sorted(counts.items(), key=lambda pair: (pair[1], pair[0]))
    final_values = list(weights.values())
    stats = {
        "min": float(min(final_values)),
        "max": float(max(final_values)),
        "mean": float(np.mean(final_values)),
        "median": float(np.median(final_values)),
        "num_prefixes": len(counts),
        "num_recommendation_samples": int(len(rec_dataset)),
        "missing_prefix_samples": int(missing),
        "epsilon": epsilon,
        "clip": [clip_min, clip_max],
    }
    payload = {
        "formula": "clip((1/sqrt(freq+1e-8))/mean(raw), 0.5, 2.0)",
        "stats": stats,
        "prefix_counts": {prefix: int(counts[prefix]) for prefix in sorted(counts)},
        "raw_weights": {prefix: float(raw[prefix]) for prefix in sorted(raw)},
        "weights": {prefix: float(weights[prefix]) for prefix in sorted(weights)},
        "top_frequency_prefixes": [
            {"prefix": prefix, "frequency": int(freq), "weight": weights[prefix]}
            for prefix, freq in sorted_by_freq[:10]
        ],
        "lowest_frequency_prefixes": [
            {"prefix": prefix, "frequency": int(freq), "weight": weights[prefix]}
            for prefix, freq in sorted_by_rare[:10]
        ],
    }
    return weights, payload


def _python_value(value):
    """Convert tensor/NumPy values into values accepted by datasets.Dataset."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


class FrequencyAwareDataCollator:
    """Keep per-sample metadata while delegating padding to the HF collator."""

    def __init__(self, base_collator):
        self.base_collator = base_collator

    def __call__(self, features):
        weights = [float(feature.get("sample_weight", 1.0)) for feature in features]
        task_types = [feature.get("task_type", "auxiliary") for feature in features]
        stripped = [
            {key: value for key, value in feature.items() if key not in {"sample_weight", "task_type"}}
            for feature in features
        ]
        batch = self.base_collator(stripped)
        batch["sample_weight"] = torch.tensor(weights, dtype=torch.float32)
        batch["task_type"] = task_types
        return batch


class FrequencyAwareTrainer(transformers.Trainer):
    """Trainer with first-level popularity weighting for recommendation rows."""

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        sample_weight = inputs.pop("sample_weight", None)
        inputs.pop("task_type", None)
        labels = inputs.get("labels")
        outputs = model(**inputs)
        if labels is None:
            loss = outputs.loss
        else:
            logits = outputs.logits
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            token_loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                reduction="none",
                ignore_index=-100,
            ).view(shift_labels.size(0), -1)
            valid_tokens = shift_labels.ne(-100)
            per_sample_loss = token_loss.sum(dim=1) / valid_tokens.sum(dim=1).clamp_min(1)
            if sample_weight is None:
                sample_weight = torch.ones_like(per_sample_loss)
            else:
                sample_weight = sample_weight.to(per_sample_loss.device, dtype=per_sample_loss.dtype)
            loss = (per_sample_loss * sample_weight).mean()
        return (loss, outputs) if return_outputs else loss

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def _get_cosine_schedule_with_warmup_lr_lambda(
    current_step, *, num_warmup_steps, num_training_steps, num_cycles
):
    if current_step < num_warmup_steps:
        return max(0.1, float(current_step) / float(max(1, num_warmup_steps)))
    progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
    return max(0.1, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

def get_cosine_schedule_with_warmup(
    optimizer, num_warmup_steps, num_training_steps, num_cycles: float = 0.5, last_epoch: int = -1
):

    lr_lambda = partial(
        _get_cosine_schedule_with_warmup_lr_lambda,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        num_cycles=num_cycles,
    )
    return LambdaLR(optimizer, lr_lambda, last_epoch)



def train(
    # model/data params
    base_model: str = "",  # the only required argument
    train_file: str="",
    eval_file: str="",
    output_dir: str = "",
    sample: int = -1,
    seed: int = 42,
    
    # training hyperparams
    batch_size: int = 128,
    micro_batch_size: int = 4,
    num_epochs: int = 10,
    learning_rate: float = 3e-4,
    cutoff_len: int = 512,
    # llm hyperparams
    group_by_length: bool = False,  # faster, but produces an odd training loss curve
    freeze_LLM: bool = False,  # freeze LLM parameters, only train new token embeddings
    # wandb params
    wandb_project: str = "",
    wandb_run_name: str = "",
    resume_from_checkpoint: str = None,  # either training checkpoint or final adapter
    category: str="",
    train_from_scratch: bool = False,
    sid_index_path: str = "",
    item_meta_path: str = "",
    max_steps: int = -1,
    rec_repeat: int = 2,
    frequency_aware: bool = False,
    prefix_weights_path: str = "",
    gradient_checkpointing: bool = True,
    save_only_model: bool = True,
):
    print("[SFT] initializing training function", flush=True)
    set_seed(seed)
    os.environ['WANDB_PROJECT'] = wandb_project
    os.environ.setdefault("WANDB_MODE", "offline")
    category_dict = {"Industrial_and_Scientific": "industrial and scientific items", "Office_Products": "office products", "Toys_and_Games": "toys and games", "Sports": "sports and outdoors", "Books": "books"}
    print(category)
    category_name = category
    category = category_dict[category]
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='decapoda-research/llama-7b-hf'"
    gradient_accumulation_steps = batch_size // micro_batch_size
    print(f"[SFT] base_model={base_model}", flush=True)
    
    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size

    if not train_from_scratch:
        print("[SFT] loading pretrained model...", flush=True)
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.bfloat16,
        )
        print("[SFT] model loaded", flush=True)
    else:
        print("[SFT] building model from config...", flush=True)
        config = AutoConfig.from_pretrained(base_model)
        model = AutoModelForCausalLM.from_config(config)
        print("Training from scratch!")
        
    print("[SFT] loading tokenizer...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    original_vocab_size = len(tokenizer)
    new_tokens = []
    tokenizer_extension = {
        "base_vocab_size": original_vocab_size,
        "added_tokens": 0,
        "tokens_by_prefix": {},
        "max_collision_suffix": 0,
        "collision_suffix_count": 0,
        "atomic_check": "not_run",
    }
    
    if sid_index_path and os.path.exists(sid_index_path):
        print(f"Loading index from {sid_index_path}")
        token_extender = TokenExtender(index_path=sid_index_path)
        new_tokens = token_extender.get_new_tokens()
        if new_tokens:
            print(f"Adding {len(new_tokens)} new tokens to tokenizer")
            added_count = tokenizer.add_tokens(new_tokens)
            model.resize_token_embeddings(len(tokenizer))
            bad_tokens = {
                token: tokenizer.encode(token, add_special_tokens=False)
                for token in new_tokens
                if len(tokenizer.encode(token, add_special_tokens=False)) != 1
            }
            if bad_tokens:
                examples = list(bad_tokens.items())[:5]
                raise ValueError(
                    "SID tokens must be atomic after tokenizer extension; "
                    f"found {len(bad_tokens)} non-atomic tokens, examples={examples}"
                )
            tokenizer_extension = {
                "base_vocab_size": original_vocab_size,
                "final_vocab_size": len(tokenizer),
                "added_tokens": int(added_count),
                **token_extender.get_token_stats(),
                "atomic_check": "passed",
            }
        else:
            tokenizer_extension["atomic_check"] = "no_sid_tokens"

    # Freeze LLM parameters if required
    if freeze_LLM:
        print("Freezing LLM parameters, only training new token embeddings")
        for param in model.parameters():
            param.requires_grad = False

        if sid_index_path and os.path.exists(sid_index_path) and new_tokens:
            embedding_layer = model.get_input_embeddings()
            if embedding_layer.weight.shape[0] > original_vocab_size:
                embedding_layer.weight.requires_grad = True

                def mask_grad(grad):
                    # grad shape: [vocab_size, hidden_dim]
                    grad[:original_vocab_size].zero_()
                    return grad
                
                embedding_layer.weight.register_hook(mask_grad)

                print(f"Unfrozen {len(new_tokens)} new token embeddings "
                    f"(indices {original_vocab_size} to {len(tokenizer)-1})")

        else:
            print("Warning: freeze_LLM=True but no new tokens added. All parameters are frozen!")

        # Print the number of trainable parameters (it will still report the size of the entire embedding matrix, but only the newly added rows will have non-zero gradients).
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params     = sum(p.numel() for p in model.parameters())
        print(f"Trainable parameters (with grad-mask): {trainable_params:,} / "
            f"{total_params:,} ({100*trainable_params/total_params:.2f}%)")
        
    print("[SFT] preparing datasets...", flush=True)
    train_data1 = SidSFTDataset(train_file=train_file, tokenizer=tokenizer, max_len=cutoff_len,
                                 sample=sample, seed=seed, category=category)
    rec_repeat = max(1, int(rec_repeat))
    train_data2 = SidItemFeatDataset(item_file=item_meta_path, index_file=sid_index_path,
                                      tokenizer=tokenizer, max_len=cutoff_len, sample=sample,
                                      seed=seed, category=category)
    train_data3 = FusionSeqRecDataset(train_file=train_file, item_file=item_meta_path,
                                       index_file=sid_index_path, tokenizer=tokenizer,
                                       max_len=cutoff_len, sample=sample, seed=seed,
                                       category=category)
    val_data = SidSFTDataset(train_file=eval_file, tokenizer=tokenizer, max_len=cutoff_len,
                             sample=sample, seed=seed, category=category)
    composition = {
        "recommendation_next_sid": len(train_data1) * rec_repeat,
        "sid_item_feature_auxiliary": len(train_data2),
        "fusion_title_auxiliary": len(train_data3),
        "total": len(train_data1) * rec_repeat + len(train_data2) + len(train_data3),
        "recommendation_fraction": (len(train_data1) * rec_repeat) / max(
            len(train_data1) * rec_repeat + len(train_data2) + len(train_data3), 1
        ),
        "rec_repeat": rec_repeat,
    }
    print(f"[SFT] sample composition={composition}", flush=True)
    print("LOAD DATA FINISHED")

    if resume_from_checkpoint:
        checkpoint_name = os.path.join(resume_from_checkpoint, "pytorch_model.bin")

    if not ddp and torch.cuda.device_count() > 1:
        model.is_parallelizable = True
        model.model_parallel = True

    sample_frac = 1
    frequency_payload = None
    if frequency_aware:
        prefix_weights, frequency_payload = _build_frequency_weights(train_data1, tokenizer)
        os.makedirs(output_dir, exist_ok=True)
        weights_file = prefix_weights_path or os.path.join(output_dir, "prefix_weights.json")
        with open(weights_file, "w", encoding="utf-8") as handle:
            json.dump(frequency_payload, handle, indent=2)
        print(f"[SFT] frequency weights saved to {weights_file}", flush=True)
        print(f"[SFT] frequency weight stats={frequency_payload['stats']}", flush=True)
        for row in frequency_payload["top_frequency_prefixes"][:5]:
            print(f"[SFT] popular prefix={row['prefix']} freq={row['frequency']} weight={row['weight']:.6f}", flush=True)
        for row in frequency_payload["lowest_frequency_prefixes"][:5]:
            print(f"[SFT] rare prefix={row['prefix']} freq={row['frequency']} weight={row['weight']:.6f}", flush=True)

        rec_rows = []
        for idx in range(len(train_data1)):
            item = train_data1[idx]
            prefix = _first_sid_prefix_from_labels(item["labels"], tokenizer)
            if prefix not in prefix_weights:
                raise ValueError(f"Missing frequency weight for target prefix {prefix!r}")
            row = {key: _python_value(value) for key, value in item.items()}
            row["task_type"] = "recommendation"
            row["sample_weight"] = float(prefix_weights[prefix])
            rec_rows.append(row)
        aux_rows = []
        for dataset in (train_data2, train_data3):
            for idx in range(len(dataset)):
                item = dataset[idx]
                row = {key: _python_value(value) for key, value in item.items()}
                row["task_type"] = "auxiliary"
                row["sample_weight"] = 1.0
                aux_rows.append(row)
        rows = []
        for _ in range(rec_repeat):
            rows.extend(dict(row) for row in rec_rows)
        rows.extend(aux_rows)
        hf_train_dataset = HFDataset.from_list(rows)
    else:
        train_datasets = [train_data1] * rec_repeat + [train_data2, train_data3]
        train_data = ConcatDataset(train_datasets)
        hf_train_dataset = HFDataset.from_dict({k: [v[k] for v in train_data] for k in train_data[0].keys()})

    hf_train_dataset = hf_train_dataset.shuffle(seed=42).select(range(int(sample_frac * len(hf_train_dataset))))
    hf_val_dataset = HFDataset.from_dict({k: [v[k] for v in val_data] for k in val_data[0].keys()}).shuffle(seed=seed)
    hf_val_dataset = hf_val_dataset.shuffle(seed=42)

    print(hf_train_dataset)
    print(hf_val_dataset)
    print("[SFT] building Trainer...", flush=True)
    use_cuda = torch.cuda.is_available()
    base_collator = transformers.DataCollatorForSeq2Seq(
        tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
    )
    trainer_cls = FrequencyAwareTrainer if frequency_aware else transformers.Trainer
    data_collator = FrequencyAwareDataCollator(base_collator) if frequency_aware else base_collator
    trainer = trainer_cls(
        # deepspeed=deepspeed,
        model=model,
        train_dataset=hf_train_dataset,
        eval_dataset=hf_val_dataset,
        args=transformers.TrainingArguments(
            # deepspeed=deepspeed,
            run_name=wandb_run_name,
            per_device_train_batch_size=micro_batch_size,
            per_device_eval_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=20,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            max_steps=max_steps,
            bf16=use_cuda,
            gradient_checkpointing=gradient_checkpointing,
            save_only_model=save_only_model,
            logging_steps=100,
            optim="adamw_torch",
            # Epoch-level evaluation keeps smoke runs fast and avoids writing a
            # full optimizer checkpoint after nearly every short-run step.
            eval_strategy="epoch",
            save_strategy="epoch",
            output_dir=output_dir,
            save_total_limit=1,
            load_best_model_at_end=True,
            ddp_find_unused_parameters=False if ddp else None,
            remove_unused_columns=False if frequency_aware else True,
            group_by_length=group_by_length,
            report_to="wandb",
            disable_tqdm=False,
        ),
        data_collator=data_collator,
        callbacks = [EarlyStoppingCallback(early_stopping_patience=3), TrainingProgressCallback()],
        # optimizers=(optimizer, lr_scheduler) 
    )
    model.config.use_cache = False
    print("[SFT] trainer ready, starting train()", flush=True)

    os.makedirs(output_dir, exist_ok=True)
    run_config = {
        "base_model": base_model,
        "train_file": train_file,
        "eval_file": eval_file,
        "sample": sample,
        "seed": seed,
        "batch_size": batch_size,
        "micro_batch_size": micro_batch_size,
        "num_epochs": num_epochs,
        "learning_rate": learning_rate,
        "cutoff_len": cutoff_len,
        "category": category_name,
        "sid_index_path": sid_index_path,
        "item_meta_path": item_meta_path,
        "freeze_LLM": freeze_LLM,
        "max_steps": max_steps,
        "rec_repeat": rec_repeat,
        "frequency_aware": frequency_aware,
        "prefix_weights_path": prefix_weights_path,
        "frequency_weighting": frequency_payload,
        "gradient_checkpointing": gradient_checkpointing,
        "save_only_model": save_only_model,
        "dataset_composition": composition,
        "tokenizer_extension": tokenizer_extension,
    }
    with open(os.path.join(output_dir, "config.json"), "w", encoding="utf-8") as handle:
        json.dump(run_config, handle, indent=2)
    with open(os.path.join(output_dir, "tokenizer_extension.json"), "w", encoding="utf-8") as handle:
        json.dump(tokenizer_extension, handle, indent=2)

    train_result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    eval_metrics = trainer.evaluate()
    metrics = {**train_result.metrics, **eval_metrics}
    with open(os.path.join(output_dir, "metrics.json"), "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    with open(os.path.join(output_dir, "training_log.json"), "w", encoding="utf-8") as handle:
        json.dump(trainer.state.log_history, handle, indent=2)
    final_checkpoint_dir = os.path.join(output_dir, "final_checkpoint")
    trainer.model.save_pretrained(final_checkpoint_dir)
    tokenizer.save_pretrained(final_checkpoint_dir)
    with open(os.path.join(final_checkpoint_dir, "tokenizer_extension.json"), "w", encoding="utf-8") as handle:
        json.dump(tokenizer_extension, handle, indent=2)



if __name__ == "__main__":
    fire.Fire(train)

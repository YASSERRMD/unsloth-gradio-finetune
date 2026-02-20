import os
import json
import threading
from typing import Callable

import torch
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset as hf_load
from .paths import new_run, save_config, save_manifest
from .dataset_io import load_hf_dataset, load_local_file
from .normalize import normalize
from .render import render_for_training


def run_sft(
    cfg: dict, dataset_spec: dict, cancel_flag: threading.Event, on_log: Callable
) -> str:
    run_dir = new_run(cfg)

    reproducibility_info = {
        "unsloth_version": "latest",
        "transformers_version": "4.38.0",
        "torch_version": torch.__version__,
        "seed": cfg.get("seed", 42),
    }
    cfg_with_repro = {**cfg, **reproducibility_info}

    save_config(run_dir, cfg_with_repro)
    on_log(f"Run dir: {run_dir}")

    on_log("Loading model...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg["base_model"],
        max_seq_length=cfg["max_seq_length"],
        dtype=cfg.get("dtype"),
        load_in_4bit=cfg.get("load_in_4bit", True),
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=cfg["lora_r"],
        target_modules=cfg["target_modules"],
        lora_alpha=cfg["lora_alpha"],
        lora_dropout=cfg["lora_dropout"],
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=cfg["seed"],
    )
    on_log("Model loaded and LoRA applied.")
    if cancel_flag.is_set():
        on_log("Cancelled after model load.")
        return run_dir

    on_log("Loading dataset...")
    mode = dataset_spec.get("mode", "HF Dataset")

    if mode == "HF Dataset":
        hf_ds = dataset_spec.get("hf_dataset")
        subset = dataset_spec.get("subset")
        split = dataset_spec.get("split", "train")
        ds = hf_load(hf_ds, subset, split=split)
        records = [dict(row) for row in ds]
    else:
        upload_path = dataset_spec.get("upload_path")
        records = []
        if upload_path.endswith(".jsonl"):
            with open(upload_path) as f:
                for line in f:
                    records.append(json.loads(line))
        elif upload_path.endswith(".json"):
            with open(upload_path) as f:
                records = json.load(f)
        elif upload_path.endswith(".csv"):
            import pandas as pd

            df = pd.read_csv(upload_path)
            records = df.to_dict("records")

    fmt = dataset_spec.get("format", "auto")
    column_map = {}
    if fmt == "csv_custom":
        column_map = {
            "user": dataset_spec.get("col_user", "user"),
            "assistant": dataset_spec.get("col_assistant", "assistant"),
            "system": dataset_spec.get("col_system"),
        }

    on_log("Normalizing dataset...")
    normalized = normalize(records, fmt=fmt, column_map=column_map)
    on_log(f"Normalized {len(normalized)} records.")

    on_log("Rendering for training...")
    train_dataset = render_for_training(normalized, tokenizer)

    save_manifest(
        run_dir,
        {
            "source": dataset_spec,
            "n_records": len(normalized),
            "format": fmt,
        },
    )

    adapter_path = os.path.join(run_dir, "artifacts", "lora")
    os.makedirs(adapter_path, exist_ok=True)

    training_args = TrainingArguments(
        per_device_train_batch_size=cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        warmup_steps=cfg["warmup_steps"],
        num_train_epochs=cfg["epochs"],
        learning_rate=cfg["learning_rate"],
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        optim=cfg["optim"],
        lr_scheduler_type=cfg["lr_scheduler_type"],
        seed=cfg["seed"],
        output_dir=os.path.join(run_dir, "artifacts", "checkpoints"),
        logging_dir=os.path.join(run_dir, "artifacts", "logs"),
        logging_steps=10,
        save_strategy="epoch",
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        args=training_args,
    )

    on_log("Starting training...")
    for step in range(int(cfg["epochs"])):
        if cancel_flag.is_set():
            on_log(f"Cancelled at epoch {step + 1}")
            break
        trainer.train()
        on_log(f"Epoch {step + 1} completed")

    on_log("Saving adapter...")
    model.save_pretrained(adapter_path)
    tokenizer.save_pretrained(adapter_path)

    on_log(f"Training done. Adapter saved to {adapter_path}")
    return run_dir

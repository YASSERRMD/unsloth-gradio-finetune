import json
import os
import torch
import numpy as np
from .paths import save_metrics


def run_eval(cfg: dict, run_dir: str) -> dict:
    adapter_path = os.path.join(run_dir, "artifacts", "lora")
    if not os.path.exists(adapter_path):
        return {"error": "No adapter found. Run training first."}

    try:
        from unsloth import FastLanguageModel
        from tqdm import tqdm
        from torch.utils.data import DataLoader

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=adapter_path,
            max_seq_length=2048,
            dtype=None,
            load_in_4bit=True,
        )
        FastLanguageModel.for_inference(model)

        manifest_path = os.path.join(run_dir, "dataset_manifest.json")
        if os.path.exists(manifest_path):
            with open(manifest_path) as f:
                manifest = json.load(f)
            source = manifest.get("source", {})

            from .dataset_io import load_hf_dataset, load_local_file
            from .normalize import normalize
            from .render import render_for_training

            mode = source.get("mode", "HF Dataset")
            if mode == "HF Dataset":
                ds = load_hf_dataset(
                    source.get("hf_dataset"),
                    source.get("subset"),
                    source.get("split", "train"),
                )
                records = [dict(row) for row in ds]
            else:
                path = source.get("upload_path")
                df, _ = load_local_file(path)
                records = df.to_dict("records")

            fmt = source.get("format", "auto")
            column_map = None
            if fmt == "csv_custom":
                column_map = {
                    "user": source.get("col_user", "user"),
                    "assistant": source.get("col_assistant", "assistant"),
                    "system": source.get("col_system"),
                }

            normalized = normalize(records, fmt=fmt, column_map=column_map)
            train_dataset = render_for_training(normalized, tokenizer)

            val_size = min(50, len(train_dataset))
            val_dataset = train_dataset.select(range(val_size))

            model.eval()
            total_loss = 0.0
            with torch.no_grad():
                for i in tqdm(range(val_size), desc="Evaluating"):
                    text = val_dataset[i]["text"]
                    inputs = tokenizer(
                        text, return_tensors="pt", truncation=True, max_length=2048
                    ).to("cuda")
                    outputs = model(**inputs, labels=inputs["input_ids"])
                    total_loss += outputs.loss.item()

            avg_loss = total_loss / val_size
            perplexity = np.exp(avg_loss)

            metrics = {
                "val_loss": avg_loss,
                "val_perplexity": perplexity,
                "n_samples": val_size,
            }
        else:
            metrics = {
                "val_loss": None,
                "val_perplexity": None,
                "n_samples": 0,
                "warning": "No dataset manifest found",
            }

        save_metrics(run_dir, metrics)
        return metrics

    except Exception as e:
        return {"error": str(e)}


def run_prompt_suite(
    model, tokenizer, run_dir: str, suite_path: str = "assets/prompt_suite.jsonl"
) -> list:
    from unsloth import FastLanguageModel

    FastLanguageModel.for_inference(model)
    results = []
    if not os.path.exists(suite_path):
        return results
    with open(suite_path) as f:
        for line in f:
            item = json.loads(line)
            inputs = tokenizer.apply_chat_template(
                [{"role": "user", "content": item["prompt"]}],
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
            ).to("cuda")
            with torch.no_grad():
                output = model.generate(
                    input_ids=inputs, max_new_tokens=256, use_cache=True
                )
            actual = tokenizer.decode(
                output[0][inputs.shape[1] :], skip_special_tokens=True
            )
            results.append(
                {
                    "prompt": item["prompt"],
                    "expected": item.get("expected"),
                    "actual": actual,
                }
            )
    out_path = os.path.join(run_dir, "artifacts", "prompt_suite_outputs.jsonl")
    with open(out_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    return results

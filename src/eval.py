import json
import os
import torch
from .paths import save_metrics


def run_eval(cfg: dict, run_dir: str) -> dict:
    metrics = {"val_loss": None, "val_perplexity": None, "n_samples": 0}
    save_metrics(run_dir, metrics)
    return metrics


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

import os
import json
import threading
import traceback
import gradio as gr

from src.dataset_io import load_hf_dataset, save_uploaded_file, load_local_file
from src.normalize import normalize
from src.render import render_for_training
from src.train import run_sft
from src.eval import run_eval, run_prompt_suite
from src.hub import push_lora, push_merged, push_gguf
from src.export import merge_to_local
from src.paths import list_runs
from src.schema import CanonicalRecord

import pandas as pd

_cancel_flag = threading.Event()
_loaded_model = None
_loaded_tokenizer = None


def require_token():
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUB_TOKEN")
    if not token:
        raise RuntimeError(
            "HF_TOKEN not found. Set it as an environment variable or Colab/Spaces secret."
        )
    return token


def build_cfg(
    hf_username,
    run_name,
    base_model,
    custom_model,
    max_seq,
    dtype,
    load_4bit,
    lora_r,
    lora_alpha,
    lora_dropout,
    target_modules,
    epochs,
    batch_size,
    grad_accum,
    lr,
    warmup_steps,
    optimizer,
    scheduler,
    seed,
):
    final_model = custom_model if custom_model else base_model
    return {
        "hf_username": hf_username,
        "run_name": run_name,
        "base_model": final_model,
        "max_seq_length": int(max_seq),
        "dtype": dtype if dtype != "None" else None,
        "load_in_4bit": load_4bit,
        "lora_r": int(lora_r),
        "lora_alpha": int(lora_alpha),
        "lora_dropout": float(lora_dropout),
        "target_modules": [m.strip() for m in target_modules.split(",")],
        "epochs": int(epochs),
        "per_device_train_batch_size": int(batch_size),
        "gradient_accumulation_steps": int(grad_accum),
        "learning_rate": float(lr),
        "warmup_steps": int(warmup_steps),
        "optim": optimizer,
        "lr_scheduler_type": scheduler,
        "seed": int(seed),
    }


def do_preview_hf(ds_name, subset, split):
    try:
        df, cols = load_hf_dataset(ds_name, subset or None, split)
        return (
            df,
            gr.Dropdown(choices=cols),
            gr.Dropdown(choices=cols),
            gr.Dropdown(choices=cols),
        )
    except Exception as e:
        return None, None, None, str(e)


def do_preview_upload(file_obj):
    try:
        path = save_uploaded_file(file_obj, "runs/uploads")
        df, cols = load_local_file(path)
        return (
            df,
            gr.Dropdown(choices=cols),
            gr.Dropdown(choices=cols),
            gr.Dropdown(choices=cols),
        )
    except Exception as e:
        return None, None, None, str(e)


def do_normalize_preview(
    ds_name,
    subset,
    split,
    upload_file,
    dataset_mode,
    col_format,
    col_user,
    col_assistant,
    col_system,
):
    try:
        if dataset_mode == "HF Dataset":
            df, _ = load_hf_dataset(ds_name, subset or None, split)
            records = df.to_dict("records")
        else:
            path = (
                upload_file
                if isinstance(upload_file, str)
                else save_uploaded_file(upload_file, "runs/uploads")
            )
            df, _ = load_local_file(path)
            records = df.to_dict("records")

        column_map = None
        if col_format == "csv_custom":
            column_map = {
                "user": col_user,
                "assistant": col_assistant,
                "system": col_system,
            }

        normalized = normalize(records, fmt=col_format, column_map=column_map)

        preview_data = []
        for rec in normalized[:20]:
            if rec.messages:
                preview_data.append(
                    {
                        "id": rec.id,
                        "messages": f"{len(rec.messages)} turns",
                        "preview": rec.messages[0].content[:100] + "..."
                        if rec.messages
                        else "",
                    }
                )
            else:
                preview_data.append(
                    {
                        "id": rec.id,
                        "messages": "completion",
                        "preview": rec.meta.get("text", "")[:100] + "...",
                    }
                )

        preview_df = pd.DataFrame(preview_data)
        return preview_df
    except Exception as e:
        return pd.DataFrame([{"error": str(e)}])


def do_render_preview(
    ds_name,
    subset,
    split,
    upload_file,
    dataset_mode,
    col_format,
    col_user,
    col_assistant,
    col_system,
    base_model,
    custom_model,
):
    try:
        if dataset_mode == "HF Dataset":
            df, _ = load_hf_dataset(ds_name, subset or None, split)
            records = df.to_dict("records")
        else:
            path = (
                upload_file
                if isinstance(upload_file, str)
                else save_uploaded_file(upload_file, "runs/uploads")
            )
            df, _ = load_local_file(path)
            records = df.to_dict("records")

        column_map = None
        if col_format == "csv_custom":
            column_map = {
                "user": col_user,
                "assistant": col_assistant,
                "system": col_system,
            }

        normalized = normalize(records, fmt=col_format, column_map=column_map)

        from unsloth import FastLanguageModel

        final_model = custom_model if custom_model else base_model

        _, tokenizer = FastLanguageModel.from_pretrained(
            model_name=final_model,
            max_seq_length=2048,
            dtype=None,
            load_in_4bit=True,
        )

        train_dataset = render_for_training(normalized, tokenizer)

        render_data = []
        for i, text in enumerate(train_dataset["text"][:5]):
            render_data.append(
                {"idx": i, "text": text[:500] + "..." if len(text) > 500 else text}
            )

        return pd.DataFrame(render_data)
    except Exception as e:
        return pd.DataFrame([{"error": str(e)}])


def do_train(*args):
    _cancel_flag.clear()
    cfg = build_cfg(*args[:18])
    dataset_spec = {
        "mode": args[19],
        "hf_dataset": args[20],
        "subset": args[21],
        "split": args[22],
        "upload_path": args[23],
        "format": args[24],
        "col_user": args[25],
        "col_assistant": args[26],
        "col_system": args[27],
    }
    try:
        import shutil

        total, used, free = shutil.disk_usage("/")
        free_gb = free // (2**30)
        if free_gb < 10:
            yield f"⚠️ Warning: Low disk space! Only {free_gb}GB free. Training may fail."

        logs = ""

        def on_log(line):
            nonlocal logs
            logs += line + "\n"

        run_dir = run_sft(cfg, dataset_spec, _cancel_flag, on_log)
        yield logs + f"\n✅ Training complete! Run dir: {run_dir}"
    except Exception:
        yield traceback.format_exc()


def do_cancel():
    _cancel_flag.set()
    return "⛔ Cancel requested. Stopping after current step..."


def do_eval(run_dir):
    try:
        metrics = run_eval({}, run_dir)
        return json.dumps(metrics, indent=2)
    except Exception:
        return traceback.format_exc()


def do_push_all(run_dir, hf_username, run_name, quant_methods_str):
    try:
        import shutil
        from src.paths import save_export_manifest

        total, used, free = shutil.disk_usage("/")
        free_gb = free // (2**30)
        if free_gb < 15:
            yield f"⚠️ Warning: Low disk space! Only {free_gb}GB free. Merging may fail."

        token = require_token()
        quant_methods = [q.strip() for q in quant_methods_str.split(",")]
        results = []

        yield "📤 Pushing LoRA adapter...\n"
        lora_repo = f"{hf_username}/{run_name}-lora"
        url = push_lora(run_dir, lora_repo, token)
        results.append(f"✅ Adapter: {url}")
        yield "\n".join(results) + "\n"

        yield "\n🔗 Merging + pushing merged model...\n"
        merged_repo = f"{hf_username}/{run_name}-merged"
        url = push_merged(run_dir, merged_repo, token)
        results.append(f"✅ Merged: {url}")
        yield "\n".join(results) + "\n"

        yield "\n📦 Quantizing GGUF + pushing...\n"
        gguf_repo = f"{hf_username}/{run_name}-GGUF"
        url = push_gguf(run_dir, gguf_repo, token, quant_methods)
        results.append(f"✅ GGUF: {url}")
        yield "\n".join(results) + "\n"

        export_manifest = {
            "lora_repo": lora_repo,
            "lora_url": results[0].split(": ")[1] if results else None,
            "merged_repo": merged_repo,
            "merged_url": results[1].split(": ")[1] if len(results) > 1 else None,
            "gguf_repo": gguf_repo,
            "gguf_url": results[2].split(": ")[1] if len(results) > 2 else None,
            "quant_methods": quant_methods,
        }
        save_export_manifest(run_dir, export_manifest)
        yield "\n✅ Export manifest saved."

    except Exception:
        yield traceback.format_exc()


def do_chat(message, history, run_dir, max_new_tokens):
    global _loaded_model, _loaded_tokenizer
    try:
        from unsloth import FastLanguageModel

        if _loaded_model is None:
            _loaded_model, _loaded_tokenizer = FastLanguageModel.from_pretrained(
                model_name=os.path.join(run_dir, "artifacts", "lora"),
                max_seq_length=2048,
                dtype=None,
                load_in_4bit=True,
            )
            FastLanguageModel.for_inference(_loaded_model)

        messages = []
        for h in history:
            messages.append({"role": "user", "content": h[0]})
            if h[1]:
                messages.append({"role": "assistant", "content": h[1]})
        messages.append({"role": "user", "content": message})

        inputs = _loaded_tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        ).to("cuda")

        from transformers import TextStreamer

        streamer = TextStreamer(_loaded_tokenizer, skip_prompt=True)

        import torch

        with torch.no_grad():
            output = _loaded_model.generate(
                input_ids=inputs,
                max_new_tokens=int(max_new_tokens),
                streamer=streamer,
                use_cache=True,
            )
        decoded = _loaded_tokenizer.decode(
            output[0][inputs.shape[1] :], skip_special_tokens=True
        )
        return decoded
    except Exception:
        return traceback.format_exc()


def get_unsloth_models():
    """Get list of supported models from Unsloth."""
    return [
        # Llama 3.1
        "unsloth/llama-3.1-8b-unsloth-bnb-4bit",
        "unsloth/llama-3.1-70b-unsloth-bnb-4bit",
        "unsloth/llama-3.1-405b-unsloth-bnb-4bit",
        # Llama 3
        "unsloth/llama-3-8b-bnb-4bit",
        "unsloth/llama-3-70b-bnb-4bit",
        # Qwen 2.5
        "unsloth/Qwen2.5-0.5B-Instruct-bnb-4bit",
        "unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit",
        "unsloth/Qwen2.5-3B-Instruct-bnb-4bit",
        "unsloth/Qwen2.5-7B-Instruct-bnb-4bit",
        "unsloth/Qwen2.5-14B-Instruct-bnb-4bit",
        "unsloth/Qwen2.5-32B-Instruct-bnb-4bit",
        "unsloth/Qwen2.5-72B-Instruct-bnb-4bit",
        # Mistral
        "unsloth/mistral-7b-instruct-v0.3-bnb-4bit",
        "unsloth/mistral-7b-v0.3-bnb-4bit",
        # Gemma 2
        "unsloth/gemma-2-2b-bnb-4bit",
        "unsloth/gemma-2-9b-bnb-4bit",
        "unsloth/gemma-2-27b-bnb-4bit",
        # Phi-3
        "unsloth/Phi-3.5-mini-instruct-bnb-4bit",
        "unsloth/Phi-3-medium-4k-instruct-bnb-4bit",
        # Phi-4
        "unsloth/Phi-4-mini-instruct-bnb-4bit",
        "unsloth/Phi-4",
        # Yi
        "unsloth/Yi-1.5-6B-Chat-bnb-4bit",
        "unsloth/Yi-1.5-9B-Chat-bnb-4bit",
        "unsloth/Yi-1.5-34B-Chat-bnb-4bit",
        # DeepSeek
        "unsloth/DeepSeek-Coder-V2-Instruct-bnb-4bit",
        "unsloth/DeepSeek-V2-Chat-bnb-4bit",
        # Command-R
        "unsloth/Command-R7B-8k-bnb-4bit",
        # Falcon
        "unsloth/falcon-7b-instruct-bnb-4bit",
        # Starling
        "unsloth/Starling-LM-7B-alpha-bnb-4bit",
        # OLMo
        "unsloth/OLMo-7B-Instruct-bnb-4bit",
    ]


def get_model_families():
    """Group models by family for dropdown."""
    families = {
        # Llama
        "Llama 4": [
            "unsloth/Llama-4-Scout-17B-16E-Instruct",
            "unsloth/Llama-4-Maverick-17B-128E-Instruct",
        ],
        "Llama 3.3": [
            "unsloth/Llama-3.3-70B-Instruct",
        ],
        "Llama 3.2": [
            "unsloth/Llama-3.2-1B-Instruct",
            "unsloth/Llama-3.2-3B-Instruct",
            "unsloth/Llama-3.2-11B-Vision-Instruct",
            "unsloth/Llama-3.2-90B-Vision-Instruct",
        ],
        "Llama 3.1": [
            "unsloth/Meta-Llama-3.1-8B-Instruct",
            "unsloth/Meta-Llama-3.1-70B-Instruct",
            "unsloth/Meta-Llama-3.1-405B-Instruct",
        ],
        "Llama 3": [
            "unsloth/llama-3-8b-Instruct",
            "unsloth/llama-3-70b-bnb-4bit",
        ],
        "Llama 2": [
            "unsloth/llama-2-7b-chat-bnb-4bit",
            "unsloth/llama-2-13b-bnb-4bit",
        ],
        "CodeLlama": [
            "unsloth/codellama-7b-bnb-4bit",
            "unsloth/codellama-13b-bnb-4bit",
            "unsloth/codellama-34b-bnb-4bit",
        ],
        # Qwen
        "Qwen 3": [
            "unsloth/Qwen3-0.6B",
            "unsloth/Qwen3-1.7B",
            "unsloth/Qwen3-4B",
            "unsloth/Qwen3-8B",
            "unsloth/Qwen3-14B",
            "unsloth/Qwen3-30B-A3B",
            "unsloth/Qwen3-32B",
        ],
        "Qwen 3 VL": [
            "unsloth/Qwen3-VL-2B-Instruct",
            "unsloth/Qwen3-VL-4B-Instruct",
            "unsloth/Qwen3-VL-8B-Instruct",
            "unsloth/Qwen3-VL-32B-Instruct",
        ],
        "Qwen 2.5": [
            "unsloth/Qwen2.5-0.5B-Instruct",
            "unsloth/Qwen2.5-1.5B-Instruct",
            "unsloth/Qwen2.5-3B-Instruct",
            "unsloth/Qwen2.5-7B-Instruct",
            "unsloth/Qwen2.5-14B-Instruct",
            "unsloth/Qwen2.5-32B-Instruct",
            "unsloth/Qwen2.5-72B-Instruct",
        ],
        "Qwen 2.5 VL": [
            "unsloth/Qwen2.5-VL-3B-Instruct",
            "unsloth/Qwen2.5-VL-7B-Instruct",
            "unsloth/Qwen2.5-VL-32B-Instruct",
            "unsloth/Qwen2.5-VL-72B-Instruct",
        ],
        "Qwen 2.5 Coder": [
            "unsloth/Qwen2.5-Coder-0.5B-Instruct",
            "unsloth/Qwen2.5-Coder-1.5B-Instruct",
            "unsloth/Qwen2.5-Coder-3B-Instruct",
            "unsloth/Qwen2.5-Coder-7B-Instruct",
            "unsloth/Qwen2.5-Coder-14B-Instruct",
        ],
        # Mistral
        "Mistral": [
            "unsloth/mistral-7b-instruct-v0.3-bnb-4bit",
            "unsloth/mistral-7b-v0.3-bnb-4bit",
        ],
        "Mistral 3": [
            "unsloth/Ministral-3-3B-Instruct",
            "unsloth/Ministral-3-8B-Instruct",
            "unsloth/Ministral-3-14B-Instruct",
        ],
        "Mistral Large": [
            "unsloth/Mistral-Large-3-675B-Instruct",
        ],
        # Gemma
        "Gemma 3": [
            "unsloth/gemma-3-270m-it",
            "unsloth/gemma-3-1b-it",
            "unsloth/gemma-3-4b-it",
            "unsloth/gemma-3-12b-it",
            "unsloth/gemma-3-27b-it",
        ],
        "Gemma 3n": [
            "unsloth/gemma-3n-E2B-it",
            "unsloth/gemma-3n-E4B-it",
        ],
        "Gemma 2": [
            "unsloth/gemma-2-2b-it-bnb-4bit",
            "unsloth/gemma-2-9b-it-bnb-4bit",
            "unsloth/gemma-2-27b-it-bnb-4bit",
        ],
        "CodeGemma": [
            "unsloth/CodeGemma-7b-bnb-4bit",
        ],
        # Phi
        "Phi": [
            "unsloth/Phi-3.5-mini-instruct-bnb-4bit",
            "unsloth/Phi-3-medium-4k-instruct-bnb-4bit",
            "unsloth/Phi-4-mini-instruct-bnb-4bit",
            "unsloth/Phi-4",
        ],
        # DeepSeek
        "DeepSeek R1": [
            "unsloth/DeepSeek-R1-Distill-Llama-8B",
            "unsloth/DeepSeek-R1-Distill-Llama-70B",
            "unsloth/DeepSeek-R1-Distill-Qwen-1.5B",
            "unsloth/DeepSeek-R1-Distill-Qwen-7B",
            "unsloth/DeepSeek-R1-Distill-Qwen-14B",
            "unsloth/DeepSeek-R1-Distill-Qwen-32B",
            "unsloth/DeepSeek-R1-0528-Qwen3-8B",
        ],
        "DeepSeek V3": [
            "unsloth/DeepSeek-V3",
        ],
        "DeepSeek Coder": [
            "unsloth/DeepSeek-Coder-V2-Instruct-bnb-4bit",
            "unsloth/DeepSeek-V2-Chat-bnb-4bit",
        ],
        # Yi
        "Yi": [
            "unsloth/Yi-1.5-6B-Chat-bnb-4bit",
            "unsloth/Yi-1.5-9B-Chat-bnb-4bit",
            "unsloth/Yi-1.5-34B-Chat-bnb-4bit",
        ],
        # Granite
        "Granite": [
            "unsloth/granite-4.0-h-small",
        ],
        # gpt-oss
        "gpt-oss": [
            "unsloth/gpt-oss-20b-unsloth-bnb-4bit",
            "unsloth/gpt-oss-120b-unsloth-bnb-4bit",
        ],
        # Other
        "Other": [
            "unsloth/Command-R7B-8k-bnb-4bit",
            "unsloth/falcon-7b-instruct-bnb-4bit",
            "unsloth/Starling-LM-7B-alpha-bnb-4bit",
            "unsloth/OLMo-7B-Instruct-bnb-4bit",
            "liquid/liquid-1.6-7b-chat",
            "liquid/liquid-1.6-8x7b-chat",
            "liquid/liquid-1.6-70b-chat",
        ],
    }
    return families


def get_all_model_choices():
    """Get flattened list of all models."""
    families = get_model_families()
    all_models = []
    for family_models in families.values():
        all_models.extend(family_models)
    return all_models


with gr.Blocks(title="Unsloth All-in-One Fine-Tuner", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🦥 Unsloth All-in-One Fine-Tuner")
    gr.Markdown("End-to-end: Dataset → Normalize → Train → Eval → Merge → Push → GGUF")

    run_dir_state = gr.State(None)

    with gr.Tab("📋 Project"):
        with gr.Row():
            hf_username = gr.Textbox(
                label="HF Username", placeholder="your-hf-username"
            )
            run_name = gr.Textbox(
                label="Run Name", placeholder="llama31-8b-mydata-20260220"
            )
        base_model_family = gr.Dropdown(
            choices=list(get_model_families().keys()),
            value="Llama 4",
            label="Model Family",
        )
        base_model = gr.Dropdown(
            choices=get_model_families()["Llama 4"],
            value="unsloth/Llama-4-Scout-17B-16E-Instruct",
            label="Base Model",
        )

        def update_models(family):
            models = get_model_families().get(family, [])
            default_value = models[0] if models else ""
            return gr.update(choices=models, value=default_value)

        base_model_family.change(update_models, base_model_family, base_model)

        gr.Markdown("Or enter custom model from Hugging Face:")
        custom_model = gr.Textbox(
            label="Custom Model (overrides above)",
            placeholder="e.g., meta-llama/Llama-3.1-8b",
        )

        def get_selected_model(family_model, custom):
            return custom if custom else family_model

        with gr.Row():
            max_seq = gr.Number(value=2048, label="Max Seq Length")
            dtype = gr.Dropdown(
                choices=["None", "float16", "bfloat16"], value="None", label="Dtype"
            )
            load_4bit = gr.Checkbox(value=True, label="Load in 4-bit")
        gr.Markdown("### LoRA Config")
        with gr.Row():
            lora_r = gr.Slider(4, 128, value=16, step=4, label="LoRA Rank (r)")
            lora_alpha = gr.Slider(4, 128, value=16, step=4, label="LoRA Alpha")
            lora_dropout = gr.Slider(
                0.0, 0.2, value=0.0, step=0.01, label="LoRA Dropout"
            )
        target_modules = gr.Textbox(
            value="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
            label="Target Modules (comma-separated)",
        )
        gr.Markdown("### Training Config")
        with gr.Row():
            epochs = gr.Number(value=3, label="Epochs")
            batch_size = gr.Number(value=2, label="Batch Size per Device")
            grad_accum = gr.Number(value=4, label="Gradient Accumulation")
        with gr.Row():
            lr = gr.Number(value=2e-4, label="Learning Rate")
            warmup_steps = gr.Number(value=5, label="Warmup Steps")
            seed = gr.Number(value=42, label="Seed")
        with gr.Row():
            optimizer = gr.Dropdown(
                choices=["adamw_8bit", "adamw_torch", "lion_8bit"],
                value="adamw_8bit",
                label="Optimizer",
            )
            scheduler = gr.Dropdown(
                choices=["linear", "cosine", "constant"],
                value="linear",
                label="LR Scheduler",
            )

    with gr.Tab("📦 Dataset"):
        dataset_mode = gr.Radio(
            ["HF Dataset", "Upload File"], value="HF Dataset", label="Dataset Source"
        )
        with gr.Group(visible=True) as hf_group:
            with gr.Row():
                hf_ds_name = gr.Textbox(
                    label="HF Dataset Name", placeholder="owner/dataset-name"
                )
                hf_subset = gr.Textbox(label="Subset (optional)")
                hf_split = gr.Textbox(label="Split", value="train")
            preview_hf_btn = gr.Button("Preview Dataset", variant="secondary")
            normalize_btn = gr.Button("Normalize & Preview", variant="secondary")
            render_btn = gr.Button("Render Training Text", variant="secondary")

        with gr.Group(visible=False) as upload_group:
            upload_file = gr.File(label="Upload (.json / .jsonl / .csv)")
            preview_upload_btn = gr.Button("Preview Uploaded File", variant="secondary")

        dataset_mode.change(
            lambda m: (
                gr.update(visible=m == "HF Dataset"),
                gr.update(visible=m == "Upload File"),
            ),
            [dataset_mode],
            [hf_group, upload_group],
        )

        preview_df = gr.Dataframe(label="Dataset Preview (first 20 rows)")
        gr.Markdown("### Column Mapping")
        with gr.Row():
            col_format = gr.Dropdown(
                choices=[
                    "auto",
                    "sharegpt",
                    "messages",
                    "alpaca",
                    "csv_custom",
                    "completion",
                ],
                value="auto",
                label="Dataset Format",
            )
            col_user = gr.Dropdown(label="User column (csv_custom)")
            col_assistant = gr.Dropdown(label="Assistant column (csv_custom)")
            col_system = gr.Dropdown(label="System column (csv_custom, optional)")

        preview_hf_btn.click(
            do_preview_hf,
            [hf_ds_name, hf_subset, hf_split],
            [preview_df, col_user, col_assistant, col_system],
            concurrency_limit=4,
        )
        preview_upload_btn.click(
            do_preview_upload,
            [upload_file],
            [preview_df, col_user, col_assistant, col_system],
            concurrency_limit=4,
        )

        gr.Markdown("### Normalized Preview")
        normalize_preview_df = gr.Dataframe(label="Normalized Records")
        normalize_btn.click(
            do_normalize_preview,
            [
                hf_ds_name,
                hf_subset,
                hf_split,
                upload_file,
                dataset_mode,
                col_format,
                col_user,
                col_assistant,
                col_system,
            ],
            [normalize_preview_df],
            concurrency_limit=4,
        )

        gr.Markdown("### Rendered Training Text")
        render_preview_df = gr.Dataframe(label="Rendered Text (first 5)")
        render_btn.click(
            do_render_preview,
            [
                hf_ds_name,
                hf_subset,
                hf_split,
                upload_file,
                dataset_mode,
                col_format,
                col_user,
                col_assistant,
                col_system,
                base_model,
                custom_model,
            ],
            [render_preview_df],
            concurrency_limit=4,
        )

    with gr.Tab("🚀 Train"):
        with gr.Row():
            train_btn = gr.Button("▶ Start Training", variant="primary")
            cancel_btn = gr.Button("⛔ Cancel", variant="stop")
        train_logs = gr.Textbox(
            label="Training Logs",
            lines=24,
            max_lines=200,
            autoscroll=True,
            interactive=False,
        )
        cancel_status = gr.Textbox(label="Cancel Status", interactive=False)

        all_cfg_inputs = [
            hf_username,
            run_name,
            base_model,
            custom_model,
            max_seq,
            dtype,
            load_4bit,
            lora_r,
            lora_alpha,
            lora_dropout,
            target_modules,
            epochs,
            batch_size,
            grad_accum,
            lr,
            warmup_steps,
            optimizer,
            scheduler,
            seed,
            dataset_mode,
            hf_ds_name,
            hf_subset,
            hf_split,
            upload_file,
            col_format,
            col_user,
            col_assistant,
            col_system,
        ]

        train_btn.click(
            do_train, inputs=all_cfg_inputs, outputs=[train_logs], concurrency_limit=1
        )
        cancel_btn.click(do_cancel, outputs=[cancel_status])

    with gr.Tab("📊 Evaluate / Test"):
        eval_run_dir = gr.Textbox(
            label="Run Directory", placeholder="runs/my-run-20260220"
        )
        with gr.Row():
            eval_btn = gr.Button("Run Evaluation", variant="secondary")
            prompt_suite_btn = gr.Button("Run Prompt Suite", variant="secondary")
        eval_out = gr.Textbox(label="Metrics", lines=10, interactive=False)
        prompt_suite_out = gr.Dataframe(label="Prompt Suite Results")

        eval_btn.click(do_eval, [eval_run_dir], [eval_out], concurrency_limit=1)

        def do_run_prompt_suite(run_dir):
            try:
                from src.eval import run_prompt_suite
                from unsloth import FastLanguageModel
                import torch

                adapter_path = os.path.join(run_dir, "artifacts", "lora")
                model, tokenizer = FastLanguageModel.from_pretrained(
                    model_name=adapter_path,
                    max_seq_length=2048,
                    dtype=None,
                    load_in_4bit=True,
                )

                results = run_prompt_suite(model, tokenizer, run_dir)

                df_data = []
                for r in results:
                    df_data.append(
                        {
                            "prompt": r["prompt"][:50] + "..."
                            if len(r["prompt"]) > 50
                            else r["prompt"],
                            "expected": r.get("expected", ""),
                            "actual": r["actual"][:100] + "..."
                            if len(r["actual"]) > 100
                            else r["actual"],
                        }
                    )
                import pandas as pd

                return pd.DataFrame(df_data)
            except Exception as e:
                import pandas as pd

                return pd.DataFrame([{"error": str(e)}])

        prompt_suite_btn.click(
            do_run_prompt_suite, [eval_run_dir], [prompt_suite_out], concurrency_limit=1
        )

    with gr.Tab("📤 Export"):
        with gr.Row():
            export_run_dir = gr.Textbox(label="Run Directory")
            export_hf_username = gr.Textbox(label="HF Username")
            export_run_name = gr.Textbox(label="Run Name")
        quant_methods = gr.Textbox(
            label="GGUF Quant Methods (comma-separated)", value="q4_k_m,q8_0"
        )
        with gr.Row():
            push_btn = gr.Button(
                "🚀 Push All (Adapter + Merged + GGUF)", variant="primary"
            )
            export_local_btn = gr.Button("💾 Export GGUF Locally", variant="secondary")
        push_logs = gr.Textbox(label="Push Logs", lines=12, interactive=False)
        local_export_logs = gr.Textbox(
            label="Local Export Logs", lines=8, interactive=False
        )

        push_btn.click(
            do_push_all,
            [export_run_dir, export_hf_username, export_run_name, quant_methods],
            [push_logs],
            concurrency_limit=1,
        )

        def do_export_gguf_local(run_dir, quant_method):
            try:
                from src.export import export_gguf_local
                from unsloth import FastLanguageModel

                adapter_path = os.path.join(run_dir, "artifacts", "lora")
                model, tokenizer = FastLanguageModel.from_pretrained(
                    model_name=adapter_path,
                    max_seq_length=2048,
                    dtype=None,
                    load_in_4bit=True,
                )

                out_path = export_gguf_local(run_dir, model, tokenizer, quant_method)
                return f"✅ GGUF exported to: {out_path}"
            except Exception:
                return traceback.format_exc()

        export_local_btn.click(
            do_export_gguf_local,
            [export_run_dir, quant_methods],
            [local_export_logs],
            concurrency_limit=1,
        )

    with gr.Tab("💬 Chat"):
        chat_run_dir = gr.Textbox(
            label="Run Directory (with adapter)", placeholder="runs/my-run-20260220"
        )
        chat_max_tokens = gr.Slider(64, 1024, value=256, label="Max New Tokens")
        chatbot = gr.ChatInterface(
            fn=do_chat,
            additional_inputs=[chat_run_dir, chat_max_tokens],
        )

demo.queue(default_concurrency_limit=1, max_size=20).launch(
    server_name="0.0.0.0",
    server_port=7860,
    share=False,
)

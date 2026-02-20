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
    return {
        "hf_username": hf_username,
        "run_name": run_name,
        "base_model": base_model,
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

        _, tokenizer = FastLanguageModel.from_pretrained(
            model_name=base_model,
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
        "mode": args[18],
        "hf_dataset": args[19],
        "subset": args[20],
        "split": args[21],
        "upload_path": args[22],
        "format": args[23],
        "col_user": args[24],
        "col_assistant": args[25],
        "col_system": args[26],
    }
    try:
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


MODELS = [
    "unsloth/llama-3.1-8b-unsloth-bnb-4bit",
    "unsloth/llama-3.1-70b-bnb-4bit",
    "unsloth/Qwen2.5-7B-Instruct-bnb-4bit",
    "unsloth/mistral-7b-instruct-v0.3-bnb-4bit",
    "unsloth/gemma-2-9b-bnb-4bit",
    "unsloth/Phi-4",
]

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
        base_model = gr.Dropdown(choices=MODELS, value=MODELS[0], label="Base Model")
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
        eval_btn = gr.Button("Run Evaluation", variant="secondary")
        eval_out = gr.Textbox(label="Metrics", lines=10, interactive=False)
        eval_btn.click(do_eval, [eval_run_dir], [eval_out], concurrency_limit=1)

    with gr.Tab("📤 Export"):
        with gr.Row():
            export_run_dir = gr.Textbox(label="Run Directory")
            export_hf_username = gr.Textbox(label="HF Username")
            export_run_name = gr.Textbox(label="Run Name")
        quant_methods = gr.Textbox(
            label="GGUF Quant Methods (comma-separated)", value="q4_k_m,q8_0"
        )
        push_btn = gr.Button("🚀 Push All (Adapter + Merged + GGUF)", variant="primary")
        push_logs = gr.Textbox(label="Push Logs", lines=12, interactive=False)

        push_btn.click(
            do_push_all,
            [export_run_dir, export_hf_username, export_run_name, quant_methods],
            [push_logs],
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

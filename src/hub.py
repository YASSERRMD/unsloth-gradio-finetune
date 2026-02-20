import os


def push_lora(run_dir: str, repo_id: str, token: str) -> str:
    from huggingface_hub import HfApi

    adapter_path = os.path.join(run_dir, "artifacts", "lora")
    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, exist_ok=True)
    api.upload_folder(folder_path=adapter_path, repo_id=repo_id)
    return f"https://huggingface.co/{repo_id}"


def push_merged(run_dir: str, repo_id: str, token: str) -> str:
    from unsloth import FastLanguageModel

    adapter_path = os.path.join(run_dir, "artifacts", "lora")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=adapter_path, max_seq_length=2048, dtype=None, load_in_4bit=True
    )
    model.push_to_hub_merged(
        repo_id, tokenizer, save_method="merged_16bit", token=token
    )
    return f"https://huggingface.co/{repo_id}"


def push_gguf(run_dir: str, repo_id: str, token: str, quant_methods: list) -> str:
    from unsloth import FastLanguageModel

    adapter_path = os.path.join(run_dir, "artifacts", "lora")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=adapter_path, max_seq_length=2048, dtype=None, load_in_4bit=True
    )
    model.push_to_hub_gguf(
        repo_id, tokenizer, quantization_method=quant_methods, token=token
    )
    return f"https://huggingface.co/{repo_id}"

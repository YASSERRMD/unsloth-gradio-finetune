import os


def merge_to_local(run_dir: str, model, tokenizer) -> str:
    out = os.path.join(run_dir, "artifacts", "merged")
    os.makedirs(out, exist_ok=True)
    model.save_pretrained_merged(out, tokenizer, save_method="merged_16bit")
    return out


def export_gguf_local(
    run_dir: str, model, tokenizer, quant_method: str = "q4_k_m"
) -> str:
    out = os.path.join(run_dir, "artifacts", "gguf")
    os.makedirs(out, exist_ok=True)
    model.save_pretrained_gguf(out, tokenizer, quantization_method=quant_method)
    return out

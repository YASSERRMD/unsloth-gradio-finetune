# 🦥 Unsloth All-in-One Fine-Tuner

A complete Gradio-based web application for fine-tuning LLMs using [Unsloth](https://github.com/unslothai/unsloth).

## Features

- **Dataset Management** — Load from Hugging Face Hub or upload JSON/JSONL/CSV files
- **Format Normalization** — Auto-detect and convert ShareGPT, Alpaca, messages, and custom CSV formats
- **Chat Template Rendering** — Preview how data looks with model-specific chat templates
- **LoRA/QLoRA Training** — Fine-tune with Unsloth's optimized training pipeline
- **Evaluation** — Validation loss/perplexity + behavioral prompt suite testing
- **Interactive Chat** — Test your fine-tuned model directly in the UI
- **Export Options** — Push to Hugging Face Hub (adapter, merged, GGUF) or save locally

## Getting Started

### Installation

```bash
pip install -r requirements.txt
```

### Running the App

```bash
python app.py
```

The app will be available at `http://localhost:7860`

### Environment Variables

Set your Hugging Face token for model uploads:

```bash
export HF_TOKEN=your_token_here
```

## Supported Models

The app supports 100+ models from Unsloth's model catalog. Select by family in the UI:

### Llama
- **Llama 4**: Scout 17B, Maverick 17B
- **Llama 3.3**: 70B
- **Llama 3.2**: 1B, 3B, 11B Vision, 90B Vision
- **Llama 3.1**: 8B, 70B, 405B
- **Llama 3**: 8B, 70B
- **Llama 2**: 7B, 13B
- **CodeLlama**: 7B, 13B, 34B

### Qwen
- **Qwen 3**: 0.6B, 1.7B, 4B, 8B, 14B, 30B-A3B, 32B
- **Qwen 3 VL**: 2B, 4B, 8B, 32B
- **Qwen 2.5**: 0.5B to 72B
- **Qwen 2.5 VL**: 3B, 7B, 32B, 72B
- **Qwen 2.5 Coder**: 0.5B to 14B

### Mistral
- **Mistral**: 7B v0.3
- **Mistral 3 (Ministral)**: 3B, 8B, 14B
- **Mistral Large**: 675B

### Gemma
- **Gemma 3**: 270M, 1B, 4B, 12B, 27B
- **Gemma 3n**: E2B, E4B
- **Gemma 2**: 2B, 9B, 27B
- **CodeGemma**: 7B
- **FunctionGemma**: 270M
- **MedGemma**: 4B, 27B

### Phi
- **Phi-3.5**: Mini
- **Phi-3**: Medium
- **Phi-4**: Mini, 14B

### DeepSeek
- **DeepSeek R1**: Distill versions (Llama 8B, 70B / Qwen 1.5B to 32B)
- **DeepSeek V3**: V3, V3-0324
- **DeepSeek Coder**: V2

### Yi
- **Yi 1.5**: 6B, 9B, 34B

### Other Models
- **Granite**: 4.0 H-Small
- **gpt-oss**: 20B, 120B
- **Command-R**: 7B
- **Falcon**: 7B
- **Starling**: 7B
- **OLMo**: 7B
- **Liquid**: 1.6 series (7B, 8x7B, 70B)

### Custom Models
You can also enter any Hugging Face model directly in the "Custom Model" field.

## Supported Dataset Formats

| Format | Description |
|--------|-------------|
| `sharegpt` | `{"conversations": [{"from": "human", "value": "..."}]}` |
| `messages` | `{"messages": [{"role": "user", "content": "..."}]}` |
| `alpaca` | `{"instruction": "...", "input": "...", "output": "..."}` |
| `csv_custom` | User-defined column mapping |
| `completion` | Raw text without chat template |

## Credit

This tool is built on top of **Unsloth** — the fastest fine-tuning library for LLMs. Unsloth provides:

- 2x faster fine-tuning
- 70% less VRAM usage
- No quality degradation
- Support for 100+ models including Llama, Mistral, Qwen, Gemma, Phi, DeepSeek, and more

Visit [unsloth.ai](https://unsloth.ai) and the [Unsloth Model Catalog](https://unsloth.ai/docs/get-started/unsloth-model-catalog) for more information.

## License

MIT License — See LICENSE file for details.

---

Built with ❤️ using Gradio, Hugging Face, and [Unsloth](https://github.com/unslothai/unsloth)

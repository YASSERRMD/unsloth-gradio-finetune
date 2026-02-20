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

- `unsloth/llama-3.1-8b-unsloth-bnb-4bit`
- `unsloth/llama-3.1-70b-bnb-4bit`
- `unsloth/Qwen2.5-7B-Instruct-bnb-4bit`
- `unsloth/mistral-7b-instruct-v0.3-bnb-4bit`
- `unsloth/gemma-2-9b-bnb-4bit`
- `unsloth/Phi-4`

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
- 30% less memory usage
- No quality degradation
- Support for Llama, Mistral, Qwen, Gemma, and more

Visit [unsloth.ai](https://unsloth.ai) for more information.

## License

MIT License — See LICENSE file for details.

---

Built with ❤️ using Gradio, Hugging Face, and Unsloth

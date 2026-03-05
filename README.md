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

The UI model picker is aligned to the naming used in the official [Unsloth notebooks](https://github.com/unslothai/notebooks), including 4-bit variants where notebooks use them (for example `...-bnb-4bit`).

Model families currently included in the dropdown:
- Llama (3, 3.1, 3.2 Vision, 3.3, 4)
- Qwen (Qwen3, Qwen3-VL, Qwen2.5, Qwen2.5-VL, Qwen2.5-Coder)
- Mistral (Mistral 7B, Ministral 3, Mistral Large)
- Gemma (Gemma 2, Gemma 3, Gemma 3n, CodeGemma)
- Phi
- DeepSeek
- Yi
- Granite
- gpt-oss
- Other (Command-R, Falcon, Starling, OLMo, Liquid)

You can also enter any Hugging Face model directly in the **Custom Model** field. The app resolves common legacy aliases to notebook-compatible model IDs automatically during preview and training.

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

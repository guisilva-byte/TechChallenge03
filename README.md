# LLaMA 3 Fine-Tuning on Amazon Product Titles

Fine-tuning **LLaMA 3 8B** with **QLoRA (4-bit)** on the [AmazonTitles-1.3MM](https://huggingface.co/datasets/dirtmaxim/AmazonTitles-1.3MM) dataset to generate product descriptions from titles — enabling a Q&A interface over Amazon product data.

> Academic project — FIAP Pós-Tech, Machine Learning Engineering (Tech Challenge Phase 3)

🎥 [Demo video](https://youtu.be/3fdYa4S5nPc)

---

## Overview

| Item | Detail |
|---|---|
| Base Model | `unsloth/llama-3-8b-bnb-4bit` |
| Fine-tuning Method | QLoRA via [Unsloth](https://github.com/unslothai/unsloth) |
| Dataset | AmazonTitles-1.3MM (`trn.json`) — 2.2M records |
| Task | Instruction-following: title → description generation |
| Hardware | NVIDIA L4 (Google Colab) |
| Training Steps | 60 steps (proof of concept) |
| LoRA Rank | 16 |

---

## Problem Statement

Given a user question containing an Amazon product title, the fine-tuned model generates a response grounded in the product description learned during training — simulating a product Q&A system powered by fine-tuned LLM knowledge.

**Example:**

```
Question: Girls Ballet Tutu Neon Pink - High quality 3 layer ballet tutu. 12 inches in length
Answer:   This is a high-quality 3-layer neon pink ballet tutu, 12 inches long, perfect for ...
```

---

## Project Structure

```
.
├── main.py            # Full pipeline: data loading → training → inference
├── requirements.txt   # Python dependencies
└── README.md
```

---

## Dataset

**AmazonTitles-1.3MM** — real user search queries paired with relevant Amazon product titles and descriptions.

- Source file: `trn.json` (JSONL, ~2.2M records)
- Fields used: `title`, `content`
- Training prompt format:

```
Please answer the question to the best of your ability.

Question:
<title> - <content>

Answer:
```

Download the dataset and place `trn.json.gz` in your Google Drive before running.

---

## Setup

### Google Colab (recommended — GPU required)

```python
# 1. Mount Drive and place the dataset at the configured path

# 2. Install dependencies
!pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
!pip install --no-deps xformers "trl<0.9.0" peft accelerate bitsandbytes
!pip install transformers datasets

# 3. Run the pipeline
!python main.py
```

### Local (conda)

> Note: local execution requires a CUDA-capable GPU with ≥ 16 GB VRAM.

```bash
conda create -n llm-amazon-finetuning python=3.10 -y
conda activate llm-amazon-finetuning
pip install -r requirements.txt
```

Update `DATASET_PATH` and `UNCOMPRESSED_PATH` in `main.py` to point to your local files, then:

```bash
python main.py
```

---

## Pipeline

```
1. Mount Google Drive
        ↓
2. Decompress trn.json.gz → trn.json
        ↓
3. Load 2.2M records, merge title + content, train/test split (80/20)
        ↓
4. Load LLaMA 3 8B (4-bit) + attach LoRA adapters
        ↓
5. Baseline inference (pre fine-tuning)
        ↓
6. SFT training via Unsloth + TRL (60 steps)
        ↓
7. Save fine-tuned model
        ↓
8. Fine-tuned inference — compare responses
```

---

## Key Hyperparameters

| Parameter | Value |
|---|---|
| LoRA rank (`r`) | 16 |
| LoRA alpha | 16 |
| Dropout | 0 |
| Batch size | 1 |
| Gradient accumulation | 4 (effective batch = 4) |
| Learning rate | 2e-4 |
| LR scheduler | Linear |
| Optimizer | AdamW 8-bit |
| Max steps | 60 |
| Warmup steps | 5 |
| Max sequence length | 2048 |
| Quantization | 4-bit NF4 (bitsandbytes) |

---

## Results

The model was evaluated qualitatively by comparing baseline vs. fine-tuned responses on 20 held-out test samples. After 60 training steps, the fine-tuned model showed improved ability to generate product-specific descriptions consistent with the Amazon dataset context.

Training loss converged to **~2.14** over 60 steps on 1.8M training examples.

---

## Technologies

- [Unsloth](https://github.com/unslothai/unsloth) — 2× faster QLoRA fine-tuning
- [HuggingFace Transformers](https://github.com/huggingface/transformers)
- [PEFT](https://github.com/huggingface/peft) — LoRA adapters
- [TRL](https://github.com/huggingface/trl) — SFTTrainer
- [bitsandbytes](https://github.com/TimDettmers/bitsandbytes) — 4-bit quantization
- PyTorch 2.4 + CUDA 12.1

---

## License

This project is for academic purposes only. The AmazonTitles-1.3MM dataset and LLaMA 3 model are subject to their respective licenses.

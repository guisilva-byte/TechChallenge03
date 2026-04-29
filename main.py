"""
Fine-tuning LLaMA 3 8B on the AmazonTitles-1.3MM dataset using LoRA (QLoRA 4-bit).

The model learns to generate product descriptions from product titles,
enabling it to answer user questions about Amazon products based on the
training data context.

Designed to run on Google Colab with a GPU (L4 / A100 recommended).
"""

import os
import gzip
import json
import torch

from datasets import Dataset
from transformers import TrainingArguments

# Unsloth is only available on Linux + CUDA (e.g. Google Colab).
# Importing it on Windows or CPU-only machines will raise an ImportError.
try:
    from unsloth import FastLanguageModel, is_bfloat16_supported
    from trl import SFTTrainer
    UNSLOTH_AVAILABLE = True
except ImportError:
    UNSLOTH_AVAILABLE = False
    print("Warning: unsloth not available — training and LoRA features disabled.")

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────

DATASET_PATH = "/content/drive/MyDrive/PosTech - Gui/Tech Challenge - Fase 3/LF-Amazon-1.3M/trn.json.gz"
UNCOMPRESSED_PATH = "/content/drive/MyDrive/PosTech - Gui/Tech Challenge - Fase 3/LF-Amazon-1.3M/trn.json"
OUTPUT_DIR = "outputs"
MODEL_SAVE_DIR = "fine_tuned_model"

MODEL_NAME = "unsloth/llama-3-8b-bnb-4bit"
LOAD_IN_4BIT = True
MAX_SEQ_LENGTH = 2048

# LoRA hyperparameters
LORA_RANK = 16
LORA_ALPHA = 16
LORA_DROPOUT = 0
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

# Training hyperparameters
BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 4
WARMUP_STEPS = 5
MAX_STEPS = 60
LEARNING_RATE = 2e-4
SEED = 3407
TEST_SPLIT = 0.2
INFERENCE_SAMPLES = 20
MAX_NEW_TOKENS = 150

INSTRUCTION = "Please answer the question to the best of your ability."


# ─────────────────────────────────────────────
# Step 1: Mount Google Drive (Colab only)
# ─────────────────────────────────────────────

def mount_drive():
    try:
        from google.colab import drive
        drive.mount("/content/drive")
        print("Google Drive mounted.")
    except ImportError:
        print("Not running on Colab — skipping Drive mount.")


# ─────────────────────────────────────────────
# Step 2: Dataset loading and preprocessing
# ─────────────────────────────────────────────

def decompress_dataset(src: str, dst: str) -> None:
    """Decompress the gzipped JSON dataset if not already extracted."""
    if os.path.exists(dst):
        print(f"Dataset already decompressed at {dst}")
        return
    print("Decompressing dataset...")
    with gzip.open(src, "rb") as f_in, open(dst, "wb") as f_out:
        f_out.write(f_in.read())
    print("Done.")


def load_records(path: str) -> list[dict]:
    """Read the JSONL file line by line and return all valid records."""
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"Skipping malformed JSON at line {line_num}: {e}")
    print(f"Loaded {len(records):,} records.")
    return records


def build_dataset(records: list[dict]) -> tuple[Dataset, Dataset]:
    """
    Combine title and content into a single text field, then split
    into train/test subsets.
    """
    dataset = Dataset.from_list(records)

    def merge_fields(example):
        # Concatenate title and content so the model learns the relationship
        example["text"] = example["title"] + " - " + example["content"]
        return example

    dataset = dataset.map(merge_fields)
    dataset = dataset.remove_columns(["uid", "title", "content", "target_ind", "target_rel"])

    splits = dataset.train_test_split(test_size=TEST_SPLIT, seed=SEED)
    train_ds = splits["train"]
    test_ds = splits["test"]

    print(f"Train samples: {len(train_ds):,} | Test samples: {len(test_ds):,}")
    return train_ds, test_ds


# ─────────────────────────────────────────────
# Step 3: Model loading + LoRA adapter
# ─────────────────────────────────────────────

def load_base_model():
    """Load the quantized base model and tokenizer via Unsloth (requires CUDA)."""
    if not UNSLOTH_AVAILABLE:
        raise RuntimeError("Unsloth is not available. Run this on Google Colab with a GPU.")

    print(f"Loading base model: {MODEL_NAME}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        MODEL_NAME,
        load_in_4bit=LOAD_IN_4BIT,
        device_map="auto",
    )
    return model, tokenizer


def attach_lora(model):
    """Wrap the base model with trainable LoRA adapters (QLoRA)."""
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_RANK,
        target_modules=TARGET_MODULES,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        use_gradient_checkpointing="unsloth",  # reduces VRAM usage
        random_state=SEED,
        use_rslora=False,
        loftq_config=None,
    )
    return model


# ─────────────────────────────────────────────
# Step 4: Inference helpers
# ─────────────────────────────────────────────

def build_prompt(text: str) -> str:
    return f"{INSTRUCTION}\n\nQuestion:\n{text}\n\nAnswer:"


def generate_response(model, tokenizer, text: str) -> str:
    """Run greedy/sample generation on a single input text."""
    prompt = build_prompt(text)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=MAX_SEQ_LENGTH)

    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            num_return_sequences=1,
        )

    # Slice off the prompt tokens so we only decode the new tokens
    generated = tokenizer.decode(output_ids[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
    return generated.strip()


def run_inference(model, tokenizer, samples: Dataset, label: str) -> None:
    FastLanguageModel.for_inference(model)
    print(f"\n{'='*60}")
    print(f"Responses — {label}")
    print("=" * 60)
    for i, sample in enumerate(samples):
        response = generate_response(model, tokenizer, sample["text"])
        print(f"\n[{i+1}] Question : {sample['text']}")
        print(f"    Answer   : {response}")


# ─────────────────────────────────────────────
# Step 5: Tokenization for training
# ─────────────────────────────────────────────

def tokenize_dataset(train_ds: Dataset, tokenizer) -> Dataset:
    """Format each example as an instruction prompt and tokenize it."""

    def preprocess(examples):
        prompts = [build_prompt(t) for t in examples["text"]]
        tokenized = tokenizer(prompts, max_length=MAX_SEQ_LENGTH, truncation=True)
        # Causal LM: labels == input_ids (the model predicts the next token)
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    return train_ds.map(preprocess, batched=True, remove_columns=train_ds.column_names)


# ─────────────────────────────────────────────
# Step 6: Fine-tuning
# ─────────────────────────────────────────────

def fine_tune(model, tokenizer, tokenized_train: Dataset) -> None:
    bf16 = is_bfloat16_supported()

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=tokenized_train,
        dataset_text_field="input_ids",
        max_seq_length=MAX_SEQ_LENGTH,
        dataset_num_proc=2,
        packing=False,
        args=TrainingArguments(
            per_device_train_batch_size=BATCH_SIZE,
            gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
            warmup_steps=WARMUP_STEPS,
            max_steps=MAX_STEPS,
            learning_rate=LEARNING_RATE,
            fp16=not bf16,
            bf16=bf16,
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=SEED,
            output_dir=OUTPUT_DIR,
        ),
    )

    print("\nStarting fine-tuning...")
    trainer.train()
    print("Fine-tuning complete.")


# ─────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────

def main():
    if not UNSLOTH_AVAILABLE:
        print("Cannot run full pipeline: unsloth is not installed.")
        print("Please run this script on Google Colab with a GPU.")
        return

    print("GPU available:", torch.cuda.is_available())

    # 1. Mount drive and decompress dataset
    mount_drive()
    decompress_dataset(DATASET_PATH, UNCOMPRESSED_PATH)

    # 2. Load data
    records = load_records(UNCOMPRESSED_PATH)
    train_ds, test_ds = build_dataset(records)
    test_samples = test_ds.select(range(INFERENCE_SAMPLES))

    # 3. Load base model
    model, tokenizer = load_base_model()
    model = attach_lora(model)

    # 4. Baseline inference (before fine-tuning)
    run_inference(model, tokenizer, test_samples, label="BASE MODEL (before fine-tuning)")

    # 5. Tokenize and train
    tokenized_train = tokenize_dataset(train_ds, tokenizer)
    fine_tune(model, tokenizer, tokenized_train)

    # 6. Save fine-tuned model
    print(f"\nSaving model to '{MODEL_SAVE_DIR}'...")
    model.save_pretrained(MODEL_SAVE_DIR)
    tokenizer.save_pretrained(MODEL_SAVE_DIR)

    # 7. Reload fine-tuned model and run inference
    model_ft, tokenizer_ft = FastLanguageModel.from_pretrained(
        MODEL_SAVE_DIR,
        load_in_4bit=LOAD_IN_4BIT,
        device_map="auto",
    )
    run_inference(model_ft, tokenizer_ft, test_samples, label="FINE-TUNED MODEL")


if __name__ == "__main__":
    main()

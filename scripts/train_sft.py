"""QLoRA SFT training for TriageSieve triage agent.

Usage:
    python scripts/train_sft.py --dataset data/sft_dataset.jsonl --output outputs/sft_model

Trains Qwen2.5-1.5B-Instruct with QLoRA on expert demonstration data.
Designed for RTX 3050 4GB VRAM.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="QLoRA SFT training for TriageSieve agent.")
    parser.add_argument("--dataset", type=str, default="data/sft_dataset.jsonl", help="Training data JSONL")
    parser.add_argument("--output", type=str, default="outputs/sft_model", help="Output model directory")
    parser.add_argument("--base-model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct", help="Base model ID")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=1, help="Per-device batch size")
    parser.add_argument("--grad-accum", type=int, default=8, help="Gradient accumulation steps")
    parser.add_argument("--max-seq-len", type=int, default=2048, help="Max sequence length")
    parser.add_argument("--lora-rank", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--eval-split", type=float, default=0.1, help="Fraction for eval split")
    args = parser.parse_args()

    # ---- Imports (heavy, so deferred) ----
    import torch
    from datasets import Dataset
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        TrainingArguments,
    )
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from trl import SFTTrainer, SFTConfig

    logger.info(f"PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}, VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    # ---- Load dataset ----
    logger.info(f"Loading dataset from {args.dataset}")
    raw_data = []
    with open(args.dataset, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                raw_data.append(json.loads(line))

    logger.info(f"Loaded {len(raw_data)} examples")

    # Split into train/eval
    split_idx = int(len(raw_data) * (1 - args.eval_split))
    train_data = raw_data[:split_idx]
    eval_data = raw_data[split_idx:]
    logger.info(f"Train: {len(train_data)}, Eval: {len(eval_data)}")

    train_dataset = Dataset.from_list([{"messages": d["messages"]} for d in train_data])
    eval_dataset = Dataset.from_list([{"messages": d["messages"]} for d in eval_data])

    # ---- Load tokenizer ----
    logger.info(f"Loading tokenizer: {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ---- Quantization config (4-bit NF4) ----
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    # ---- Load model ----
    logger.info(f"Loading model: {args.base_model} (4-bit quantized)")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    model = prepare_model_for_kbit_training(model)

    # ---- LoRA config ----
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )

    logger.info(f"LoRA config: rank={args.lora_rank}, alpha={args.lora_alpha}")

    # ---- Training arguments ----
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        weight_decay=0.01,
        bf16=True,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        max_length=args.max_seq_len,
        report_to="none",
        gradient_checkpointing=True,
        optim="adamw_torch",
        dataloader_pin_memory=False,
        packing=False,
    )

    # ---- Trainer ----
    logger.info("Initializing SFTTrainer...")
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=lora_config,
    )

    # ---- Train ----
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(f"Trainable params: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")
    logger.info("Starting training...")

    train_result = trainer.train()

    # ---- Save ----
    logger.info(f"Saving model to {output_dir}")
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    # ---- Log metrics ----
    metrics = train_result.metrics
    logger.info(f"Training complete!")
    logger.info(f"  Train loss: {metrics.get('train_loss', 'N/A'):.4f}")
    logger.info(f"  Train runtime: {metrics.get('train_runtime', 0):.0f}s")
    logger.info(f"  Train samples/sec: {metrics.get('train_samples_per_second', 0):.2f}")

    # Eval
    eval_metrics = trainer.evaluate()
    logger.info(f"  Eval loss: {eval_metrics.get('eval_loss', 'N/A'):.4f}")

    # Save metrics
    all_metrics = {**metrics, **eval_metrics}
    with open(output_dir / "training_metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2)
    logger.info(f"Metrics saved to {output_dir / 'training_metrics.json'}")


if __name__ == "__main__":
    main()

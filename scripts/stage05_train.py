#!/usr/bin/env python3
"""
Stage 0.5 Visual Aux Decoder Pretrain — 独立训练脚本

替代 swift sft，零 swift 依赖。使用 HuggingFace Trainer +
自定义 compute_loss 劫持，用 visual_aux_pretrain_forward() 替代默认 CE。

数据格式：encoded JSONL（由 prepare_stage05_data.py encode 生成）
  {"messages": [{"role": "user", "content": "<image>"}],
   "images": ["path/to/frame_0.jpg"],
   "future_image_tokens": "<|image start|>12*21<|image token|>..."}

用法:
    source venv/onevl/bin/activate
    torchrun --nproc_per_node=1 scripts/stage05_train.py \\
        --model_name_or_path outputs/navsim/.../checkpoint-2300 \\
        --data_path demo_data/navsim/data.jsonl \\
        --output_dir outputs/navsim/stage05_pure \\
        --max_steps 100 --per_device_train_batch_size 1 \\
        --gradient_accumulation_steps 2 --learning_rate 1e-4 \\
        --deepspeed ds_configs/zero3.json --freeze_vit True
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import (
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    AutoProcessor,
    Qwen3VLForConditionalGeneration,
)

from onevl import add_visual_tokens_to_tokenizer
from onevl.forward import visual_aux_pretrain_forward

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Arguments
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class DataArguments:
    """Data configuration arguments."""

    data_path: str = field(
        default="demo_data/navsim/layer_c_train.jsonl",
        metadata={"help": "Path to encoded JSONL training data (from prepare_stage05_data.py encode)"},
    )
    dataset_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Dataset root directory (for resolving image paths). "
                          "Defaults to dirname of data_path"},
    )
    chunk_size: int = field(
        default=512,
        metadata={"help": "Chunk size for chunked LM-head + CE loss"},
    )
    freeze_vit: bool = field(
        default=True,
        metadata={"help": "Freeze ViT parameters (feature extraction only)"},
    )


@dataclass
class ModelArguments:
    """Model configuration arguments."""

    model_name_or_path: str = field(
        default="Qwen/Qwen3-VL-2B-Instruct",
        metadata={"help": "Path to pretrained model or checkpoint"},
    )


# ═══════════════════════════════════════════════════════════════════════════
# Dataset
# ═══════════════════════════════════════════════════════════════════════════


def resolve_image_path(img_rel: str, dataset_dir: str) -> str | None:
    """解析图像路径（支持相对路径多种前缀）。"""
    if os.path.isabs(img_rel):
        return img_rel if os.path.exists(img_rel) else None

    candidates = [
        img_rel,
        os.path.join("/root/autodl-tmp/OneVL_training", img_rel),
        os.path.join("/root/autodl-tmp/OneVL_training/data", img_rel),
        os.path.join(dataset_dir, img_rel),
    ]
    for c in candidates:
        if os.path.exists(c):
            return os.path.abspath(c)
    return None


class LayerCDataset(Dataset):
    """Encoded JSONL 数据集。

    每个样本:
      - "images": [img_path]
      - "future_image_tokens": str
    """

    def __init__(self, data_path: str, dataset_dir: str | None = None):
        super().__init__()
        self.data_path = data_path
        self.dataset_dir = dataset_dir or os.path.dirname(os.path.abspath(data_path))

        with open(data_path) as f:
            self.records = [json.loads(line) for line in f]

        logger.info(f"Loaded {len(self.records)} records from {data_path}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        record = self.records[idx]

        # 解析图像路径
        img_rel = record["images"][0]
        img_path = resolve_image_path(img_rel, self.dataset_dir)
        if img_path is None:
            raise FileNotFoundError(
                f"Image not found: {img_rel} (dataset_dir={self.dataset_dir})"
            )

        return {
            "image_path": img_path,
            "future_image_tokens": record["future_image_tokens"],
        }


# ═══════════════════════════════════════════════════════════════════════════
# Data Collator
# ═══════════════════════════════════════════════════════════════════════════


class VisualAuxDataCollator:
    """Data collator for Stage 0.5 visual aux pretrain.

    逐样本：
      1. 加载图像 → PIL Image
      2. 调用 processor(images=[img], text="") 得到 pixel_values, image_grid_thw
      3. 收集 future_image_tokens 为 list[str]

    ViT 原生支持 batch（不同分辨率通过 torch.concat 拼接）。
    """

    def __init__(self, processor, dataset_dir: str):
        self.processor = processor
        self.dataset_dir = dataset_dir

    def __call__(self, examples: list[dict]) -> dict:
        pixel_values_list = []
        grid_thw_list = []
        future_tokens = []

        for ex in examples:
            img_path = ex["image_path"]
            # 如果 image_path 是相对路径，尝试解析
            if not os.path.isabs(img_path):
                resolved = resolve_image_path(img_path, self.dataset_dir)
                if resolved is None:
                    raise FileNotFoundError(f"Image not found: {img_path}")
                img_path = resolved

            img = Image.open(img_path).convert("RGB")
            proc_out = self.processor(
                text="", images=[img], return_tensors="pt"
            )
            pixel_values_list.append(proc_out["pixel_values"])   # [1, 3, H, W]
            grid_thw_list.append(proc_out["image_grid_thw"])     # [1, 3]
            future_tokens.append(ex["future_image_tokens"])

        # Qwen3-VL style: concat along dim=0 (不同分辨率兼容)
        pixel_values = torch.cat(pixel_values_list, dim=0)
        image_grid_thw = torch.cat(grid_thw_list, dim=0)

        return {
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
            "future_image_tokens": future_tokens,  # list[str]
        }


# ═══════════════════════════════════════════════════════════════════════════
# Custom Trainer
# ═══════════════════════════════════════════════════════════════════════════


class VisualAuxTrainer(Trainer):
    """HuggingFace Trainer with custom compute_loss for Stage 0.5."""

    def __init__(self, tokenizer=None, chunk_size: int = 512, **kwargs):
        super().__init__(**kwargs)
        self._visual_aux_tokenizer = tokenizer
        self._chunk_size = chunk_size

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """Custom loss: delegate to visual_aux_pretrain_forward."""
        loss, aux_info = visual_aux_pretrain_forward(
            model=model,
            pixel_values=inputs["pixel_values"],
            image_grid_thw=inputs["image_grid_thw"],
            future_image_tokens=inputs["future_image_tokens"],
            tokenizer=self._visual_aux_tokenizer,
            chunk_size=self._chunk_size,
        )

        # Log aux info
        if self.state is not None and self.state.global_step % 10 == 0:
            logger.info(
                f"[Stage0.5] step={self.state.global_step} "
                f"loss={aux_info['loss_value']:.4f} "
                f"batch_size={aux_info['batch_size']} "
                f"n_vis={aux_info['n_vis']} "
                f"n_target={aux_info['n_target']} "
                f"seq_len={aux_info['seq_len']}"
            )

        return (loss, aux_info) if return_outputs else loss


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════


def freeze_vit(model: torch.nn.Module) -> None:
    """冻结 ViT 参数（feature extraction only）。"""
    for name, param in model.model.visual.named_parameters():
        param.requires_grad = False
    frozen = sum(1 for _ in model.model.visual.named_parameters())
    logger.info(f"[Stage0.5] Frozen ViT: {frozen} parameter groups")


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    # Allow yaml/json config files
    if len(sys.argv) > 1 and sys.argv[1].endswith((".yaml", ".yml", ".json")):
        model_args, data_args, training_args = parser.parse_yaml_file(
            sys.argv[1], allow_extra_keys=True
        )
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # ── Setup logging ──────────────────────────────────────────────────
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
    )
    logger.setLevel(logging.INFO)

    # ── 1. Load processor & model ──────────────────────────────────────
    logger.info("=" * 60)
    logger.info(f"Loading model: {model_args.model_name_or_path}")
    logger.info(f"Training args: {training_args}")

    processor = AutoProcessor.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=True,
    )

    # Register 131k visual code tokens on the tokenizer
    add_visual_tokens_to_tokenizer(processor.tokenizer)

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    model.resize_token_embeddings(len(processor.tokenizer))
    model.train()

    logger.info(f"  Vocab size: {len(processor.tokenizer)}")
    logger.info(f"  Model params: {sum(p.numel() for p in model.parameters()):,}")

    # ── 2. Freeze ViT ───────────────────────────────────────────────────
    if data_args.freeze_vit:
        freeze_vit(model)

    # ── 3. Load dataset ────────────────────────────────────────────────
    dataset_dir = data_args.dataset_dir or os.path.dirname(
        os.path.abspath(data_args.data_path)
    )
    train_dataset = LayerCDataset(data_args.data_path, dataset_dir=dataset_dir)
    logger.info(f"  Dataset: {len(train_dataset)} samples")

    # ── 4. Data collator ───────────────────────────────────────────────
    data_collator = VisualAuxDataCollator(processor, dataset_dir=dataset_dir)

    # ── 5. Trainer ─────────────────────────────────────────────────────
    trainer = VisualAuxTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=processor.tokenizer,
        chunk_size=data_args.chunk_size,
        data_collator=data_collator,
        processing_class=processor,
    )

    # ── 6. Train ───────────────────────────────────────────────────────
    logger.info("Starting training...")
    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)

    # ── 7. Save final model ────────────────────────────────────────────
    trainer.save_model()
    logger.info(f"Model saved to {training_args.output_dir}")


if __name__ == "__main__":
    main()
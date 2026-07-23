#!/usr/bin/env python3
"""
Stage 0.5 推理可视化脚本

加载训练好的 visual aux decoder pretrain checkpoint，在验证集上跑推理，
将输入图、GT 未来帧、模型预测未来帧解码为图像并保存。

每样本独立子文件夹：input.jpg / gt_0.jpg / gt_1.jpg / pred_0.jpg / pred_1.jpg

用法:
    source venv/onevl/bin/activate
    python scripts/stage05_inference_vis.py

支持两种推理模式:
  --mode teacher_forcing (默认, 单次 forward, 使用 GT context, 快)
  --mode autoregressive (逐步自回归生成, 慢但更真实)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from PIL import Image

from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from onevl import add_visual_tokens_to_tokenizer
from onevl.emu35_tokenizer import (
    load_vision_tokenizer,
    tokens_to_image,
)

# ── Constants ──────────────────────────────────────────────────────────────
VIS_TOKEN_START = 151673     # <|visual token 000000|> 的 token ID
NUM_VIS_CODES = 131072
GRID_H, GRID_W = 12, 21     # 该数据集固定 spatial grid 尺寸
NUM_FRAMES = 2              # future_image_tokens 含 2 帧


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


def extract_visual_codes_from_ids(
    pred_ids: list[int],
    vis_start: int = VIS_TOKEN_START,
    num_codes: int = NUM_VIS_CODES,
) -> list[int]:
    """从预测的 token ID 序列中提取 visual code ID。"""
    return [tid - vis_start for tid in pred_ids if vis_start <= tid < vis_start + num_codes]


def codes_to_grids(
    codes: list[int],
    grid_h: int = GRID_H,
    grid_w: int = GRID_W,
    num_frames: int = NUM_FRAMES,
) -> list[torch.Tensor]:
    """扁平 visual code ID 列表 → [H, W] tensor 网格列表。"""
    expected = grid_h * grid_w * num_frames
    if len(codes) < expected:
        codes = codes + [0] * (expected - len(codes))
    codes = codes[:expected]

    grids = []
    for i in range(num_frames):
        frame = codes[i * grid_h * grid_w:(i + 1) * grid_h * grid_w]
        grids.append(torch.tensor(frame, dtype=torch.long).reshape(grid_h, grid_w))
    return grids


# ═══════════════════════════════════════════════════════════════════════════
# Teacher-Forcing 模式：单次 forward + lm_head 预测所有 token
# ═══════════════════════════════════════════════════════════════════════════

def predict_teacher_forcing(
    model,
    vis_embeds: torch.Tensor,
    future_text: str,
    tokenizer,
    device: str,
) -> list[int]:
    """Teacher-forcing 推理：用 GT 做 context，一次 forward 预测所有位置。

    Returns: 预测的 visual code ID 列表 (~504 个, 252/frame × 2)。
    """
    # 1. Tokenize future_text
    target_ids = tokenizer.encode(future_text, add_special_tokens=False)
    if tokenizer.eos_token_id is not None:
        target_ids = target_ids + [tokenizer.eos_token_id]
    target_tensor = torch.tensor(target_ids, device=device, dtype=torch.long)

    # 2. 构建 combined_embeds = [vis_embeds, target_embeds]
    target_embeds = model.get_input_embeddings()(target_tensor)  # [n_target, hidden]
    combined = torch.cat([vis_embeds, target_embeds], dim=0).unsqueeze(0)  # [1, seq_len, hidden]
    attn_mask = torch.ones(1, combined.shape[1], dtype=torch.long, device=device)

    # 3. Language model forward
    with torch.no_grad():
        outputs = model.model.language_model(
            input_ids=None,
            attention_mask=attn_mask,
            inputs_embeds=combined,
            use_cache=False,
            output_hidden_states=True,
        )
    last_hidden = outputs.hidden_states[-1]  # [1, seq_len, hidden]

    # 4. lm_head 预测
    n_vis = vis_embeds.shape[0]
    pred_hidden = last_hidden[0, n_vis - 1:-1]  # [n_target, hidden]
    pred_logits = model.lm_head(pred_hidden)     # [n_target, vocab]
    pred_ids = pred_logits.argmax(dim=-1)        # [n_target]

    # 5. 提取 visual codes
    return extract_visual_codes_from_ids(pred_ids.tolist())


# ═══════════════════════════════════════════════════════════════════════════
# Autoregressive 模式：逐步生成
# ═══════════════════════════════════════════════════════════════════════════

def predict_autoregressive(
    model,
    vis_embeds: torch.Tensor,
    tokenizer,
    device: str,
    max_new_tokens: int = 600,
) -> list[int]:
    """自回归推理：逐步生成 token，使用 KV cache 加速。

    Returns: 生成的 visual code ID 列表。
    """
    lm = model.model.language_model
    end_token_id = tokenizer.convert_tokens_to_ids("<|image end|>")

    generated_ids: list[int] = []

    # Pre-allocate combined embedding buffer
    max_len = vis_embeds.shape[0] + max_new_tokens
    hidden = vis_embeds.shape[-1]
    combined = torch.empty(1, max_len, hidden, dtype=vis_embeds.dtype, device=device)
    combined[:, :vis_embeds.shape[0]] = vis_embeds.unsqueeze(0)
    seq_len = vis_embeds.shape[0]

    for step in range(max_new_tokens):
        # Full forward (no KV cache to save memory)
        with torch.no_grad():
            outputs = lm(
                input_ids=None,
                inputs_embeds=combined[:, :seq_len],
                use_cache=False,
            )

        logits = model.lm_head(outputs.last_hidden_state[0, -1])
        next_id = logits.argmax().item()
        generated_ids.append(next_id)

        # Append predicted token embedding to the sequence
        next_embed = model.get_input_embeddings()(
            torch.tensor([next_id], device=device)).unsqueeze(0)
        combined[:, seq_len:seq_len + 1] = next_embed
        seq_len += 1

        # Free intermediate tensors
        del outputs

        # Stop after 2 <|image end|> tokens
        if next_id == end_token_id:
            end_count = sum(1 for g in generated_ids if g == end_token_id)
            if end_count >= 2:
                break

        if (step + 1) % 50 == 0:
            print(f"    ... generated {step + 1} tokens (seq_len={seq_len})")

    print(f"    Generated {len(generated_ids)} tokens total")
    return extract_visual_codes_from_ids(generated_ids)


# ═══════════════════════════════════════════════════════════════════════════
# GT token 解析（直接使用 emu35 demo 中的 parse_token_block）
# ═══════════════════════════════════════════════════════════════════════════

def parse_gt_future_blocks(ft_text: str) -> list[torch.Tensor]:
    """解析 future_image_tokens 字符串为 [H, W] grid 列表。"""
    from onevl.emu35_tokenizer import extract_future_blocks, parse_token_block
    blocks = extract_future_blocks(ft_text)
    # extract_future_blocks 返回的是 <|image start|>...<|image end|> 之间的内容，
    # 但 parse_token_block 需要包含标记的完整块，所以重新封装
    return [parse_token_block(f"<|image start|>{b}<|image end|>") for b in blocks]


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Stage 0.5 Inference Visualization")
    parser.add_argument("--ckpt_dir", type=str,
                        default="/root/autodl-tmp/OneVL_training/outputs/navsim/"
                                "qwen3_vl_visual_aux_pretrain_stage0_5_vis4_txt2/checkpoint-2300")
    parser.add_argument("--emu35_path", type=str,
                        default="/root/.cache/huggingface/hub/models--BAAI--Emu3.5-VisionTokenizer/"
                                "snapshots/a5c45bdb8084763048e094ae124778bfeca7d5ee/")
    parser.add_argument("--dataset_path", type=str,
                        default="/root/autodl-tmp/OneVL_training/demo_data/navsim/"
                                "navsim_vis4_text2_demo100.jsonl")
    parser.add_argument("--output_dir", type=str,
                        default="/root/autodl-tmp/OneVL_training/vis_results")
    parser.add_argument("--num_samples", type=int, default=6)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--mode", type=str, default="autoregressive",
                        choices=["teacher_forcing", "autoregressive"])
    parser.add_argument("--start_idx", type=int, default=0,
                        help="从数据集第几条开始（用于跳过前面的样本）")
    args = parser.parse_args()

    device = args.device

    # ── 1. Load processor & model ──────────────────────────────────────
    print("=" * 60)
    print("Loading model checkpoint...")
    print(f"  Checkpoint: {args.ckpt_dir}")

    processor = AutoProcessor.from_pretrained(args.ckpt_dir, trust_remote_code=True)
    add_visual_tokens_to_tokenizer(processor.tokenizer)

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.ckpt_dir,
        trust_remote_code=True,
        dtype=torch.bfloat16,
    ).eval()
    model.to(device)
    model.resize_token_embeddings(len(processor.tokenizer))
    print(f"  Vocab size: {len(processor.tokenizer)}")
    print(f"  Device: {device}")

    # ── 2. Load Emu3.5 VisionTokenizer ────────────────────────────────
    print("\nLoading Emu3.5 VisionTokenizer...")
    vq = load_vision_tokenizer(args.emu35_path, device=device)
    print("  Done.")

    # ── 3. Load dataset ────────────────────────────────────────────────
    print(f"\nLoading dataset: {args.dataset_path}")
    with open(args.dataset_path) as f:
        records = [json.loads(line) for line in f]
    records = records[args.start_idx:args.start_idx + args.num_samples]
    print(f"  Records to process: {len(records)}")

    # ── 4. Process each sample ─────────────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)
    dataset_dir = os.path.dirname(os.path.abspath(args.dataset_path))

    for idx, record in enumerate(records):
        sample_dir = os.path.join(args.output_dir, f"sample_{args.start_idx + idx:04d}")
        os.makedirs(sample_dir, exist_ok=True)
        print(f"\n{'─' * 60}")
        print(f"Sample {args.start_idx + idx} → {sample_dir}/")

        # 4a. Load input image
        img_rel = record["images"][0]
        img_path = resolve_image_path(img_rel, dataset_dir)
        if img_path is None:
            print(f"  ⚠ Image not found: {img_rel}, skipping...")
            continue

        input_image = Image.open(img_path).convert("RGB")
        input_image.save(os.path.join(sample_dir, "input.jpg"))
        print(f"  Input: {os.path.relpath(img_path)} ({input_image.size})")

        # 4b. Process input for ViT
        proc_out = processor(text="", images=[input_image], return_tensors="pt")
        pixel_values = proc_out["pixel_values"].to(device, dtype=torch.bfloat16)
        image_grid_thw = proc_out["image_grid_thw"].to(device)

        # 4c. Run ViT → vis_embeds
        with torch.no_grad():
            vision_out = model.model.visual(
                hidden_states=pixel_values,
                grid_thw=image_grid_thw,
            )
        vis_embeds = vision_out.pooler_output
        if isinstance(vis_embeds, (list, tuple)):
            vis_embeds = vis_embeds[0]
        print(f"  vis_embeds: {vis_embeds.shape[0]} patches × {vis_embeds.shape[1]} dim")

        # 4d. Decode GT future_image_tokens
        print("  GT blocks:")
        ft = record["future_image_tokens"]
        gt_grids = parse_gt_future_blocks(ft)
        for gi, grid in enumerate(gt_grids):
            gt_img = tokens_to_image(grid, vq, device=device)
            gt_img.save(os.path.join(sample_dir, f"gt_{gi}.jpg"))
            print(f"    gt_{gi}.jpg: grid {grid.shape} → image {gt_img.size}")

        # 4e. Generate predictions
        print(f"  Predicting ({args.mode})...")
        if args.mode == "teacher_forcing":
            pred_codes = predict_teacher_forcing(
                model, vis_embeds, ft, processor.tokenizer, device,
            )
        else:
            pred_codes = predict_autoregressive(
                model, vis_embeds, processor.tokenizer, device,
            )
        print(f"    Extracted {len(pred_codes)} visual codes")

        # 4f. Decode predictions
        pred_grids = codes_to_grids(pred_codes)
        print(f"  Pred blocks: {len(pred_grids)}")
        for pi, grid in enumerate(pred_grids):
            pred_img = tokens_to_image(grid, vq, device=device)
            pred_img.save(os.path.join(sample_dir, f"pred_{pi}.jpg"))
            print(f"    pred_{pi}.jpg: grid {grid.shape} → image {pred_img.size}")

    print(f"\n{'=' * 60}")
    print(f"All done! Results saved to: {args.output_dir}")
    print(f"=" * 60)


if __name__ == "__main__":
    main()

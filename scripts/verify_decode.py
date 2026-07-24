#!/usr/bin/env python3
"""
Phase 1: 反向验证 Emu3.5 decode 正确性

从 demo_data JSONL 读取 future_image_tokens，用 Emu3.5 VisionTokenizer 解码为图像，
同时保存当前帧原始图像，输出到 vis_results/decode_verify/ 供肉眼观察。

用法:
    python scripts/verify_decode.py \
        --dataset_path demo_data/navsim/navsim_vis4_text2_demo100.jsonl \
        --emu35_path /path/to/Emu3.5-VisionTokenizer \
        --output_dir vis_results/decode_verify \
        --num_samples 5
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from PIL import Image

from onevl.emu35_tokenizer import (
    extract_future_blocks,
    load_vision_tokenizer,
    parse_token_block,
    tokens_to_image,
)


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


def main():
    parser = argparse.ArgumentParser(
        description="Phase 1: 反向验证 Emu3.5 decode 正确性",
    )
    parser.add_argument("--dataset_path", type=str, required=True,
                        help="demo_data JSONL 路径")
    parser.add_argument("--emu35_path", type=str, required=True,
                        help="Emu3.5-VisionTokenizer 模型路径")
    parser.add_argument("--output_dir", type=str, default="vis_results/decode_verify",
                        help="输出目录")
    parser.add_argument("--num_samples", type=int, default=5,
                        help="处理的样本数量")
    parser.add_argument("--start_idx", type=int, default=0,
                        help="起始样本索引")
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    device = args.device
    output_dir = Path(args.output_dir)

    # 1. Load Emu3.5 VisionTokenizer
    print("=" * 60)
    print("Loading Emu3.5 VisionTokenizer...")
    print(f"  Model: {args.emu35_path}")
    vq = load_vision_tokenizer(args.emu35_path, device=device)
    print("  Done.")

    # 2. Load dataset
    print(f"\nLoading dataset: {args.dataset_path}")
    with open(args.dataset_path) as f:
        records = [json.loads(line) for line in f]
    records = records[args.start_idx:args.start_idx + args.num_samples]
    print(f"  Samples to process: {len(records)}")

    # 3. Process each sample
    dataset_dir = os.path.dirname(os.path.abspath(args.dataset_path))
    output_dir.mkdir(parents=True, exist_ok=True)

    for idx, record in enumerate(records):
        sample_dir = output_dir / f"sample_{args.start_idx + idx:04d}"
        sample_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n{'─' * 60}")
        print(f"Sample {args.start_idx + idx} → {sample_dir}/")

        # 3a. Save current frame image
        img_rel = record["images"][0]
        img_path = resolve_image_path(img_rel, dataset_dir)
        if img_path is None:
            print(f"  ⚠ Current image not found: {img_rel}, skipping...")
            continue

        current_img = Image.open(img_path).convert("RGB")
        current_img.save(str(sample_dir / "current.jpg"))
        print(f"  Current: {os.path.relpath(img_path)} ({current_img.size})")

        # 3b. Parse and decode future_image_tokens
        ft = record.get("future_image_tokens", "")
        if not ft:
            print(f"  ⚠ No future_image_tokens found, skipping...")
            continue

        blocks = extract_future_blocks(ft)
        print(f"  Found {len(blocks)} future frame block(s)")

        for bi, block in enumerate(blocks):
            # Re-wrap for parse_token_block (it expects the full <|image start|>...<|image end|>)
            full_block = f"<|image start|>{block}<|image end|>"
            try:
                grid = parse_token_block(full_block)
                print(f"    Frame {bi}: grid {grid.shape}, "
                      f"token range=[{grid.min().item()}, {grid.max().item()}]")

                decoded = tokens_to_image(grid, vq, device=device)
                out_path = sample_dir / f"future_{bi}.jpg"
                decoded.save(str(out_path))
                print(f"      → saved {out_path} ({decoded.size})")
            except Exception as e:
                print(f"    ⚠ Frame {bi} decode error: {e}")

    print(f"\n{'=' * 60}")
    print(f"All done! Results saved to: {output_dir.resolve()}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
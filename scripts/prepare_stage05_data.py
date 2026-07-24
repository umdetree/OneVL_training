#!/usr/bin/env python3
"""
Stage 0.5 数据预处理管线

将 raw_index（路径三元组）编码为训练用的精简 JSONL（含 Emu3.5 future_image_tokens）。

数据层次:
  raw_index（原始路径索引）:
    {"current": "images/<scene>/CAM_F0/<frame>.jpg",
     "future_0.5s": "images/<scene>/CAM_F0/<frame>.jpg",
     "future_1.0s": "images/<scene>/CAM_F0/<frame>.jpg"}

  encoded（训练数据层，精简 JSONL）:
    {"messages": [{"role": "user", "content": "<image>"}],
     "images": ["path/to/frame_0.jpg"],
     "future_image_tokens": "<|image start|>12*21<|image token|>..."}

子命令:
  encode     读取 raw_index.jsonl，调用 Emu3.5 VisionTokenizer 编码未来帧 → data.jsonl
  minify     从现有完整 JSONL 剥离多余字段 → encoded JSONL（无需 Emu3.5 模型）
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional

from PIL import Image


# ---------------------------------------------------------------------------
# 图像缩放工具
# ---------------------------------------------------------------------------

def _resize_max(img: Image.Image, max_res: int) -> Image.Image:
    """保持宽高比缩放到最大边长不超过 max_res 像素。"""
    w, h = img.size
    if max(w, h) <= max_res:
        return img
    scale = max_res / max(w, h)
    new_w, new_h = int(w * scale), int(h * scale)
    return img.resize((new_w, new_h), Image.LANCZOS)


# ═══════════════════════════════════════════════════════════════════════════
# 子命令: encode — raw_index → encoded（需要 Emu3.5 VisionTokenizer）
# ═══════════════════════════════════════════════════════════════════════════

def cmd_encode(args: argparse.Namespace) -> None:
    """读取 raw_index.jsonl，编码未来帧，输出 data.jsonl。"""
    from onevl.emu35_tokenizer import add_future_tokens_to_record, load_vision_tokenizer

    # 1. Load Emu3.5 VisionTokenizer
    print(f"Loading Emu3.5 VisionTokenizer from {args.emu35_path}...")
    vq = load_vision_tokenizer(args.emu35_path, device=args.device)
    print("  Done.")

    # 2. Load raw_index
    print(f"Loading raw_index from {args.input}...")
    with open(args.input) as f:
        records = [json.loads(line) for line in f]
    print(f"  {len(records)} records")

    # 3. Resolve data root
    data_root = Path(args.data_root).resolve() if args.data_root else Path.cwd()
    print(f"  data_root: {data_root}")

    out_records: list[dict] = []
    errors = 0

    for idx, rec in enumerate(records):
        if args.max_records and idx >= args.max_records:
            break

        current_path = data_root / rec["current"]
        future_0_path = data_root / rec["future_0.5s"]
        future_1_path = data_root / rec["future_1.0s"]

        if not current_path.exists():
            print(f"  ⚠ [{idx}] current not found: {current_path}, skipping")
            errors += 1
            continue

        # 4. Load images
        from PIL import Image
        current_img = Image.open(str(current_path)).convert("RGB")
        future_imgs = []
        for fp, label in [(future_0_path, "future_0.5s"), (future_1_path, "future_1.0s")]:
            if fp.exists():
                img = Image.open(str(fp)).convert("RGB")
                # 缩放到指定尺寸（如有指定），与 demo 数据对齐
                if args.resize is not None:
                    img = img.resize(args.resize, Image.LANCZOS)
                future_imgs.append(img)
            else:
                print(f"  ⚠ [{idx}] {label} not found: {fp}, using placeholder")
                future_imgs.append(current_img)  # fallback

        # 5. Build minimal record
        record = {
            "messages": [{"role": "user", "content": "<image>"}],
            "images": [str(current_path)],
        }

        # 6. Encode future images → future_image_tokens
        record = add_future_tokens_to_record(
            record, future_imgs, vq, device=args.device,
        )

        out_records.append(record)

        if (idx + 1) % 10 == 0:
            print(f"  [{idx + 1}/{len(records)}] processed")

    # 7. Write output
    out_path = args.output
    with open(out_path, "w") as f:
        for r in out_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Wrote {len(out_records)} records to {out_path}")
    if errors:
        print(f"  ({errors} errors/skipped)")


# ═══════════════════════════════════════════════════════════════════════════
# 子命令: minify — 从现有完整 JSONL 剥离多余字段 → encoded JSONL
# ═══════════════════════════════════════════════════════════════════════════

def cmd_minify(args: argparse.Namespace) -> None:
    """从现有完整 JSONL 中剥离 messages/think_steps 等多余字段。

    保留: images, future_image_tokens
    简化: messages 只保留 "<image>" 占位符
    去掉: think_steps, 完整的 messages 内容
    """
    print(f"Reading from {args.input}...")
    with open(args.input) as f:
        records = [json.loads(line) for line in f]
    print(f"  {len(records)} records")

    out_records: list[dict] = []
    for idx, rec in enumerate(records):
        simplified = {
            "messages": [{"role": "user", "content": "<image>"}],
            "images": rec.get("images", []),
            "future_image_tokens": rec.get("future_image_tokens", ""),
        }
        out_records.append(simplified)

    # 验证 future_image_tokens 格式
    for idx, rec in enumerate(out_records):
        ft = rec.get("future_image_tokens", "")
        if not ft.startswith("<|image start|>"):
            print(f"  ⚠ [{idx}] future_image_tokens may be malformed "
                  f"(starts with {ft[:50]!r})")

    out_path = args.output
    with open(out_path, "w") as f:
        for r in out_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Wrote {len(out_records)} simplified records to {out_path}")
    print(f"  Fields: {list(out_records[0].keys())}")


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage 0.5 数据预处理管线",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    # encode — raw_index → data.jsonl
    p_encode = sub.add_parser(
        "encode",
        help="raw_index.jsonl → data.jsonl（调用 Emu3.5 编码未来帧）",
    )
    p_encode.add_argument("--input", type=str, required=True,
                          help="raw_index.jsonl 路径（含 current/future_0.5s/future_1.0s 路径三元组）")
    p_encode.add_argument("--emu35_path", type=str, required=True,
                          help="Emu3.5-VisionTokenizer 模型路径")
    p_encode.add_argument("--output", type=str, default="data.jsonl",
                          help="输出 data.jsonl 路径（默认：data.jsonl）")
    p_encode.add_argument("--data_root", type=str, default=None,
                          help="数据集根目录（用于解析 raw_index 中的相对路径）")
    p_encode.add_argument("--device", type=str, default="cuda:0")
    p_encode.add_argument("--max_records", type=int, default=None,
                          help="最多处理记录数（调试用）")
    p_encode.add_argument("--resize", type=int, nargs=2, default=None,
                          metavar=("WIDTH", "HEIGHT"),
                          help="将未来帧缩放到指定宽高 (W H)，如 --resize 336 192 与 demo 对齐")

    # minify — 完整 JSONL → encoded JSONL
    p_minify = sub.add_parser(
        "minify",
        help="从现有完整 JSONL 剥离多余字段 → encoded JSONL（无需 Emu3.5 模型）",
    )
    p_minify.add_argument("--input", type=str, required=True,
                          help="现有完整 JSONL 路径")
    p_minify.add_argument("--output", type=str, default="encoded.jsonl",
                          help="输出 encoded JSONL 路径")

    args = parser.parse_args()

    if args.cmd == "encode":
        cmd_encode(args)
    elif args.cmd == "minify":
        cmd_minify(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Stage 0.5 数据预处理管线

将原始图像对转换为训练用的精简 JSONL（Layer C）。

数据层次:
  Layer A（原始数据层）:
    {"current": "path/to/frame_0.jpg", "future_0.5s": "path/to/frame_1.jpg", "future_1.0s": "path/to/frame_2.jpg"}

  Layer C（训练数据层，精简 JSONL）:
    {"messages": [{"role": "user", "content": "<image>"}],
     "images": ["path/to/frame_0.jpg"],
     "future_image_tokens": "<|image start|>12*21<|image token|>..."}

子命令:
  prepare    扫描 NavSim 格式的时序图像目录 → Layer A JSON
  encode     读取 Layer A，调用 Emu3.5 VisionTokenizer 编码未来帧 → Layer C JSONL
  minify     从现有完整 JSONL 剥离多余字段 → Layer C JSONL（无需 Emu3.5 模型）
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional


# ═══════════════════════════════════════════════════════════════════════════
# 子命令: prepare — 扫描 NavSim 时序图像目录 → Layer A
# ═══════════════════════════════════════════════════════════════════════════

def cmd_prepare(args: argparse.Namespace) -> None:
    """扫描 NavSim 目录结构，生成 Layer A JSON（路径索引）。

    NavSim 场景目录结构:
      <root>/dataset/sensor_blobs/trainval/<scene>/CAM_F0/<frame>.jpg

    按文件名字典序排序，每 3 帧为一组（当前帧, +0.5s, +1.0s）。
    步长为 1（滑动窗口），重叠采样。
    """
    root = Path(args.navsim_root)
    scene_glob = args.scene_glob or "*/CAM_F0/*.jpg"
    # 构建完整 glob: <root>/dataset/sensor_blobs/trainval/<scene_glob>
    full_glob = str(root / "dataset/sensor_blobs/trainval" / scene_glob)

    frames = sorted(Path(full_glob).resolve().parent.glob("*.jpg")
                    if os.path.isdir(os.path.dirname(full_glob))
                    else Path().glob(full_glob))

    if not frames:
        # Try simpler glob
        full_glob = str(root / "dataset/sensor_blobs/trainval" / args.scene_glob)
        frames = sorted(Path().glob(full_glob))

    if not frames:
        # Try without trainval
        if "trainval" in args.scene_glob:
            alt_glob = args.scene_glob.replace("trainval/", "")
            full_glob = str(root / "dataset/sensor_blobs" / alt_glob)
            frames = sorted(Path().glob(full_glob))
            if not frames:
                full_glob = str(root / "dataset" / alt_glob)
                frames = sorted(Path().glob(full_glob))

    if not frames:
        print(f"ERROR: No images found matching '{full_glob}'")
        sys.exit(1)

    print(f"Found {len(frames)} frames")

    records: list[dict] = []
    for i in range(len(frames) - 2):
        rel_current = os.path.relpath(str(frames[i]), str(root))
        rel_future_0 = os.path.relpath(str(frames[i + 1]), str(root))
        rel_future_1 = os.path.relpath(str(frames[i + 2]), str(root))
        records.append({
            "current": rel_current,
            "future_0.5s": rel_future_0,
            "future_1.0s": rel_future_1,
        })

    out_path = args.output
    with open(out_path, "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Wrote {len(records)} records to {out_path}")
    if records:
        print(f"  Example: {json.dumps(records[0], indent=2)}")


# ═══════════════════════════════════════════════════════════════════════════
# 子命令: encode — Layer A → Layer C（需要 Emu3.5 VisionTokenizer）
# ═══════════════════════════════════════════════════════════════════════════

def cmd_encode(args: argparse.Namespace) -> None:
    """读取 Layer A JSON，编码未来帧，输出 Layer C JSONL。"""
    from onevl.emu35_tokenizer import add_future_tokens_to_record, load_vision_tokenizer

    # 1. Load Emu3.5 VisionTokenizer
    print(f"Loading Emu3.5 VisionTokenizer from {args.emu35_path}...")
    vq = load_vision_tokenizer(args.emu35_path, device=args.device)
    print("  Done.")

    # 2. Load Layer A
    print(f"Loading Layer A from {args.input}...")
    with open(args.input) as f:
        records = [json.loads(line) for line in f]
    print(f"  {len(records)} records")

    # 3. Resolve paths
    navsim_root = Path(args.navsim_root) if args.navsim_root else Path.cwd()
    out_records: list[dict] = []
    errors = 0

    for idx, rec in enumerate(records):
        if args.max_records and idx >= args.max_records:
            break

        current_path = navsim_root / rec["current"]
        future_0_path = navsim_root / rec["future_0.5s"]
        future_1_path = navsim_root / rec["future_1.0s"]

        if not current_path.exists():
            print(f"  ⚠ [{idx}] current not found: {current_path}, skipping")
            errors += 1
            continue

        # 4. Load images
        from PIL import Image
        current_img = Image.open(str(current_path)).convert("RGB")
        future_imgs = []
        for fp in [future_0_path, future_1_path]:
            if fp.exists():
                future_imgs.append(Image.open(str(fp)).convert("RGB"))
            else:
                print(f"  ⚠ [{idx}] future not found: {fp}, using placeholder")
                future_imgs.append(current_img)  # fallback

        # 5. Build minimal record
        record = {
            "messages": [{"role": "user", "content": "<image>"}],
            "images": [str(current_path)],
        }

        # 6. Encode future images
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
# 子命令: minify — 从现有完整 JSONL 剥离多余字段 → Layer C
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

    # prepare
    p_prepare = sub.add_parser(
        "prepare",
        help="扫描 NavSim 时序图像目录 → Layer A JSON",
    )
    p_prepare.add_argument("--navsim_root", type=str, required=True,
                           help="NavSim 数据集根目录")
    p_prepare.add_argument("--scene_glob", type=str,
                           default="*/CAM_F0/*.jpg",
                           help="场景 glob 模式（相对于 trainval/）")
    p_prepare.add_argument("--output", type=str, default="layer_a.jsonl",
                           help="输出 Layer A JSONL 路径")

    # encode
    p_encode = sub.add_parser(
        "encode",
        help="Layer A → Layer C（调用 Emu3.5 VisionTokenizer 编码未来帧）",
    )
    p_encode.add_argument("--input", type=str, required=True,
                          help="Layer A JSONL 路径")
    p_encode.add_argument("--emu35_path", type=str, required=True,
                          help="Emu3.5-VisionTokenizer 模型路径")
    p_encode.add_argument("--output", type=str, default="layer_c.jsonl",
                          help="输出 Layer C JSONL 路径")
    p_encode.add_argument("--navsim_root", type=str, default=None,
                          help="NavSim 数据集根目录（用于解析相对路径）")
    p_encode.add_argument("--device", type=str, default="cuda:0")
    p_encode.add_argument("--max_records", type=int, default=None,
                          help="最多处理记录数（调试用）")

    # minify
    p_minify = sub.add_parser(
        "minify",
        help="从现有完整 JSONL 剥离多余字段 → Layer C（无需 Emu3.5 模型）",
    )
    p_minify.add_argument("--input", type=str, required=True,
                          help="现有完整 JSONL 路径")
    p_minify.add_argument("--output", type=str, default="layer_c.jsonl",
                          help="输出 Layer C JSONL 路径")

    args = parser.parse_args()

    if args.cmd == "prepare":
        cmd_prepare(args)
    elif args.cmd == "encode":
        cmd_encode(args)
    elif args.cmd == "minify":
        cmd_minify(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
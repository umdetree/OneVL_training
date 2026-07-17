#!/usr/bin/env python3
"""
Emu3.5 VisionTokenizer — Image ⇄ Discrete Visual Token 双向转换工具

提供两个核心功能：
  1. image_to_tokens():  图像 → token IDs 网格 + future_image_tokens 字符串
  2. tokens_to_image():  token IDs 网格 → 重建 PIL Image

支持 Emu3.5-VisionTokenizer (131072 codebook) 和 Emu3-VisionTokenizer (32768)。

用法：
  from emu35_image_tokenize_demo import load_vision_tokenizer, image_to_tokens, tokens_to_image

  vq = load_vision_tokenizer("/path/to/Emu3.5-VisionTokenizer")
  codes_2d, future_str = image_to_tokens(vq, pil_image)
  recon = tokens_to_image(codes_2d, vq)
"""

from __future__ import annotations

import re
from typing import Optional, Tuple

import numpy as np
import torch
from PIL import Image


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_vision_tokenizer(model_root: str, device: str = "cpu") -> torch.nn.Module:
    """加载 Emu3.5/Emu3 VisionTokenizer VQ 模型（encode + decode）。"""
    from transformers import AutoModel

    model = AutoModel.from_pretrained(
        model_root,
        trust_remote_code=True,
        torch_dtype=torch.float32,
    ).eval()
    model = model.to(device)
    return model


def _get_embed_dim(vq_model: torch.nn.Module) -> int:
    """获取 VQ codebook 的 embedding 维度。"""
    return vq_model.config.embed_dim


def _is_emu35(vq_model: torch.nn.Module) -> bool:
    """判断是 Emu3.5（131k codebook）还是 Emu3（32k codebook）。"""
    return vq_model.config.codebook_size >= 131072


# ---------------------------------------------------------------------------
# Image → Tokens (encode)
# ---------------------------------------------------------------------------

def image_to_tokens(
    vq_model: torch.nn.Module,
    image: Image.Image,
    device: Optional[str] = None,
) -> Tuple[torch.Tensor, str]:
    """将 PIL Image 编码为离散 visual token。

    Args:
        vq_model: load_vision_tokenizer 加载的 VQ 模型
        image:    输入图像（RGB PIL Image）
        device:   推理设备，默认跟随模型

    Returns:
        codes:       token ID 网格 [H, W] (long)
        future_str:  Stage 0.5 训练格式的字符串
                     "<|image start|>H*W<|image token|><|visual token ...>...<|image end|>"
    """
    if device is None:
        device = next(vq_model.parameters()).device

    # 预处理：RGB，转为 tensor [1, 3, H, W]
    img_rgb = image.convert("RGB")
    img_h, img_w = img_rgb.height, img_rgb.width
    arr = np.array(img_rgb, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)

    with torch.no_grad():
        raw = vq_model.encode(tensor)

    if _is_emu35(vq_model):
        # Emu3.5 encode returns (quant_embed, None, (None, None, token_ids))
        # token_ids is 1D flat: [N] where N = h * w
        _, _, (_, _, token_ids) = raw
        # Infer grid dimensions from original image size.
        # Emu3.5 encoder downsamples equally in both dimensions.
        n_tokens = token_ids.shape[0]
        spatial_scale = int(round((img_h * img_w / n_tokens) ** 0.5))
        h = img_h // spatial_scale
        w = img_w // spatial_scale
        codes_2d = token_ids.reshape(h, w).cpu()
    else:
        # Emu3 encode returns codes directly: [1, h, w]
        codes_2d = raw.squeeze(0).cpu()

    future_str = grid_to_future_str(codes_2d)
    return codes_2d, future_str


def grid_to_future_str(grid: torch.Tensor) -> str:
    """将 token ID 网格 [H, W] 转为 Stage 0.5 future_image_tokens 字符串。

    格式：<|image start|>H*W<|image token|>
          <|visual token XXXXXX|><|visual token YYYYYY|>...<|extra_200|>\n
          ...（共 H 行）...
          <|image end|>
    """
    h, w = grid.shape
    parts = [f"<|image start|>{h}*{w}<|image token|>"]

    for row in range(h):
        for col in range(w):
            parts.append(f"<|visual token {int(grid[row, col]):06d}|>")
        parts.append("<|extra_200|>\n")

    parts.append("<|image end|>")
    return "".join(parts)


# ---------------------------------------------------------------------------
# Tokens → Image (decode)
# ---------------------------------------------------------------------------

def tokens_to_image(
    grid: torch.Tensor,
    vq_model: torch.nn.Module,
    embed_dim: Optional[int] = None,
    device: Optional[str] = None,
) -> Image.Image:
    """将 token ID 网格 [H, W] 解码回 PIL Image。

    Args:
        grid:      token ID 网格 [H, W]
        vq_model:  VQ 模型
        embed_dim: （Emu3 兼容参数，Emu3.5 自动忽略）
        device:    推理设备

    Returns:
        重建的 RGB PIL Image
    """
    if device is None:
        device = next(vq_model.parameters()).device

    codes = grid.unsqueeze(0).to(device)  # [1, h, w]

    with torch.no_grad():
        if _is_emu35(vq_model):
            # Emu3.5: use decode_code with explicit shape
            h, w = grid.shape
            recon = vq_model.decode_code(codes, shape=(1, h, w, -1))
        else:
            # Emu3: decode directly
            recon = vq_model.decode(codes)

    # 后处理：tensor → PIL
    # Output is [1, 3, H, W] in [-1, 1] or [0, 1] range
    recon = recon.squeeze(0).cpu()  # [3, H, W]
    # Emu3 range is [-1, 1], Emu3.5 might be [0, 1]; normalize to [0, 1]
    if recon.min() < -0.5:
        recon = (recon + 1.0) / 2.0
    recon = torch.clamp(recon, 0.0, 1.0)
    arr = (recon.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return Image.fromarray(arr)


# ---------------------------------------------------------------------------
# Text ↔ Grid (parse / serialize)
# ---------------------------------------------------------------------------

_FUTURE_BLOCK_RE = re.compile(r"<\|image start\|>(.*?)<\|image end\|>", re.DOTALL)


def parse_token_block(block: str) -> torch.Tensor:
    """解析单个 <|image start|>...<|image end|> 块为 token ID 网格。

    格式期望：
        <|image start|>H*W<|image token|>
        <|visual token XXXXXX|>...<|extra_200|>\n  （H 行）
        <|image end|>

    返回 [H, W] long tensor。
    """
    dims_match = re.search(r"<\|image start\|>\s*(\d+)\*(\d+)", block)
    if not dims_match:
        raise ValueError(f"Cannot parse dimensions from block: {block[:80]}...")
    h, w = int(dims_match.group(1)), int(dims_match.group(2))

    token_ids = re.findall(r"<\|visual token (\d{6})\|>", block)
    if len(token_ids) != h * w:
        raise ValueError(
            f"Expected {h}*{w}={h * w} visual tokens, found {len(token_ids)}"
        )

    ids = [int(t) for t in token_ids]
    grid = torch.tensor(ids, dtype=torch.long).reshape(h, w)
    return grid


def extract_future_blocks(future_image_tokens: str) -> list[str]:
    """从 future_image_tokens 字符串中提取所有 <|image start|>...<|image end|> 块。"""
    return _FUTURE_BLOCK_RE.findall(future_image_tokens or "")


# ---------------------------------------------------------------------------
# Batch utility: JSONL 扩展
# ---------------------------------------------------------------------------

def add_future_tokens_to_record(
    record: dict,
    future_images: list[Image.Image],
    vq_model: torch.nn.Module,
    device: Optional[str] = None,
) -> dict:
    """为一条 JSONL record 补充 future_image_tokens 字段。

    Args:
        record:         原始 JSONL record（含 "messages", "images" 等）
        future_images:  未来帧图像列表（PIL Image）
        vq_model:       VQ 模型
        device:         推理设备

    Returns:
        更新后的 record（非破坏性修改，返回新 dict）
    """
    record = {**record}
    blocks = []
    for img in future_images:
        _, future_str = image_to_tokens(vq_model, img, device=device)
        blocks.append(future_str)
    record["future_image_tokens"] = "\n".join(blocks)
    return record


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def cli_main():
    import argparse
    import json
    import os

    parser = argparse.ArgumentParser(
        description="Emu3.5 VisionTokenizer — 图像离散化 & 可视化"
    )
    sub = parser.add_subparsers(dest="cmd")

    enc = sub.add_parser("encode", help="图像 → visual tokens")
    enc.add_argument("image", type=str, help="输入图像路径")
    enc.add_argument("--model_root", type=str, default=None, required=True,
                     help="Emu3.5-VisionTokenizer 本地路径")
    enc.add_argument("--device", type=str, default="cpu")
    enc.add_argument("--out_txt", type=str, default=None,
                     help="输出 token 字符串到文件")

    dec = sub.add_parser("decode", help="visual tokens → 图像")
    dec.add_argument("block_file", type=str,
                     help="包含 <|image start|>...<|image end|> 的文本文件")
    dec.add_argument("--model_root", type=str, default=None, required=True,
                     help="Emu3.5-VisionTokenizer 本地路径")
    dec.add_argument("--device", type=str, default="cpu")
    dec.add_argument("--out_png", type=str, default="recon.png")

    args = parser.parse_args()

    if args.cmd == "encode":
        vq = load_vision_tokenizer(args.model_root, args.device)
        img = Image.open(args.image)
        codes_2d, future_str = image_to_tokens(vq, img, device=args.device)
        print(f"Grid: {codes_2d.shape[0]}x{codes_2d.shape[1]}, "
              f"tokens={codes_2d.numel()}, "
              f"range=[{codes_2d.min().item()}, {codes_2d.max().item()}]")
        if args.out_txt:
            with open(args.out_txt, "w") as f:
                f.write(future_str)
            print(f"Saved to {args.out_txt}")
        else:
            print(f"\n--- future_image_tokens (first 500 chars) ---")
            print(future_str[:500] + ("..." if len(future_str) > 500 else ""))

    elif args.cmd == "decode":
        vq = load_vision_tokenizer(args.model_root, args.device)
        with open(args.block_file) as f:
            text = f.read()
        blocks = extract_future_blocks(text)
        if not blocks:
            print("No <|image start|>...<|image end|> block found!")
            exit(1)
        print(f"Found {len(blocks)} block(s), decoding first one...")
        grid = parse_token_block(blocks[0])
        recon = tokens_to_image(grid, vq, device=args.device)
        recon.save(args.out_png)
        print(f"Saved {args.out_png} ({grid.shape[0]}x{grid.shape[1]} grid"
              f" → {recon.size[0]}x{recon.size[1]} px)")

    else:
        parser.print_help()

def test_example():
    model_path = "/root/.cache/huggingface/hub/models--BAAI--Emu3.5-VisionTokenizer/snapshots/a5c45bdb8084763048e094ae124778bfeca7d5ee/"
    test_image_path = "./image.png"
    device = "cuda:0"
    test_file_path = "./test_tokens.txt"
    vq = load_vision_tokenizer(model_path, device=device)
    # encode into token file
    img = Image.open(test_image_path)
    import time
    start = time.time()
    codes_2d, future_str = image_to_tokens(vq, img, device=device)
    print(f"Grid: {codes_2d.shape[0]}x{codes_2d.shape[1]}, "
          f"tokens={codes_2d.numel()}, "
          f"range=[{codes_2d.min().item()}, {codes_2d.max().item()}]")
    with open(test_file_path, "w") as f:
        f.write(future_str)

    # decode from token file
    with open(test_file_path) as f:
        example_text = f.read()
    grid = parse_token_block(example_text)
    recon = tokens_to_image(grid, vq, device=device)
    recon.save("test_recon.png")
    end = time.time()
    print(f"Collapsed time: {end - start:.2f}s, saved test_recon.png")

if __name__ == "__main__":
    # cli_main()
    test_example()
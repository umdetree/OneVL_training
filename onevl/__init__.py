"""
OneVL — Core modules for Stage 0.5 Visual Aux Decoder Pretrain.

Decoupled from the ms-swift training framework.  Pure functions with
complete type hints, ready for integration into any training pipeline.

Modules:
    constants        — Shared constants (token IDs, grid sizes, etc.)
    token_registry   — Register 131k visual code tokens on a tokenizer
    chunked_loss     — Chunked LM-head + Cross-Entropy loss (ZeRO-3 safe)
    forward          — Stage 0.5 forward pass: ViT → LLM → chunked loss
    patch            — ms-swift adapter (only module that depends on swift)
    emu35_tokenizer  — Emu3.5 VisionTokenizer encode/decode utilities
"""

from .constants import (
    NUM_VISUAL_TOKENS,
    VIS_TOKEN_START,
    VIS_TOKEN_END,
    IMAGE_STRUCTURAL_TOKENS,
    GRID_H,
    GRID_W,
    NUM_FRAMES,
)
from .token_registry import add_visual_tokens_to_tokenizer
from .chunked_loss import chunked_lm_head_ce
from .forward import visual_aux_pretrain_forward
from .patch import patch_model_for_visual_aux_pretrain

__all__ = [
    'NUM_VISUAL_TOKENS',
    'VIS_TOKEN_START',
    'VIS_TOKEN_END',
    'IMAGE_STRUCTURAL_TOKENS',
    'GRID_H',
    'GRID_W',
    'NUM_FRAMES',
    'add_visual_tokens_to_tokenizer',
    'chunked_lm_head_ce',
    'visual_aux_pretrain_forward',
    'patch_model_for_visual_aux_pretrain',
]
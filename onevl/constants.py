"""
Stage 0.5 Visual Aux Decoder Pretrain — Shared Constants.

All model-architecture constants live here so they can be imported
without pulling in any framework dependencies.
"""

from typing import Final

# ---------------------------------------------------------------------------
# Visual token codebook
# ---------------------------------------------------------------------------

NUM_VISUAL_TOKENS: Final[int] = 131072
"""Number of discrete visual code tokens from the Emu3.5 VisionTokenizer."""

VIS_TOKEN_START: Final[int] = 151673
"""Token ID of the first visual code token (``<|visual token 000000|>``)."""

VIS_TOKEN_END: Final[int] = VIS_TOKEN_START + NUM_VISUAL_TOKENS - 1  # 282744
"""Token ID of the last visual code token (``<|visual token 131071|>``)."""

# ---------------------------------------------------------------------------
# Image structural tokens (row markers / separators)
# ---------------------------------------------------------------------------

IMAGE_STRUCTURAL_TOKENS: Final[list[str]] = [
    '<|image start|>',
    '<|image token|>',
    '<|image end|>',
    '<|extra_200|>',
]
"""Structural tokens that delimit visual token rows in the image sequence."""

# ---------------------------------------------------------------------------
# Default spatial grid dimensions (NavSim dataset)
# ---------------------------------------------------------------------------

GRID_H: Final[int] = 12
"""Default grid height (number of visual token rows per frame)."""

GRID_W: Final[int] = 21
"""Default grid width (number of visual token columns per frame)."""

NUM_FRAMES: Final[int] = 2
"""Default number of future frames (0.5 s and 1.0 s)."""
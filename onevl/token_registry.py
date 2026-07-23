"""
Register 131k discrete visual code tokens + image structural tokens
on a HuggingFace tokenizer so that they are encoded as single tokens.

Usage::

    from onevl import add_visual_tokens_to_tokenizer
    add_visual_tokens_to_tokenizer(tokenizer)
"""

from typing import Optional

from transformers import PreTrainedTokenizerBase

from .constants import IMAGE_STRUCTURAL_TOKENS, NUM_VISUAL_TOKENS


def add_visual_tokens_to_tokenizer(
    tokenizer: PreTrainedTokenizerBase,
    verbose: bool = True,
) -> PreTrainedTokenizerBase:
    """Register 131k discrete visual code tokens + image structural tokens.

    The Emu3.5 VisionTokenizer produces 131,072 discrete code IDs.  In the
    dataset these appear as ``<|visual token NNNNNN|>`` strings.  Without
    explicit registration they are split into 13+ subword tokens by the BPE
    tokenizer, which means the aux decoder would predict subword sequences
    rather than individual visual code IDs.

    This function also registers image structural tokens
    (``<|image start|>``, ``<|image token|>``, etc.) that delimit rows in
    the image token raster.

    After calling this, ``tokenizer.encode('<|visual token 000001|>')``
    returns a single token ID instead of 13 fragments.

    Args:
        tokenizer: A HuggingFace tokenizer (e.g. ``processor.tokenizer``).
        verbose:   Whether to log a summary of added tokens.

    Returns:
        The tokenizer (modified in-place).
    """
    existing: set[str] = set(tokenizer.get_vocab().keys())

    # 1. Image structural tokens (row markers / separators)
    structural = [t for t in IMAGE_STRUCTURAL_TOKENS if t not in existing]
    if structural:
        tokenizer.add_tokens(structural, special_tokens=True)

    # 2. 131k visual code tokens
    visual_tokens = [f'<|visual token {i:06d}|>' for i in range(NUM_VISUAL_TOKENS)]
    new_visual = [t for t in visual_tokens if t not in existing]
    if new_visual:
        tokenizer.add_tokens(new_visual, special_tokens=True)
        if verbose:
            print(
                f'[VisualTokens] Added {len(new_visual)} visual code tokens '
                f'(vocab now {len(tokenizer)})'
            )

    return tokenizer
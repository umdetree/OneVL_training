"""
Chunked LM-head + Cross-Entropy loss.

The LM-Head projection is a per-token independent linear transformation.
Chunking produces mathematically identical results to computing all logits
at once, but uses far less peak GPU memory.

This is especially important under DeepSpeed ZeRO-3, where calling
``lm_head(chunk)`` per chunk automatically gathers only the parameters
needed for the current chunk rather than the full vocabulary projection.
"""

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import CrossEntropyLoss


def chunked_lm_head_ce(
    hidden_states: Tensor,       # [1, seq_len, hidden]
    labels: Tensor,               # [1, seq_len], -100 for ignored positions
    lm_head: nn.Linear,           # weight: [vocab_size, hidden]
    chunk_size: int = 512,
) -> Tensor:
    """Apply lm_head in chunks and compute shifted CE loss.

    The LM-Head is applied per-token independently, so chunking produces
    identical results to computing all logits at once but uses far less
    peak memory.  Under DeepSpeed ZeRO-3, each ``lm_head(chunk)`` call
    automatically gathers only the chunk-sized portion of the output
    projection weights.

    Args:
        hidden_states: Last hidden states from the language model.
            Shape ``[1, seq_len, hidden_size]``.
        labels: Target token IDs with ``-100`` for positions to ignore
            (e.g. the ViT prefix).  Shape ``[1, seq_len]``.
        lm_head: The output projection (``nn.Linear`` with
            ``out_features == vocab_size``).
        chunk_size: Number of tokens per chunk.  Smaller values reduce
            peak memory at the cost of more loop iterations.  Default 512.

    Returns:
        Scalar loss tensor (mean-reduced over effective tokens).
    """
    loss_fct = CrossEntropyLoss(reduction='sum')
    shift_labels = labels[0, 1:]  # [seq_len - 1]
    n_tokens = hidden_states.shape[1] - 1
    total_loss = torch.tensor(0.0, device=hidden_states.device)
    total_effective = 0

    for c_start in range(0, n_tokens, chunk_size):
        c_end = min(c_start + chunk_size, n_tokens)
        chunk_hidden = hidden_states[0, c_start:c_end]  # [chunk, hidden]
        chunk_labels = shift_labels[c_start:c_end]       # [chunk]
        chunk_logits = lm_head(chunk_hidden)              # [chunk, vocab]

        effective_mask = chunk_labels != -100
        effective_tokens = effective_mask.sum()
        if effective_tokens > 0:
            eff_logits = chunk_logits[effective_mask].float()
            eff_labels = chunk_labels[effective_mask]
            item_loss = loss_fct(eff_logits, eff_labels)
            total_loss = total_loss + item_loss
            total_effective += effective_tokens.item()

        # Free intermediate logits tensor early
        del chunk_logits

    if total_effective > 0:
        return total_loss / total_effective
    return torch.tensor(0.0, device=hidden_states.device)
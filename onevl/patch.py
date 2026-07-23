"""
ms-swift adapter for Stage 0.5 Visual Aux Decoder Pretrain.

This is the **only module** in the ``onevl`` package that depends on the
ms-swift training framework.  It monkey-patches the model's ``forward``
method to use the pure forward function from ``onevl.forward``.

If you are not using ms-swift, ignore this module and call
``visual_aux_pretrain_forward()`` directly from your own training loop.
"""

from types import MethodType
from typing import Optional

import torch
from transformers import PreTrainedModel, ProcessorMixin
from transformers.modeling_outputs import CausalLMOutputWithPast

from .forward import visual_aux_pretrain_forward
from .token_registry import add_visual_tokens_to_tokenizer


def _make_patched_forward(chunk_size: int = 512):
    """Create a patched forward method with the given chunk_size baked in."""

    def _forward_visual_aux_pretrain(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.Tensor] = None,
        video_grid_thw: Optional[torch.Tensor] = None,
        think_steps: Optional[str] = None,
        future_image_tokens: Optional[str] = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        """Patched forward for Stage 0.5 visual aux decoder pretrain.

        Delegates to ``visual_aux_pretrain_forward()`` from
        ``onevl.forward``, then wraps the result in a
        ``CausalLMOutputWithPast`` for compatibility with ms-swift's
        training loop.
        """
        tokenizer = self._visual_aux_tokenizer

        loss, aux_info = visual_aux_pretrain_forward(
            model=self,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            future_image_tokens=future_image_tokens,
            tokenizer=tokenizer,
            chunk_size=chunk_size,
        )

        if hasattr(self, '_latent_cot_cache'):
            pass  # keep existing cache for downstream tracking

        return CausalLMOutputWithPast(loss=loss)

    return _forward_visual_aux_pretrain


def patch_model_for_visual_aux_pretrain(
    model: PreTrainedModel,
    processor: ProcessorMixin,
    chunk_size: int = 512,
) -> PreTrainedModel:
    """Patch Qwen3-VL in-place for Stage 0.5 visual aux decoder pretrain.

    What this does:
    1. Registers 131k visual code tokens (+ structural tokens) on the
       processor's tokenizer and resizes model embeddings.
    2. Replaces ``model.forward`` with a patched version that calls
       ``visual_aux_pretrain_forward()`` from ``onevl.forward``.

    The ViT should be frozen via ``--freeze_vit true`` in the training
    script so gradients flow only through the LLM.

    Args:
        model: A Qwen3-VL (or compatible) model.
        processor: The corresponding processor (used to access the
            tokenizer).
        chunk_size: Chunk size for the chunked LM-head + CE loss.
            Default 512.

    Returns:
        The model (modified in-place).
    """
    # Register 131k visual tokens on the processor's tokenizer, resize model
    tokenizer = processor.tokenizer if hasattr(processor, 'tokenizer') else processor
    add_visual_tokens_to_tokenizer(tokenizer)
    model.resize_token_embeddings(len(tokenizer))

    # Store tokenizer reference for the patched forward
    model._visual_aux_tokenizer = tokenizer

    # Save original forward for inner LLM call, then patch
    model._origin_forward = model.forward
    model.forward = MethodType(_make_patched_forward(chunk_size=chunk_size), model)
    model._latent_cot_cache = {}

    print(
        f'[Stage0.5] Visual aux decoder pretrain patch applied '
        f'(vocab_size={len(tokenizer)}, chunk_size={chunk_size})'
    )
    return model
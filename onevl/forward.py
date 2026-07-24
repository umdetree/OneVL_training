"""
Stage 0.5 Visual Aux Decoder Pretrain — Pure forward function.

This module contains the core forward logic: given a Qwen3-VL model,
current-frame pixel values, and future-frame discrete visual token strings,
compute the causal LM prediction loss on the future tokens.

The function is a **pure function** — it accepts explicit inputs and returns
``(loss, aux_info)``.  It does NOT depend on any training framework (swift,
DeepSpeed, etc.) and can be called from any context.
"""

from typing import Union

import torch
from torch import Tensor, nn
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from .chunked_loss import chunked_lm_head_ce


def visual_aux_pretrain_forward(
    model: PreTrainedModel,
    pixel_values: Tensor,
    image_grid_thw: Tensor,
    future_image_tokens: Union[str, list[str]],
    tokenizer: PreTrainedTokenizerBase,
    chunk_size: int = 512,
) -> tuple[Tensor, dict]:
    """Stage 0.5 visual aux decoder pretrain — single step forward.

    The model IS the visual aux decoder (single-model design).  This
    function:

    1. Runs ``model.model.get_image_features`` on ``pixel_values`` to obtain visual
       embeddings for the current frame (returns per-image tuple).
    2. Iterates over each sample in the batch, tokenizes
       ``future_image_tokens`` (the discrete visual token strings
       for the next 0.5 s and 1.0 s frames) into target token IDs.
    3. For each sample, builds ``combined_embeds = [vis_embeds, target_embeds]``.
    4. Forwards through the language model (``model.model.language_model``).
    5. Computes chunked LM-head + CE loss on the hidden states per sample,
       then averages across the batch.

    Batch support: ViT processes all images in one forward call
    (``pooler_output`` returns a list of tensors, one per image).
    The LLM forward is called per-sample since each sample has a different
    ``combined_embeds`` sequence length. This matches the pattern used in
    Stage 1/2 ``compute_visual_explain_loss``.

    DeepSpeed ZeRO-3 compatibility:
    - ``model.model.visual`` — ViT parameters are typically small and not
      partitioned; direct call works.
    - ``model.model.language_model`` — LLM forward handles its own
      partitioning internally.
    - ``model.lm_head`` — accessed per-chunk inside ``chunked_lm_head_ce``,
      so only the chunk-sized portion of the output projection is gathered.

    Gradient accumulation compatibility:
    The returned loss is a scalar (mean-reduced over effective tokens).
    Use ``loss = loss / grad_accum_steps; loss.backward()`` in the training
    loop for standard gradient accumulation.

    Args:
        model: A Qwen3-VL (or compatible) model loaded via
            ``Qwen3VLForConditionalGeneration.from_pretrained(...)``.
            Must have ``model.model.visual``, ``model.model.language_model``,
            ``model.get_input_embeddings()``, and ``model.lm_head``.
        pixel_values: Current-frame image tensor after processor.
            Shape ``[1, 3, H, W]`` or ``[B, 3, H, W]``.
        image_grid_thw: Image grid dimensions from processor.
            Shape ``[[1, grid_h, grid_w]]``.
        future_image_tokens: One or more strings containing the future-frame
            discrete visual token sequences in the format produced by
            ``emu35_tokenizer.grid_to_future_str()``.
        tokenizer: Tokenizer that has been extended with visual tokens
            (see ``token_registry.add_visual_tokens_to_tokenizer()``).
        chunk_size: Chunk size for the chunked LM-head + CE loss.
            Smaller values reduce peak memory.  Default 512.

    Returns:
        ``(loss, aux_info)`` where:
        - ``loss`` is a scalar tensor (mean-reduced CE loss).
        - ``aux_info`` is a dict with keys ``n_vis``, ``n_target``,
          ``seq_len``, and ``loss_value`` for debugging.
    """
    # 1. Run ViT to get visual embeddings
    #    Use get_image_features which returns per-image split pooler_output
    #    (model.model.visual returns a flat tensor, get_image_features splits it)
    vision_output = model.model.get_image_features(
        pixel_values=pixel_values,
        image_grid_thw=image_grid_thw,
        return_dict=True,
    )
    vis_embeds_list = vision_output.pooler_output  # tuple of [n_patches_i, hidden]
    batch_size = len(vis_embeds_list)

    # 2. LLM forward — per-sample loop (each sample has different seq length)
    total_loss = torch.tensor(0.0, device=pixel_values.device)
    num_items = 0

    for b in range(batch_size):
        vis_embeds = vis_embeds_list[b]  # [n_patches, hidden]
        n_vis = vis_embeds.shape[0]

        # 2a. Tokenize future_image_tokens → target IDs
        ft = (
            future_image_tokens[b]
            if isinstance(future_image_tokens, list)
            else future_image_tokens
        )
        target_ids = tokenizer.encode(str(ft), add_special_tokens=False)
        if tokenizer.eos_token_id is not None:
            target_ids = target_ids + [tokenizer.eos_token_id]
        target_tensor = torch.tensor(
            target_ids, device=vis_embeds.device, dtype=torch.long,
        )

        # 2b. Build combined_embeds = [vis_embeds, target_embeds]
        target_embeds = model.get_input_embeddings()(target_tensor)
        combined_embeds = torch.cat([vis_embeds, target_embeds], dim=0).unsqueeze(0)
        seq_len = combined_embeds.shape[1]
        attn_mask = torch.ones(1, seq_len, dtype=torch.long, device=vis_embeds.device)

        # 2c. Inner LLM forward via language_model (raw Qwen3VLTextModel)
        lm = model.model.language_model
        outputs = lm(
            input_ids=None,
            attention_mask=attn_mask,
            inputs_embeds=combined_embeds,
            use_cache=False,
            output_hidden_states=True,
        )
        last_hidden = outputs.hidden_states[-1]  # [1, seq_len, hidden]

        # 2d. Build labels: ViT prefix = -100, target part = ground truth
        labels = torch.full(
            (1, seq_len), -100,
            dtype=torch.long, device=vis_embeds.device,
        )
        labels[0, n_vis:] = target_tensor

        # 2e. Chunked LM-head + CE
        item_loss = chunked_lm_head_ce(
            last_hidden, labels, model.lm_head, chunk_size=chunk_size,
        )
        total_loss = total_loss + item_loss
        num_items += 1

    total_loss = total_loss / num_items if num_items > 0 else total_loss

    # 7. Return loss + debug info
    aux_info = {
        'batch_size': batch_size,
        'n_vis': n_vis,
        'n_target': len(target_ids),
        'seq_len': seq_len,
        'loss_value': total_loss.item(),
    }

    return total_loss, aux_info
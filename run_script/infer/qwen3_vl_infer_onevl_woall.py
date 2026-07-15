"""
OneVL (Latent CoT) inference script for Qwen3-VL.

Supports:
- Standard latent-token inference (generate answer from latent prefix)
- Optional aux decoder explain: decode the latent hidden states into
  explicit reasoning text using the trained auxiliary decoder
- Optional visual aux decoder explain: decode latent states into
  future visual tokens

Hyperparameters are passed via argparse flags.

The trained checkpoint stores all weights (base model + aux decoder + projections)
in the same safetensors files. This script extracts sub-module weights by prefix.
"""
import sys
import os
import json
import ast
import argparse
import glob
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from PIL import Image
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor, AutoConfig
from safetensors.torch import load_file


# ---------------------------------------------------------------------------
# GT / waypoint parsing for coordinate prefill (--prefix_k)
# ---------------------------------------------------------------------------

def parse_gt_waypoints(gt_str):
    """Parse GT or assistant trajectory string into a list of points (each point is a list).

    Accepts forms like:
      - ``[582, 963], [573, 942], ...`` (trainfmt ``GT`` field)
      - ``[[582, 963], [573, 942], ...]`` (full list-of-lists)
      - ``[x, y, heading], ...`` (navsim-style triples)
    """
    if not gt_str or not isinstance(gt_str, str):
        return []
    s = gt_str.strip()
    if not s:
        return []
    try:
        data = ast.literal_eval(s)
    except (SyntaxError, ValueError):
        try:
            data = ast.literal_eval('[' + s + ']')
        except (SyntaxError, ValueError):
            return []
    if not data:
        return []
    if isinstance(data[0], (int, float)):
        return [list(data)]
    return [list(p) for p in data]


def format_gt_prefix_points(points):
    """Format first k points as continuation after ``<answer>[`` (opens with ``[x, y], ...``)."""
    parts = []
    for p in points:
        inner = ", ".join(str(int(x)) if isinstance(x, float) and x == int(x) else str(x) for x in p)
        parts.append(f"[{inner}]")
    return ", ".join(parts) + ","


# ---------------------------------------------------------------------------
# Utility functions for finding latent positions in original-vocab mode
# (ported from swift/model/models/latent_cot.py to keep inference self-contained)
# ---------------------------------------------------------------------------

def _get_latent_pattern_ids(tokenizer):
    """Pre-compute token IDs for pattern-matching latent markers in original vocab mode."""
    def _single_id(text):
        enc = tokenizer.encode(text, add_special_tokens=False)
        return enc[0] if len(enc) == 1 else None
    return {
        'latent_keyword_id': _single_id('latent'),
        'pipe_id': _single_id('|'),
        'vis_suffix_id': _single_id('-vis'),
    }


def _get_marker_component_ids(tokenizer):
    """Get the set of token IDs that form latent marker strings."""
    texts = ['<', '>', '|', '><', 'latent', 'start', 'end', '-lat', 'ent', '-vis']
    ids = set()
    for text in texts:
        enc = tokenizer.encode(text, add_special_tokens=False)
        if len(enc) == 1:
            ids.add(enc[0])
    return ids


def _find_latent_keyword_positions(ids_list, latent_keyword_id, pipe_id):
    """Find keyword positions of ``<|latent|>`` via the ``| latent |`` pattern."""
    positions = []
    n = len(ids_list)
    for i in range(1, n - 1):
        if (ids_list[i] == latent_keyword_id
                and ids_list[i - 1] == pipe_id
                and ids_list[i + 1] == pipe_id):
            positions.append(i)
    return positions


def _find_visual_latent_keyword_positions(ids_list, latent_keyword_id, pipe_id, vis_suffix_id):
    """Find keyword positions of ``<|latent-vis|>`` via the ``| latent -vis`` pattern."""
    if vis_suffix_id is None:
        return []
    positions = []
    n = len(ids_list)
    for i in range(1, n - 1):
        if (ids_list[i] == latent_keyword_id
                and ids_list[i - 1] == pipe_id
                and ids_list[i + 1] == vis_suffix_id):
            positions.append(i)
    return positions


def _find_text_latent_block_start(ids_list, pipe_id, vis_suffix_id, tokenizer):
    """Return the first index of the text latent block ``<|start-latent|>``."""
    def _first_id(text):
        enc = tokenizer.encode(text, add_special_tokens=False)
        return enc[0] if len(enc) == 1 else None
    start_id = _first_id('start')
    neglat_id = _first_id('-lat')
    ent_id = _first_id('ent')
    if start_id is None or neglat_id is None or ent_id is None:
        return len(ids_list)
    n = len(ids_list)
    for i in range(1, n - 4):
        if (ids_list[i] == pipe_id
                and ids_list[i + 1] == start_id
                and ids_list[i + 2] == neglat_id
                and ids_list[i + 3] == ent_id
                and ids_list[i + 4] == pipe_id
                and ids_list[i - 1] != vis_suffix_id):
            return i
    return len(ids_list)


def _expand_keyword_positions_with_stop(ids_list, keyword_positions, marker_component_ids, stop_before):
    """Expand from each keyword position through contiguous marker tokens."""
    stop_set = set(stop_before)
    all_positions = []
    used = set()
    for kw_pos in keyword_positions:
        start = kw_pos
        while (start > 0
               and (start - 1) not in stop_set
               and ids_list[start - 1] in marker_component_ids
               and (start - 1) not in used):
            start -= 1
        end = kw_pos
        n = len(ids_list)
        while (end < n - 1
               and (end + 1) not in stop_set
               and ids_list[end + 1] in marker_component_ids
               and (end + 1) not in used):
            end += 1
        for p in range(start, end + 1):
            if p not in used:
                all_positions.append(p)
                used.add(p)
    return all_positions


def compute_inference_latent_positions(
    input_ids_single, tokenizer,
    use_original_vocab=False, use_all_subtokens=False,
    use_separate_visual_latent_tokens=False,
    pattern_ids=None, marker_component_ids=None,
):
    """Compute text and visual latent positions for a single input sequence.

    Returns (text_positions, visual_positions).
    """
    if use_original_vocab:
        ids_list = (input_ids_single.tolist()
                    if hasattr(input_ids_single, 'tolist') else input_ids_single)
        lkw = pattern_ids['latent_keyword_id']
        pipe = pattern_ids['pipe_id']
        vis_suffix_id = pattern_ids.get('vis_suffix_id')

        if use_separate_visual_latent_tokens:
            text_kw = _find_latent_keyword_positions(ids_list, lkw, pipe)
            vis_kw = (_find_visual_latent_keyword_positions(
                ids_list, lkw, pipe, vis_suffix_id) if vis_suffix_id else [])

            if use_all_subtokens:
                text_block_start = _find_text_latent_block_start(
                    ids_list, pipe, vis_suffix_id, tokenizer)
                stop_txt = set(text_kw)
                vis_pos_full = (_expand_keyword_positions_with_stop(
                    ids_list, vis_kw, marker_component_ids, stop_txt) if vis_kw else [])
                vis_pos = [p for p in vis_pos_full if p < text_block_start]
                text_pos = _expand_keyword_positions_with_stop(
                    ids_list, text_kw, marker_component_ids, vis_pos)
                text_pos = [p for p in text_pos if p >= text_block_start]
                return text_pos, vis_pos
            else:
                return text_kw, vis_kw

        elif use_all_subtokens:
            kw_positions = _find_latent_keyword_positions(ids_list, lkw, pipe)
            if not kw_positions:
                return [], []
            all_positions = []
            used = set()
            for kw_pos in kw_positions:
                start = kw_pos
                while (start > 0
                       and ids_list[start - 1] in marker_component_ids
                       and (start - 1) not in used):
                    start -= 1
                end = kw_pos
                while (end < len(ids_list) - 1
                       and ids_list[end + 1] in marker_component_ids
                       and (end + 1) not in used):
                    end += 1
                for p in range(start, end + 1):
                    if p not in used:
                        all_positions.append(p)
                        used.add(p)
            return all_positions, all_positions
        else:
            positions = _find_latent_keyword_positions(ids_list, lkw, pipe)
            return positions, positions
    else:
        latent_token_id = tokenizer.convert_tokens_to_ids('<|latent|>')
        positions = (input_ids_single == latent_token_id).nonzero(
            as_tuple=True)[0].tolist()
        if use_separate_visual_latent_tokens:
            vis_token_id = tokenizer.convert_tokens_to_ids('<|latent-vis|>')
            vis_positions = (input_ids_single == vis_token_id).nonzero(
                as_tuple=True)[0].tolist()
            return positions, vis_positions
        return positions, positions


# ---------------------------------------------------------------------------
# Checkpoint loading utilities
# ---------------------------------------------------------------------------

def collect_state_dict_from_safetensors(ckpt_dir, prefix):
    """Load weight tensors matching a given prefix from all safetensors in ckpt_dir."""
    result = {}
    for sf in sorted(glob.glob(os.path.join(ckpt_dir, '*.safetensors'))):
        sd = load_file(sf)
        for k, v in sd.items():
            if k.startswith(prefix):
                result[k[len(prefix):]] = v
    return result


def build_aux_decoder_from_checkpoint(ckpt_dir, prefix, aux_base_model_path, device, dtype):
    """Build an aux decoder model and load its weights from the checkpoint."""
    config = AutoConfig.from_pretrained(aux_base_model_path, trust_remote_code=True)
    model_type = getattr(config, 'model_type', '')
    if 'qwen3_vl' in model_type:
        from transformers import Qwen3VLForConditionalGeneration as Cls
    elif 'qwen2_vl' in model_type:
        from transformers import Qwen2VLForConditionalGeneration as Cls
    else:
        from transformers import AutoModelForCausalLM as Cls

    print(f"[INFO] Building aux decoder architecture from {aux_base_model_path}")
    model = Cls.from_pretrained(aux_base_model_path, dtype=dtype, trust_remote_code=True)

    sd = collect_state_dict_from_safetensors(ckpt_dir, prefix)
    if sd:
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if missing:
            resolved = []
            unresolved = list(missing)
            # Handle tie_word_embeddings: lm_head.weight shares embed_tokens.weight
            lm_head_missing = [k for k in missing if 'lm_head.weight' in k]
            if lm_head_missing:
                embed_key = 'model.language_model.embed_tokens.weight'
                if embed_key in sd or (hasattr(model, 'lm_head') and hasattr(model, 'model')):
                    model.lm_head.weight = model.model.language_model.embed_tokens.weight
                    resolved.extend(lm_head_missing)
                    unresolved = [k for k in unresolved if k not in lm_head_missing]
            print(f"[INFO] Loaded {len(sd)} weights with prefix '{prefix}' "
                  f"(tied lm_head via weight_sharing, all weights OK)")
            if unresolved:
                print(f"[WARN] Unresolved missing keys: {unresolved}")
        else:
            print(f"[INFO] Loaded {len(sd)} weights with prefix '{prefix}' (all weights matched)")
        if unexpected:
            print(f"[WARN] Unexpected keys: {unexpected}")
    else:
        print(f"[WARN] No weights found with prefix '{prefix}' in {ckpt_dir}")

    model.to(device).eval()
    return model


def build_projection_from_checkpoint(ckpt_dir, prefix, in_dim, out_dim, device, dtype):
    """Build the latent projection and load its weights from the checkpoint."""
    proj = nn.Sequential(
        nn.Linear(in_dim, in_dim),
        nn.GELU(),
        nn.Linear(in_dim, out_dim),
        nn.LayerNorm(out_dim),
    )
    sd = collect_state_dict_from_safetensors(ckpt_dir, prefix)
    if sd:
        proj.load_state_dict(sd)
        print(f"[INFO] Loaded projection weights with prefix '{prefix}' ({len(sd)} tensors)")
    else:
        print(f"[WARN] No projection weights found with prefix '{prefix}', using random init")
    proj.to(device=device, dtype=dtype).eval()
    return proj


def get_aux_input_embeddings(aux_decoder):
    if hasattr(aux_decoder, 'model') and hasattr(aux_decoder.model, 'get_input_embeddings'):
        return aux_decoder.model.get_input_embeddings()
    return aux_decoder.get_input_embeddings()


def call_aux_decoder_lm(
    aux_decoder, inputs_embeds, use_cache=False, past_key_values=None,
):
    """Call the aux decoder's language model directly, bypassing the VL wrapper.

    Qwen3VLForConditionalGeneration.forward -> Qwen3VLModel.forward calls
    get_rope_index(input_ids, ...) which crashes when input_ids is None.
    By calling the inner language_model + lm_head directly, we avoid this.
    For non-VL models (AutoModelForCausalLM), we fall back to the normal call.

    Returns:
        (logits, past_key_values): logits (B, seq, vocab); past_key_values is
        None when use_cache is False.
    """
    if (hasattr(aux_decoder, 'model')
            and hasattr(aux_decoder.model, 'language_model')
            and hasattr(aux_decoder, 'lm_head')):
        lm_out = aux_decoder.model.language_model(
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            past_key_values=past_key_values,
        )
        hidden = (lm_out.last_hidden_state if hasattr(lm_out, 'last_hidden_state')
                  else lm_out[0])
        logits = aux_decoder.lm_head(hidden)
        new_past = (lm_out.past_key_values if use_cache and hasattr(lm_out, 'past_key_values')
                    else None)
        return logits, new_past
    out = aux_decoder(
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        past_key_values=past_key_values,
    )
    logits = out.logits if hasattr(out, 'logits') else out[0]
    new_past = (out.past_key_values if use_cache and hasattr(out, 'past_key_values')
                else None)
    return logits, new_past


def extract_visual_embeds(student_embeds, input_ids, image_token_id, video_token_id=None):
    vis_mask = (input_ids == image_token_id)
    if video_token_id is not None:
        vis_mask = vis_mask | (input_ids == video_token_id)
    if not vis_mask.any():
        return None
    return student_embeds[vis_mask]


@torch.no_grad()
def decode_latent_with_aux(
    model, aux_decoder, latent_proj, input_ids, hidden_states,
    processor, c_thought, device,
    text_positions_list=None,
    use_visual_condition=False, image_token_id=None, video_token_id=None,
    max_explain_tokens=512,
    vit_embeds=None,
):
    """Use the auxiliary decoder to generate explicit reasoning from latent hidden states.

    Args:
        text_positions_list: pre-computed list of text latent positions per batch element.
            When provided, these positions are used directly instead of searching for
            <|latent|> token IDs (required for original-vocab / all-subtokens mode).
    """
    tokenizer = processor.tokenizer if hasattr(processor, 'tokenizer') else processor
    latent_token_id = tokenizer.convert_tokens_to_ids('<|latent|>')
    last_hidden = hidden_states[-1]

    batch_size = input_ids.size(0)
    results = []

    aux_embedding = get_aux_input_embeddings(aux_decoder)

    for b in range(batch_size):
        if text_positions_list is not None:
            positions = text_positions_list[b]
        else:
            positions = (input_ids[b] == latent_token_id).nonzero(as_tuple=True)[0].tolist()
        if not positions:
            results.append("")
            continue

        latent_embeds = last_hidden[b, positions, :]
        if latent_proj is not None:
            latent_embeds = latent_proj(latent_embeds)

        parts = []
        if use_visual_condition and image_token_id is not None:
            if vit_embeds is not None:
                student_embeds_b = vit_embeds[b]
            else:
                embed_fn = (model.model.get_input_embeddings()
                            if hasattr(model, 'model') else model.get_input_embeddings())
                student_embeds_b = embed_fn(input_ids[b])
            vis_embeds = extract_visual_embeds(
                student_embeds_b, input_ids[b], image_token_id, video_token_id)
            if vis_embeds is not None:
                parts.append(vis_embeds)
        parts.append(latent_embeds)
        combined = torch.cat(parts, dim=0).unsqueeze(0)

        print(f"  [TextAux] batch={b}, n_positions={len(positions)}, "
              f"vis_cond={use_visual_condition}, "
              f"vit_embeds={'hook' if vit_embeds is not None else 'placeholder'}, "
              f"combined_shape={combined.shape}")

        generated_ids = []
        past_kv = None
        cur_embeds = combined
        for _ in range(max_explain_tokens):
            logits, past_kv = call_aux_decoder_lm(
                aux_decoder, cur_embeds, use_cache=True, past_key_values=past_kv)
            next_id = logits[:, -1, :].argmax(dim=-1)
            generated_ids.append(next_id.item())
            if next_id.item() == tokenizer.eos_token_id:
                break
            next_embed = aux_embedding(next_id).unsqueeze(1)
            cur_embeds = next_embed

        text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        results.append(text)

    return results


@torch.no_grad()
def decode_latent_with_visual_aux(
    model, visual_aux_decoder, visual_latent_proj, input_ids, hidden_states,
    processor, c_thought_visual, device,
    visual_positions_list=None,
    use_visual_condition=False, image_token_id=None, video_token_id=None,
    max_visual_tokens=512,
    vit_embeds=None,
):
    """Use the visual auxiliary decoder to generate future visual tokens from latent states.

    Matches the training implementation in compute_visual_explain_loss:
    - Uses visual latent positions (not text latent positions) when separated
    - Directly concatenates latent embeddings (NO pooling/mean)
    - Projects with visual_latent_proj
    - Optionally prepends ViT embedding condition (visual_aux_visual_condition)

    Args:
        visual_positions_list: pre-computed list of visual latent positions per batch
            element. When provided, uses these directly instead of searching for
            <|latent|> token IDs.
    """
    tokenizer = processor.tokenizer if hasattr(processor, 'tokenizer') else processor
    last_hidden = hidden_states[-1]

    vis_aux_tokenizer = None
    try:
        from transformers import AutoTokenizer
        base_path = getattr(visual_aux_decoder.config, '_name_or_path', None)
        if base_path:
            vis_aux_tokenizer = AutoTokenizer.from_pretrained(base_path, trust_remote_code=True)
    except Exception:
        pass
    if vis_aux_tokenizer is None:
        vis_aux_tokenizer = tokenizer

    batch_size = input_ids.size(0)
    results = []
    aux_embedding = get_aux_input_embeddings(visual_aux_decoder)

    for b in range(batch_size):
        if visual_positions_list is not None:
            positions = visual_positions_list[b]
        else:
            latent_token_id = tokenizer.convert_tokens_to_ids('<|latent|>')
            positions = (input_ids[b] == latent_token_id).nonzero(as_tuple=True)[0].tolist()
        if not positions:
            results.append("")
            continue

        latent_embeds = last_hidden[b, positions, :]

        if visual_latent_proj is not None:
            latent_embeds = visual_latent_proj(latent_embeds)

        parts = []
        n_vis = 0
        if use_visual_condition and image_token_id is not None:
            if vit_embeds is not None:
                student_embeds_b = vit_embeds[b]
            else:
                embed_fn = (model.model.get_input_embeddings()
                            if hasattr(model, 'model') else model.get_input_embeddings())
                student_embeds_b = embed_fn(input_ids[b])
            vis_embeds = extract_visual_embeds(
                student_embeds_b, input_ids[b], image_token_id, video_token_id)
            if vis_embeds is not None:
                n_vis = vis_embeds.shape[0]
                parts.append(vis_embeds)
        parts.append(latent_embeds)
        combined = torch.cat(parts, dim=0).unsqueeze(0)

        print(f"  [VisualAux] batch={b}, n_positions={len(positions)}, "
              f"n_vis={n_vis}, vis_cond={use_visual_condition}, "
              f"vit_embeds={'hook' if vit_embeds is not None else 'placeholder'}, "
              f"combined_shape={combined.shape}")

        generated_ids = []
        past_kv = None
        cur_embeds = combined
        eos_id = vis_aux_tokenizer.eos_token_id
        for _ in range(max_visual_tokens):
            logits, past_kv = call_aux_decoder_lm(
                visual_aux_decoder, cur_embeds, use_cache=True, past_key_values=past_kv)
            next_id = logits[:, -1, :].argmax(dim=-1)
            generated_ids.append(next_id.item())
            if next_id.item() == eos_id:
                break
            next_embed = aux_embedding(next_id).unsqueeze(1)
            cur_embeds = next_embed

        text = vis_aux_tokenizer.decode(generated_ids, skip_special_tokens=True)
        results.append(text)

    return results


def main():
    parser = argparse.ArgumentParser(description="OneVL (Latent CoT) inference for Qwen3-VL")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the trained checkpoint (contains base + aux weights)")
    parser.add_argument("--test_set_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--num_latent", type=int, default=6,
                        help="Number of latent tokens in the prefix")
    parser.add_argument("--num_latent-vis", type=int, default=0,
                        help="Number of vis latent tokens in the prefix")
    parser.add_argument("--max_new_tokens", type=int, default=1024)

    parser.add_argument("--decoder_explain", action="store_true",
                        help="Enable text aux decoder to explain latent reasoning")
    parser.add_argument("--aux_model_path", type=str, default=None,
                        help="Base architecture path for aux decoder "
                             "(e.g. Qwen3-VL-2B-Instruct-latent). "
                             "Weights are loaded from the main checkpoint.")
    parser.add_argument("--aux_visual_condition", action="store_true",
                        help="Condition aux decoder on visual tokens")
    parser.add_argument("--c_thought", type=int, default=6,
                        help="Latent tokens per thought group for aux decoder")
    parser.add_argument("--max_explain_tokens", type=int, default=512)

    parser.add_argument("--visual_decoder_explain", action="store_true",
                        help="Enable visual aux decoder to decode future visual tokens")
    parser.add_argument("--visual_aux_model_path", type=str, default=None,
                        help="Base architecture path for visual aux decoder. "
                             "Weights are loaded from the main checkpoint.")
    parser.add_argument("--visual_aux_visual_condition", action="store_true",
                        help="Condition visual aux decoder on ViT embeddings "
                             "(separate from --aux_visual_condition)")
    parser.add_argument("--c_thought_visual", type=int, default=6)
    parser.add_argument("--max_visual_tokens", type=int, default=1024)

    parser.add_argument("--use_original_vocab", action="store_true",
                        help="Use original vocab mode (pattern matching for latent positions)")
    parser.add_argument("--use_all_subtokens", action="store_true",
                        help="Use all sub-tokens of each latent marker (not just keyword)")
    parser.add_argument("--use_separate_visual_latent_tokens", action="store_true",
                        help="Separate visual (<|latent-vis|>) and text (<|latent|>) "
                             "latent positions for their respective decoders")
    parser.add_argument("--add_assistant_prefix", action="store_true",
                        help="Add assistant prefix to the input text")
    parser.add_argument("--prefix_k", type=int, default=0,
                        help="If >0, prefill the first K waypoints from GT after <answer>[ "
                             "(e.g. ``[582, 963], [573, 942], ...``). Default 0 = disabled.")

    args = parser.parse_args()
    device = args.device
    dtype = torch.bfloat16

    # ---- Load main model (only base model weights are consumed) ----
    print(f"[INFO] Loading base model from {args.model_path}")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_path, dtype=dtype, trust_remote_code=True)
    model.to(device).eval()
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)

    MAX_IMAGE_SIZE = 1792
    processor.image_processor.max_pixels = MAX_IMAGE_SIZE * MAX_IMAGE_SIZE
    processor.image_processor.size["longest_edge"] = MAX_IMAGE_SIZE * MAX_IMAGE_SIZE
    print(f"[INFO] image_processor.size = {processor.image_processor.size}")

    image_token_id = getattr(model.config, 'image_token_id', None)
    video_token_id = getattr(model.config, 'video_token_id', None)

    # ---- Pre-compute pattern IDs for original vocab mode ----
    tokenizer = processor.tokenizer if hasattr(processor, 'tokenizer') else processor
    pattern_ids = None
    marker_component_ids = None
    if args.use_original_vocab:
        pattern_ids = _get_latent_pattern_ids(tokenizer)
        marker_component_ids = _get_marker_component_ids(tokenizer)
        print(f"[INFO] Original vocab mode: pattern_ids={pattern_ids}")
    print(f"[INFO] use_original_vocab={args.use_original_vocab}, "
          f"use_all_subtokens={args.use_all_subtokens}, "
          f"use_separate_visual_latent_tokens={args.use_separate_visual_latent_tokens}")
    print(f"[INFO] aux_visual_condition={args.aux_visual_condition}, "
          f"visual_aux_visual_condition={args.visual_aux_visual_condition}")

    # ---- Resolve hidden sizes ----
    base_hidden = (model.config.text_config.hidden_size
                   if hasattr(model.config, 'text_config')
                   else model.config.hidden_size)

    # ---- Build latent prefix ----
    if args.num_latent_vis > 0:
        latent_block = "<<"*55 + "<|end-latent|><answer>["
    else:
        latent_block = "<|start-latent|>" + "<|latent|>" * args.num_latent + "<|end-latent|><answer>["
    if args.add_assistant_prefix:
        assistant_prefix = latent_block
    else:
        assistant_prefix = ""

    print(f"[INFO] assistant_prefix = {repr(assistant_prefix)}")

    # ---- Load aux decoder + projection from checkpoint ----
    aux_decoder = None
    latent_proj = None
    if args.decoder_explain:
        if not args.aux_model_path:
            raise ValueError("--aux_model_path required when --decoder_explain is set")

        aux_decoder = build_aux_decoder_from_checkpoint(
            args.model_path, '_latent_cot_aux_decoder.',
            args.aux_model_path, device, dtype)

        aux_cfg = AutoConfig.from_pretrained(args.aux_model_path, trust_remote_code=True)
        aux_hidden = (aux_cfg.text_config.hidden_size
                      if hasattr(aux_cfg, 'text_config') else aux_cfg.hidden_size)

        latent_proj = build_projection_from_checkpoint(
            args.model_path, '_latent_cot_latent_proj.',
            base_hidden, aux_hidden, device, dtype)

    visual_aux_decoder = None
    visual_latent_proj = None
    if args.visual_decoder_explain:
        if not args.visual_aux_model_path:
            raise ValueError("--visual_aux_model_path required when --visual_decoder_explain")

        visual_aux_decoder = build_aux_decoder_from_checkpoint(
            args.model_path, '_latent_cot_visual_aux_decoder.',
            args.visual_aux_model_path, device, dtype)

        vis_cfg = AutoConfig.from_pretrained(args.visual_aux_model_path, trust_remote_code=True)
        vis_hidden = (vis_cfg.text_config.hidden_size
                      if hasattr(vis_cfg, 'text_config') else vis_cfg.hidden_size)

        visual_latent_proj = build_projection_from_checkpoint(
            args.model_path, '_latent_cot_visual_latent_proj.',
            base_hidden, vis_hidden, device, dtype)

    # ---- Load test set ----
    with open(args.test_set_path, 'r') as f:
        test_set = json.load(f)
    print(f"[INFO] Loaded {len(test_set)} test samples from {args.test_set_path}")

    # ---- Inference loop ----
    output_list = []
    need_hidden = (aux_decoder is not None or visual_aux_decoder is not None)

    for idx, item in enumerate(tqdm(test_set, desc="Inference")):
        output_dict = {}

        prompt = item["messages"][0]["content"].replace("<image>", "")
        test_image_path = item["images"][0]

        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": test_image_path},
                {"type": "text", "text": prompt},
            ],
        }]

        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        text += assistant_prefix

        if args.prefix_k > 0:
            gt_src = item.get("GT")
            if not gt_src and len(item.get("messages", [])) > 1:
                gt_src = item["messages"][1].get("content", "")
            pts = parse_gt_waypoints(gt_src)
            if pts:
                k = min(args.prefix_k, len(pts))
                prefix_piece = format_gt_prefix_points(pts[:k])
                text += prefix_piece

        print(f"  [prefix_k={args.prefix_k}] text: {text}")

        try:
            img = Image.open(test_image_path).convert("RGB")
        except Exception as e:
            print(f"[WARN] Skipping sample {idx}: cannot open image {test_image_path}: {e}")
            continue

        inputs = processor(
            text=[text], images=[img], return_tensors="pt", padding=True
        ).to(device)

        vit_embeds = None
        if need_hidden:
            _captured = {}
            def _capture_vit_hook(module, args, kwargs):
                ie = kwargs.get('inputs_embeds')
                if ie is not None:
                    _captured['embeds'] = ie.detach()
                return None
            _hook = model.model.language_model.register_forward_pre_hook(
                _capture_vit_hook, with_kwargs=True)

            fwd_out = model(
                **inputs,
                output_hidden_states=True,
                return_dict=True,
            )
            _hook.remove()
            hidden_states = fwd_out.hidden_states
            vit_embeds = _captured.get('embeds')

            batch_size_cur = inputs['input_ids'].size(0)
            text_positions_list = []
            visual_positions_list = []
            for b in range(batch_size_cur):
                txt_pos, vis_pos = compute_inference_latent_positions(
                    inputs['input_ids'][b], tokenizer,
                    use_original_vocab=args.use_original_vocab,
                    use_all_subtokens=args.use_all_subtokens,
                    use_separate_visual_latent_tokens=args.use_separate_visual_latent_tokens,
                    pattern_ids=pattern_ids,
                    marker_component_ids=marker_component_ids,
                )
                text_positions_list.append(txt_pos)
                visual_positions_list.append(vis_pos)

            if idx < 3:
                print(f"  [Positions] text={[len(p) for p in text_positions_list]}, "
                      f"visual={[len(p) for p in visual_positions_list]}")
                for b in range(batch_size_cur):
                    ids_b = inputs['input_ids'][b]
                    if text_positions_list[b]:
                        txt_ids = ids_b[text_positions_list[b]].tolist()
                        txt_decoded = tokenizer.decode(txt_ids, skip_special_tokens=False)
                        print(f"  [Debug b={b}] text_latent positions={text_positions_list[b]}")
                        print(f"  [Debug b={b}] text_latent token_ids={txt_ids}")
                        print(f"  [Debug b={b}] text_latent decoded='{txt_decoded}'")
                    if visual_positions_list[b]:
                        vis_ids = ids_b[visual_positions_list[b]].tolist()
                        vis_decoded = tokenizer.decode(vis_ids, skip_special_tokens=False)
                        print(f"  [Debug b={b}] vis_latent  positions={visual_positions_list[b]}")
                        print(f"  [Debug b={b}] vis_latent  token_ids={vis_ids}")
                        print(f"  [Debug b={b}] vis_latent  decoded='{vis_decoded}'")

            if idx < 3 and vit_embeds is not None and image_token_id is not None:
                _vis_mask = (inputs['input_ids'][0] == image_token_id)
                if _vis_mask.any():
                    _vit_at_img = vit_embeds[0][_vis_mask]
                    _embed_fn = model.model.get_input_embeddings()
                    _placeholder_at_img = _embed_fn(inputs['input_ids'][0])[_vis_mask]
                    print(f"  [Debug ViT vs Placeholder] n_img_tokens={_vis_mask.sum().item()}")
                    print(f"    ViT embeds std across positions: "
                          f"{_vit_at_img.float().std(dim=0).mean().item():.6f}")
                    print(f"    Placeholder std across positions: "
                          f"{_placeholder_at_img.float().std(dim=0).mean().item():.6f}")
                    print(f"    ViT embeds norm (first 3): "
                          f"{_vit_at_img[:3].float().norm(dim=-1).tolist()}")
                    print(f"    Placeholder norm (first 3): "
                          f"{_placeholder_at_img[:3].float().norm(dim=-1).tolist()}")
                    print(f"    Are ViT embeds diverse (std > 1e-6)? "
                          f"{_vit_at_img.float().std(dim=0).mean().item() > 1e-6}")

            if aux_decoder is not None:
                explains = decode_latent_with_aux(
                    model, aux_decoder, latent_proj,
                    inputs['input_ids'], hidden_states, processor,
                    args.c_thought, device,
                    text_positions_list=text_positions_list,
                    use_visual_condition=args.aux_visual_condition,
                    image_token_id=image_token_id,
                    video_token_id=video_token_id,
                    max_explain_tokens=args.max_explain_tokens,
                    vit_embeds=vit_embeds,
                )
                if explains and explains[0]:
                    output_dict["decoder_explain"] = explains[0]

            if visual_aux_decoder is not None:
                vis_explains = decode_latent_with_visual_aux(
                    model, visual_aux_decoder, visual_latent_proj,
                    inputs['input_ids'], hidden_states, processor,
                    args.c_thought_visual, device,
                    visual_positions_list=visual_positions_list,
                    use_visual_condition=args.visual_aux_visual_condition,
                    image_token_id=image_token_id,
                    video_token_id=video_token_id,
                    max_visual_tokens=args.max_visual_tokens,
                    vit_embeds=vit_embeds,
                )
                if vis_explains and vis_explains[0]:
                    output_dict["visual_decoder_explain"] = vis_explains[0]

            del fwd_out, hidden_states
            torch.cuda.empty_cache()

        torch.cuda.synchronize()
        start_time = time.time()

        # Generate answer
        gen_outputs = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            return_dict_in_generate=True,
            output_scores=True,
        )
        torch.cuda.synchronize()
        latency = time.time() - start_time
        print(f"[INFO] Generation latency: {latency} seconds")

        generated_ids = gen_outputs.sequences
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=False, clean_up_tokenization_spaces=False
        )

        output_dict["latency"] = latency
        output_dict["messages"] = messages
        output_dict["GT"] = item.get("GT", "")
        output_dict["output_text"] = output_text[0]

        scores = gen_outputs.scores
        batch_size = generated_ids.shape[0]

        entropies = []
        for step_logits in scores:
            probs = F.softmax(step_logits.float(), dim=-1)
            step_entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)
            entropies.append(step_entropy)

        entropies_tensor = torch.stack(entropies).transpose(0, 1)
        avg_entropy = entropies_tensor.mean(dim=1)

        transition_scores = model.compute_transition_scores(
            generated_ids, scores, normalize_logits=True)
        avg_log_prob = transition_scores.mean(dim=1)
        seq_confidence = torch.exp(avg_log_prob)

        output_dict["avg_entropy"] = avg_entropy.item()
        output_dict["avg_log_prob"] = avg_log_prob.item()
        output_dict["seq_confidence"] = seq_confidence.item()

        if idx < 3 or idx % 100 == 0:
            print(f"\n=== Sample {idx} ===")
            print(f"  Output: {output_text[0][:]}")
            print(f"  Entropy: {avg_entropy.item():.4f}, Confidence: {seq_confidence.item():.2%}")
            if output_dict.get("decoder_explain"):
                print(f"  Explain: {output_dict['decoder_explain'][:]}")
            if output_dict.get("visual_decoder_explain"):
                print(f"  VisExplain: {output_dict['visual_decoder_explain'][:]}")

        output_list.append(output_dict)

        with open(args.output_path, 'w') as f:
            json.dump(output_list, f, indent=4, ensure_ascii=False)

    print(f"\n[INFO] Done. {len(output_list)} results saved to {args.output_path}")


if __name__ == "__main__":
    main()

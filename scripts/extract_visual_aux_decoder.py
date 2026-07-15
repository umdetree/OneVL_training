#!/usr/bin/env python3
"""
Extract the visual aux decoder from a Stage 0.5 pretrain checkpoint
and save it as a standalone HuggingFace model directory.

Usage:
    python scripts/extract_visual_aux_decoder.py \
        --checkpoint_dir outputs/navsim/qwen3_vl_visual_aux_pretrain_stage0_5_vis4_txt2/checkpoint-XXX \
        --output_dir outputs/navsim/visual_aux_decoder_pretrained_2b

The output directory will contain a standard Qwen3-VL model that can be
used directly as VISUAL_AUX_MODEL_PATH in Stage 1/2 training.
"""
import argparse
import json
import os

import torch
from safetensors.torch import load_file, save_file


def extract_visual_aux_decoder(checkpoint_dir: str, output_dir: str):
    """Extract _latent_cot_visual_aux_decoder.* weights from checkpoint,
    strip the prefix, and save as standalone HF model."""

    # 1. Load all weights from checkpoint
    index_path = os.path.join(checkpoint_dir, 'model.safetensors.index.json')
    single_path = os.path.join(checkpoint_dir, 'model.safetensors')

    aux_weights = {}
    if os.path.exists(index_path):
        from collections import defaultdict
        with open(index_path) as f:
            weight_map = json.load(f).get('weight_map', {})
        shards: dict[str, list[str]] = defaultdict(list)
        for key, shard_file in weight_map.items():
            if key.startswith('_latent_cot_visual_aux_decoder.'):
                shards[shard_file].append(key)
        if not shards:
            print(f'ERROR: No _latent_cot_visual_aux_decoder.* weights found in {checkpoint_dir}')
            return
        for shard_file, keys in shards.items():
            shard_path = os.path.join(checkpoint_dir, shard_file)
            shard_weights = load_file(shard_path, device='cpu')
            for k in keys:
                if k in shard_weights:
                    # Strip the prefix: _latent_cot_visual_aux_decoder.model.layers.0...
                    # → model.layers.0...
                    new_key = k[len('_latent_cot_visual_aux_decoder.'):]
                    aux_weights[new_key] = shard_weights[k]
    elif os.path.exists(single_path):
        all_weights = load_file(single_path, device='cpu')
        for k, v in all_weights.items():
            if k.startswith('_latent_cot_visual_aux_decoder.'):
                new_key = k[len('_latent_cot_visual_aux_decoder.'):]
                aux_weights[new_key] = v
    else:
        print(f'ERROR: No safetensors found in {checkpoint_dir}')
        return

    print(f'Extracted {len(aux_weights)} weight tensors from visual aux decoder')

    # 2. Copy config.json and tokenizer files from checkpoint_dir
    os.makedirs(output_dir, exist_ok=True)

    # Copy config.json
    config_src = os.path.join(checkpoint_dir, 'config.json')
    if not os.path.exists(config_src):
        # Try to get from the visual_aux_model_path or model_path
        print(f'WARNING: No config.json in checkpoint, will create from model metadata')
    else:
        import shutil
        shutil.copy2(config_src, os.path.join(output_dir, 'config.json'))

    # Copy tokenizer files
    tokenizer_files = [
        'tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json',
        'merges.txt', 'vocab.json', 'chat_template.jinja',
        'preprocessor_config.json', 'processor_config.json',
        'processing_qwen3vl_visual.py', 'tokenization_qwen3vl_visual.py',
        'video_preprocessor_config.json',
    ]
    import shutil
    for fname in tokenizer_files:
        src = os.path.join(checkpoint_dir, fname)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(output_dir, fname))
            print(f'  Copied {fname}')

    # 3. Save weights as safetensors
    save_file(aux_weights, os.path.join(output_dir, 'model.safetensors'))
    print(f'Saved visual aux decoder to {output_dir}')

    # 4. Verify: load back and check
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(output_dir, trust_remote_code=True)
    print(f'Verified: model_type={cfg.model_type}, '
          f'hidden_size={cfg.text_config.hidden_size}, '
          f'num_layers={cfg.text_config.num_hidden_layers}')


def main():
    parser = argparse.ArgumentParser(description='Extract visual aux decoder from pretrain checkpoint')
    parser.add_argument('--checkpoint_dir', type=str, required=True,
                        help='Path to the pretrain checkpoint directory')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for the standalone visual aux decoder')
    args = parser.parse_args()
    extract_visual_aux_decoder(args.checkpoint_dir, args.output_dir)


if __name__ == '__main__':
    main()

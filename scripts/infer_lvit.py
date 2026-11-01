#!/usr/bin/env python
"""Segment a new image with its image-only-generated or supplied description."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np
import torch

from src.experiment.common import atomic_json, gpu_memory, project_path, sha256_file, synchronize
from src.experiment.data import load_image
from src.experiment.engine import load_checkpoint
from src.experiment.model import build_model
from src.experiment.text import FrozenBertEncoder


def infer_image(image_path, description, checkpoint_path, output_dir,
                bert_model='weights/bert-base-uncased', device='cuda'):
    started = time.perf_counter()
    image_path, checkpoint_path = project_path(image_path), project_path(checkpoint_path)
    output = project_path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if not isinstance(description, str) or not description.strip():
        raise ValueError('New-image inference requires a nonempty description')
    if torch.device(device).type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    encoder = FrozenBertEncoder(bert_model, device)
    synchronize(device)
    encode_started = time.perf_counter()
    text, token_records = encoder.encode([description])
    synchronize(device)
    encode_seconds = time.perf_counter() - encode_started
    encoder_fingerprint = encoder.fingerprint
    del encoder
    if torch.device(device).type == 'cuda':
        torch.cuda.empty_cache()
    model = build_model().to(device)
    checkpoint = load_checkpoint(checkpoint_path, model)
    model.eval()
    # Training cache metadata are embedded in run_config.json, while the cache path
    # is also retained in checkpoint.config. Validate the actual encoder bytes.
    cache = torch.load(project_path(checkpoint['config']['feature_cache']), map_location='cpu', weights_only=False)
    expected_encoder = cache['metadata']['encoder']
    for key in ('version', 'model_files_sha256'):
        if expected_encoder[key] != encoder_fingerprint[key]:
            raise ValueError(f'Inference encoder differs from training: {key}')
    del cache
    image = load_image(image_path).unsqueeze(0).to(device)
    text = text.to(device)
    synchronize(device)
    infer_started = time.perf_counter()
    with torch.inference_mode():
        probability = model(image, text)
    synchronize(device)
    inference_seconds = time.perf_counter() - infer_started
    if not torch.isfinite(probability).all():
        raise RuntimeError('Nonfinite new-image prediction')
    probability = probability[0, 0].cpu().numpy()
    binary = (probability >= 0.5).astype(np.uint8) * 255
    if not cv2.imwrite(str(output/'mask_224.png'), binary):
        raise OSError('Failed to write predicted mask')
    np.save(output/'probability_224.npy', probability)
    result = {'image_path': str(image_path), 'image_sha256': sha256_file(image_path),
              'description': description, 'checkpoint': str(checkpoint_path),
              'checkpoint_sha256': sha256_file(checkpoint_path), 'checkpoint_epoch': checkpoint['epoch'] + 1,
              'training_source': checkpoint['config']['source'], 'text_encoder': encoder_fingerprint,
              'tokenization': token_records[0], 'mask_grid': [224, 224], 'threshold': 0.5,
              'foreground_fraction': float((binary > 0).mean()),
              'probability_min': float(probability.min()), 'probability_max': float(probability.max()),
              'probability_mean': float(probability.mean()), 'encode_seconds': encode_seconds,
              'segmentation_forward_seconds': inference_seconds,
              'total_seconds': time.perf_counter() - started, **gpu_memory(device)}
    atomic_json(output/'inference.json', result)
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image', required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--description')
    group.add_argument('--description-file', help='UTF-8 plain text generated from this image, without a GT mask')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--bert-model', default='weights/bert-base-uncased')
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    description = args.description if args.description is not None else project_path(args.description_file).read_text(encoding='utf-8')
    infer_image(args.image, description, args.checkpoint, args.output_dir, args.bert_model, args.device)


if __name__ == '__main__':
    main()

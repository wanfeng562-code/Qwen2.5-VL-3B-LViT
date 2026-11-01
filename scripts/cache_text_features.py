#!/usr/bin/env python
"""Cache one frozen BERT representation for all immutable dataset splits."""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.experiment.common import atomic_json, atomic_torch_save, gpu_memory, project_path, sha256_file, sha256_text, synchronize, timestamp
from src.experiment.data import manifest_rows
from src.experiment.text import FrozenBertEncoder, load_descriptions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', required=True, choices=['monuseg', 'cov19', 'mosmeddata'])
    parser.add_argument('--source', required=True, choices=['provided_legacy_text', 'qwen_predicted_text', 'qwen_description'])
    parser.add_argument('--descriptions', help='Qwen JSONL with plain_text and image provenance')
    parser.add_argument('--bert-model', default='weights/bert-base-uncased')
    parser.add_argument('--output', required=True)
    parser.add_argument('--protocol', default='configs/data_protocol_v1.json')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    import torch
    torch.set_num_threads(args.threads)
    output = project_path(args.output)
    if output.exists() and not args.overwrite:
        raise FileExistsError(f'Cache exists: {output}; use --overwrite intentionally')
    if args.batch_size < 1:
        raise ValueError('batch-size must be positive')
    started = time.perf_counter()
    rows, hashes = manifest_rows(args.dataset, args.protocol, verify_assets=True)
    all_rows = [row for split in ('train', 'val', 'test') for row in rows[split]]
    descriptions, generation_metadata = load_descriptions(all_rows, args.source, args.descriptions, return_metadata=True)
    source = 'qwen_predicted_text' if args.source == 'qwen_description' else args.source
    if torch.device(args.device).type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    load_started = time.perf_counter()
    encoder = FrozenBertEncoder(args.bert_model, args.device)
    synchronize(args.device)
    load_seconds = time.perf_counter() - load_started
    features, provenance = [], []
    encode_started = time.perf_counter()
    for start in range(0, len(all_rows), args.batch_size):
        subset = all_rows[start:start + args.batch_size]
        tensors, records = encoder.encode([descriptions[row['sample_id']] for row in subset])
        features.append(tensors)
        for row, record in zip(subset, records):
            provenance.append({**record, 'sample_id': row['sample_id'], 'split': row['split'],
                               'image_sha256': row['image_sha256'],
                               'raw_text_sha256': sha256_text(descriptions[row['sample_id']]),
                               'raw_text': descriptions[row['sample_id']]})
    synchronize(args.device)
    encode_seconds = time.perf_counter() - encode_started
    metadata = {'created_at': timestamp(), 'dataset': args.dataset, 'source': source,
                'manifest_sha256': hashes, 'protocol_sha256': sha256_file(project_path(args.protocol)),
                'encoder': encoder.fingerprint, 'descriptions_path': args.descriptions,
                'descriptions_sha256': sha256_file(project_path(args.descriptions)) if args.descriptions else None,
                **generation_metadata, 'sample_count': len(all_rows), 'model_load_seconds': load_seconds,
                'encode_seconds': encode_seconds, 'total_seconds': time.perf_counter() - started,
                'encode_samples_per_second': len(all_rows) / encode_seconds,
                'torch_version': torch.__version__, 'device': args.device, **gpu_memory(args.device)}
    payload = {'schema_version': 1, 'sample_ids': [row['sample_id'] for row in all_rows],
               'features': torch.cat(features), 'records': provenance, 'metadata': metadata}
    atomic_torch_save(output, payload)
    report = {**metadata, 'output': str(output), 'cache_bytes': output.stat().st_size,
              'cache_sha256': sha256_file(output)}
    atomic_json(output.with_suffix(output.suffix + '.json'), report)
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()

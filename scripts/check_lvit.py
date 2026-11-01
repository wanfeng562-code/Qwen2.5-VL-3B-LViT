#!/usr/bin/env python
"""Real-data checks: preprocessing, loss parity, forward/update, learn, resume."""
import argparse
import copy
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from src.experiment.common import (ROOT, atomic_json, atomic_torch_save, gpu_memory, project_path,
                                   seed_everything, sha256_file, synchronize)
from src.experiment.data import ManifestDataset, make_loader
from src.experiment.engine import (checkpoint_payload, configure, evaluate, load_checkpoint,
                                   make_optimizer, run_identity, setup_data, update_batch)
from src.experiment.loss import WeightedDiceBCE
from src.experiment.model import build_model


def check_loss_parity():
    import ast
    source = (ROOT/'src/lvit/utils.py').read_text(encoding='utf-8')
    classes = {'WeightedBCE', 'WeightedDiceLoss', 'WeightedDiceBCE'}
    nodes = [node for node in ast.parse(source).body if isinstance(node, ast.ClassDef) and node.name in classes]
    namespace = {'torch': torch, 'nn': nn, 'F': F}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), '<original_lvit_loss_classes>', 'exec'), namespace)
    original = namespace['WeightedDiceBCE'](dice_weight=0.5, BCE_weight=0.5)
    adapted = WeightedDiceBCE()
    results = []
    generator = torch.Generator().manual_seed(17)
    for case in ('mixed', 'empty', 'full'):
        raw = torch.rand((2, 1, 9, 11), generator=generator) * 0.98 + 0.01
        target = (torch.rand((2, 1, 9, 11), generator=generator) > 0.6).float()
        if case == 'empty':
            target.zero_()
        elif case == 'full':
            target.fill_(1)
        p1, p2 = raw.clone().requires_grad_(), raw.clone().requires_grad_()
        loss1, loss2 = original(p1, target), adapted(p2, target)
        loss1.backward()
        loss2.backward()
        value_error = float((loss1 - loss2).abs())
        gradient_error = float((p1.grad - p2.grad).abs().max())
        if value_error > 1e-7 or gradient_error > 1e-7:
            raise AssertionError('Adapted binary WeightedDiceBCE differs from uploaded source')
        results.append({'case': case, 'loss_absolute_difference': value_error, 'gradient_max_absolute_difference': gradient_error})
    return results


def largest_difference(first, second):
    if torch.is_tensor(first):
        return (first.detach().cpu() - second.detach().cpu()).abs().max().item() if first.numel() else 0.0
    if isinstance(first, dict):
        if first.keys() != second.keys():
            return float('inf')
        return max((largest_difference(first[key], second[key]) for key in first), default=0.0)
    if isinstance(first, (list, tuple)):
        if len(first) != len(second):
            return float('inf')
        return max((largest_difference(a, b) for a, b in zip(first, second)), default=0.0)
    return 0.0 if first == second else float('inf')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--mode', choices=['data', 'batch', 'full'], default='full')
    parser.add_argument('--steps', type=int, default=100)
    parser.add_argument('--device', default=None)
    args = parser.parse_args()
    config = configure(json.loads(project_path(args.config).read_text(encoding='utf-8')))
    if args.device:
        config['device'] = args.device
    device = config['device']
    torch.set_num_threads(8)
    seed_everything(config['seed'], config['deterministic'])
    output = project_path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    result = {'status': 'running', 'mode': args.mode, 'dataset': config['dataset'], 'source': config['source'],
              'loss_parity': check_loss_parity(), 'test_masks_used_for_training': False}
    rows, hashes, features = setup_data(config)
    datasets = {split: ManifestDataset(rows[split], features, preload=True) for split in ('train', 'val', 'test')}
    result['data'] = {}
    for split, dataset in datasets.items():
        foreground = [float(item['mask'].mean()) for item in dataset]
        if not all(tuple(item['image'].shape) == (3, 224, 224) and tuple(item['mask'].shape) == (1, 224, 224)
                   and tuple(item['text'].shape) == (10, 768) and set(item['mask'].unique().tolist()) <= {0.0, 1.0}
                   and item['image'].min() >= 0 and item['image'].max() <= 1
                   and torch.isfinite(item['text']).all() for item in dataset):
            raise AssertionError('Real-data shape/range/binary/feature check failed')
        result['data'][split] = {'sample_count': len(dataset), 'foreground_min': min(foreground),
                                  'foreground_max': max(foreground), 'manifest_sha256': hashes[split]}
    if args.mode == 'data':
        result['status'] = 'passed'
        atomic_json(output/'checks.json', result)
        print(json.dumps(result), flush=True)
        return
    if torch.device(device).type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    dataset = ManifestDataset(rows['train'][:config['batch_size']], features)
    loader = make_loader(dataset, config['batch_size'], config['seed'], device=device)
    batch = next(iter(loader))
    result['real_training_sample_ids'] = batch['sample_id']
    model = build_model().to(device)
    result['parameter_count'] = sum(parameter.numel() for parameter in model.parameters())
    criterion = WeightedDiceBCE(config['dice_weight'], config['bce_weight'])
    optimizer, scheduler = make_optimizer(model, config)
    initial = evaluate(model, loader, criterion, device)
    model.train()
    synchronize(device)
    started = time.perf_counter()
    first = update_batch(model, optimizer, criterion, batch, device)
    synchronize(device)
    result['single_batch'] = {**first, 'seconds': time.perf_counter() - started,
                              'image_shape': list(batch['image'].shape), 'mask_shape': list(batch['mask'].shape),
                              'text_shape': list(batch['text'].shape), **gpu_memory(device)}
    if args.mode == 'full':
        if args.steps < 2:
            raise ValueError('Overfit check needs at least two steps')
        curve = [{'step': 1, **first}]
        for step in range(1, args.steps):
            stats = update_batch(model, optimizer, criterion, batch, device)
            curve.append({'step': step + 1, **stats})
            if (step + 1) % 10 == 0:
                print(json.dumps({'overfit_step': step + 1, **stats}), flush=True)
        final = evaluate(model, loader, criterion, device)
        result['overfit'] = {'steps': args.steps, 'initial': initial, 'final': final, 'curve': curve,
                            'learning_confirmed': final['loss'] < initial['loss'] and final['dice'] > initial['dice']}
        if not result['overfit']['learning_confirmed']:
            result['status'] = 'failed_learning_check'
            atomic_json(output/'checks.json', result)
            raise AssertionError('Real-sample overfit did not improve both loss and foreground Dice')
        if scheduler:
            scheduler.step(1)
        identity = run_identity(config, hashes, features)
        model.train()
        payload = checkpoint_payload(model, optimizer, scheduler, config, identity, 0, final['dice'], 0, args.steps, 0.0)
        checkpoint_path = output/'temporary_resume_check.pt'
        atomic_torch_save(checkpoint_path, payload)
        checkpoint_hash = sha256_file(checkpoint_path)
        del payload
        expected_stats = update_batch(model, optimizer, criterion, batch, device)
        if scheduler:
            scheduler.step(2)
        expected_model = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        # Comparing all optimizer moments is meaningful, but avoid keeping a second GPU copy.
        expected_optimizer = copy.deepcopy(optimizer.state_dict())
        for state in expected_optimizer['state'].values():
            for key, value in state.items():
                if torch.is_tensor(value):
                    state[key] = value.cpu()
        expected_scheduler = copy.deepcopy(scheduler.state_dict()) if scheduler else None
        load_checkpoint(checkpoint_path, model, optimizer, scheduler, identity, restore_random=True)
        model.train()
        resumed_stats = update_batch(model, optimizer, criterion, batch, device)
        if scheduler:
            scheduler.step(2)
        errors = {'model_max_absolute_difference': largest_difference(expected_model, model.state_dict()),
                  'optimizer_max_absolute_difference': largest_difference(expected_optimizer, optimizer.state_dict()),
                  'scheduler_max_absolute_difference': largest_difference(expected_scheduler, scheduler.state_dict() if scheduler else None),
                  'loss_absolute_difference': abs(expected_stats['loss'] - resumed_stats['loss'])}
        result['resume'] = {**errors, 'temporary_checkpoint_sha256': checkpoint_hash,
                             'rng_dropout_restored': True, 'passed': all(value == 0 for value in errors.values())}
        if not result['resume']['passed']:
            result['status'] = 'failed_resume_check'
            atomic_json(output/'checks.json', result)
            raise AssertionError('Save/resume next optimizer step is not exactly equivalent')
        checkpoint_path.unlink()
    result['status'] = 'passed'
    result.update(gpu_memory(device))
    atomic_json(output/'checks.json', result)
    print(json.dumps({key: value for key, value in result.items() if key != 'overfit'}), flush=True)


if __name__ == '__main__':
    main()

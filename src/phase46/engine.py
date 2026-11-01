"""Training, epoch-boundary recovery and protocol evaluation of versioned Qwen-conditioned LViT."""
import copy
import importlib.metadata
import json
import math
import os
from pathlib import Path
import time

import numpy as np
import torch

from scripts.protocol_metrics import binary_sample_scores, macro_scores
from src.experiment.common import (ROOT, atomic_json, atomic_torch_save, gpu_memory, project_path, restore_rng,
                     rng_state, seed_everything, sha256_file, sha256_text, synchronize, timestamp)
from .data import FeatureStore, ManifestDataset, make_loader, manifest_rows, read_protocol
from src.experiment.loss import WeightedDiceBCE
from .model import MODEL_VERSION, build_model


def configure(config):
    config = copy.deepcopy(config)
    config.setdefault('fusion','legacy');config.setdefault('feature_mode','correct')
    assert config['architecture'] in ('S','V_mid','V_merged')
    assert config['fusion'] in ('legacy','cross_attention')
    assert config['feature_mode'] in ('correct','constant','shuffled','legacy')
    required = {'dataset', 'source', 'feature_cache', 'seed', 'max_epochs', 'batch_size', 'learning_rate'}
    if missing := required - config.keys():
        raise ValueError(f'Missing training configuration keys: {missing}')
    defaults = {'protocol': 'configs/data_protocol_v1.json', 'workers': 0, 'preload': True,
                'deterministic': True, 'device': 'cuda', 'patience': 50, 'min_epochs': 50,
                'optimizer': 'Adam', 'weight_decay': 0.0, 'scheduler': 'cosine_warm_restarts',
                'scheduler_T_0': 10, 'scheduler_T_mult': 1, 'scheduler_eta_min': 1e-4,
                'dice_weight': 0.5, 'bce_weight': 0.5, 'amp': False, 'geometric_augmentation': False}
    defaults.update({'precision': 'float32', 'unet_pretrained': False})
    for key, value in defaults.items():
        config.setdefault(key, value)
    if config['source'] == 'qwen_description':
        config['source'] = 'qwen_predicted_text'
    if (config['optimizer'] != 'Adam' or config['amp'] or config['geometric_augmentation']
            or config['precision'] != 'float32' or config['unet_pretrained']):
        raise ValueError('This frozen experiment uses full-float32 Adam without geometric augmentation')
    if min(config['max_epochs'], config['batch_size'], config['learning_rate']) <= 0:
        raise ValueError('Training epoch/batch/lr must be positive')
    if config['patience'] < 1 or config['min_epochs'] < 1:
        raise ValueError('Early-stopping patience and min_epochs must be positive')
    if config['min_epochs'] > config['max_epochs']:
        raise ValueError('min_epochs cannot exceed max_epochs')
    return config


def setup_data(config, verify_assets=True):
    rows, hashes = manifest_rows(config['dataset'], config['protocol'], verify_assets)
    all_rows = [row for split in ('train', 'val') for row in rows[split]]
    features = FeatureStore(config['feature_cache'], all_rows, config['source'], hashes, config.get('feature_mode','correct'),config['seed'])
    if features.metadata['protocol_sha256'] != sha256_file(project_path(config['protocol'])):
        raise ValueError('Feature cache protocol hash differs')
    return rows, hashes, features


def make_optimizer(model, config):
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    if config['scheduler'] == 'cosine_warm_restarts':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=config['scheduler_T_0'], T_mult=config['scheduler_T_mult'], eta_min=config['scheduler_eta_min'])
    elif config['scheduler'] == 'constant':
        scheduler = None
    else:
        raise ValueError(f'Unknown scheduler: {config["scheduler"]}')
    return optimizer, scheduler


def move_batch(batch, device):
    return batch['image'].to(device),batch['mask'].to(device),{k:v.to(device) for k,v in batch['text'].items()}


def update_batch(model, optimizer, criterion, batch, device):
    image, target, text = move_batch(batch, device)
    optimizer.zero_grad(set_to_none=True)
    prediction = model(image, text)
    if not torch.isfinite(prediction).all() or prediction.min() < 0 or prediction.max() > 1:
        raise RuntimeError('LViT probabilities nonfinite/outside [0,1]')
    loss = criterion(prediction, target)
    if not torch.isfinite(loss):
        raise RuntimeError('Nonfinite loss')
    loss.backward()
    squared_gradient = torch.zeros((), device=device)
    gradient_tensors = 0
    for parameter in model.parameters():
        if parameter.grad is not None:
            if not torch.isfinite(parameter.grad).all():
                raise RuntimeError('Nonfinite model gradient')
            squared_gradient += parameter.grad.detach().float().square().sum()
            gradient_tensors += 1
    norm = squared_gradient.sqrt().item()
    if norm == 0 or not math.isfinite(norm):
        raise RuntimeError('Zero or nonfinite gradient norm')
    text_grad = sum(float(p.grad.detach().float().square().sum()) for name,p in model.named_parameters() if p.grad is not None and ('text_module' in name or 'downVit.CTBN3' in name or 'downVit.text_attention' in name or name.startswith('adapter') or name=='gate')) ** .5
    stats = {'text_gradient_l2':text_grad,'loss': loss.item(), 'gradient_l2': norm, 'gradient_tensors': gradient_tensors,
             'prediction_min': prediction.detach().min().item(), 'prediction_max': prediction.detach().max().item(),
             'prediction_mean': prediction.detach().mean().item(), 'target_foreground_fraction': target.mean().item()}
    optimizer.step()
    return stats


@torch.inference_mode()
def evaluate(model, loader, criterion, device, threshold=0.5, prediction_dir=None):
    model.eval()
    synchronize(device)
    started = time.perf_counter()
    scores, loss_sum, sample_count = [], 0.0, 0
    output_stats = []
    if prediction_dir is not None:
        import cv2
        prediction_dir = Path(prediction_dir)
        prediction_dir.mkdir(parents=True, exist_ok=True)
    for batch in loader:
        image, target, text = move_batch(batch, device)
        prediction = model(image, text)
        if not torch.isfinite(prediction).all():
            raise RuntimeError('Nonfinite evaluation prediction')
        loss = criterion(prediction, target)
        if not torch.isfinite(loss):
            raise RuntimeError('Nonfinite evaluation loss')
        count = len(image)
        sample_count += count
        loss_sum += loss.item() * count
        probs, truths = prediction.cpu().numpy(), target.cpu().numpy()
        for index, sample_id in enumerate(batch['sample_id']):
            score = binary_sample_scores(probs[index, 0], truths[index, 0], threshold)
            scores.append({'sample_id': sample_id, 'image_path': batch['image_path'][index], **score})
            output_stats.append(float(probs[index].mean()))
            if prediction_dir is not None:
                if not cv2.imwrite(str(prediction_dir / (sample_id + '.png')), (probs[index, 0] >= threshold).astype(np.uint8) * 255):
                    raise OSError('Failed to write prediction PNG')
    synchronize(device)
    elapsed = time.perf_counter() - started
    if sample_count != len(loader.dataset):
        raise RuntimeError('Evaluation did not consume exactly the actual manifest length')
    metrics = macro_scores(scores)
    return {**metrics, 'dice_percent': 100 * metrics['dice'], 'iou_percent': 100 * metrics['iou'],
            'loss': loss_sum / sample_count, 'seconds': elapsed, 'samples_per_second': sample_count / elapsed,
            'prediction_mean': sum(output_stats) / len(output_stats), 'samples': scores}


def checkpoint_payload(model, optimizer, scheduler, config, identity, epoch, best_dice, best_epoch, global_step, elapsed):
    return {'schema_version': 1, 'model_version': MODEL_VERSION, 'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(), 'scheduler_state': scheduler.state_dict() if scheduler else None,
            'rng_state': rng_state(), 'config': config, 'identity': identity, 'epoch': epoch,
            'best_validation_dice': best_dice, 'best_epoch': best_epoch, 'global_step': global_step,
            'training_elapsed_seconds': elapsed, 'resume_granularity': 'completed_epoch',
            'data_order_policy': 'torch_generator_seed_plus_epoch_times_100003'}


def load_checkpoint(path, model, optimizer=None, scheduler=None, expected_identity=None, restore_random=False):
    checkpoint = torch.load(project_path(path), map_location='cpu', weights_only=False)
    if checkpoint.get('schema_version') != 1 or checkpoint['model_version'] != MODEL_VERSION:
        raise ValueError('Checkpoint schema/model version mismatch')
    if expected_identity is not None and checkpoint['identity'] != expected_identity:
        raise ValueError('Resume identity differs from frozen run (config/cache/manifest/code/protocol)')
    model.load_state_dict(checkpoint['model_state'], strict=True)
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state'])
    if scheduler is not None:
        scheduler.load_state_dict(checkpoint['scheduler_state'])
    if restore_random:
        restore_rng(checkpoint['rng_state'])
    return checkpoint


def run_identity(config, hashes, features):
    config_hash = sha256_text(json.dumps(config, sort_keys=True, separators=(',', ':')))
    source_paths = sorted((ROOT / 'src/experiment').glob('*.py')) + [ROOT/'src/phase45'/name for name in ('__init__.py','attributes.py','budget.py','data.py','text.py','model.py','engine.py')] + [ROOT/'src/phase46'/name for name in ('__init__.py','budget.py','data.py','model.py','engine.py')]
    vendor_paths = sorted((ROOT / 'src/lvit/nets').glob('*.py'))
    return {'config_sha256': config_hash, 'feature_cache_sha256': features.sha256,
            'protocol_sha256': sha256_file(project_path(config['protocol'])), 'manifest_sha256': hashes,
            'code_sha256': {str(path.relative_to(ROOT)): sha256_file(path) for path in source_paths + vendor_paths},
            'metrics_code_sha256': sha256_file(ROOT/'scripts/protocol_metrics.py')}


def train(config, output_dir, resume=None):
    config = configure(config)
    output = project_path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if (output/'last.pt').exists() and resume is None:
        raise FileExistsError('Run already has last.pt; explicitly --resume it or choose a new output directory')
    device = torch.device(config['device'])
    if device.type == 'cuda' and os.environ.get('CUBLAS_WORKSPACE_CONFIG') not in (':4096:8', ':16:8'):
        raise RuntimeError('Set CUBLAS_WORKSPACE_CONFIG=:4096:8 before starting deterministic CUDA training')
    seed_everything(config['seed'], config['deterministic'])
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    rows, hashes, features = setup_data(config)
    identity = run_identity(config, hashes, features)
    train_dataset = ManifestDataset(rows['train'], features, config['preload'])
    val_dataset = ManifestDataset(rows['val'], features, config['preload'])
    model = build_model(config).to(device)
    import hashlib
    initial_digest=hashlib.sha256()
    for name,value in model.state_dict().items():
        initial_digest.update(name.encode());initial_digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    initial_hash=initial_digest.hexdigest()
    criterion = WeightedDiceBCE(config['dice_weight'], config['bce_weight'])
    optimizer, scheduler = make_optimizer(model, config)
    start_epoch, best_dice, best_epoch, global_step, previous_elapsed = 0, -1.0, -1, 0, 0.0
    if resume:
        checkpoint = load_checkpoint(resume, model, optimizer, scheduler, identity, restore_random=True)
        start_epoch = checkpoint['epoch'] + 1
        best_dice, best_epoch, global_step = checkpoint['best_validation_dice'], checkpoint['best_epoch'], checkpoint['global_step']
        previous_elapsed = checkpoint['training_elapsed_seconds']
        # A resumed run retains the best checkpoint selected before interruption.
        if best_epoch >= 0 and not (output/'best.pt').is_file():
            raise FileNotFoundError('Resume output must retain its original best.pt')
    versions = {}
    for name in ('torch', 'torchvision', 'transformers', 'timm', 'opencv-python-headless', 'numpy'):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    atomic_json(output/'run_config.json', {'config': config, 'identity': identity, 'versions': versions,
                'parameter_count': sum(p.numel() for p in model.parameters()), 'created_at': timestamp(),
                'text_encoder': features.metadata['encoder'], 'initial_model_sha256':initial_hash,'feature_mapping':features.mapping,'counts': {s: len(r) for s, r in rows.items()},
                'gpu': torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
                'test_access': 'training never instantiates a test loader; evaluate selected best separately'})
    started = time.perf_counter()
    # A crash after last.pt but before result.json must not add an extra epoch
    # once the exact frozen stopping rule has already been reached.
    stopped_in_checkpoint = (start_epoch >= config['min_epochs'] and
                             start_epoch - 1 - best_epoch >= config['patience'])
    stop_reason = 'validation_patience' if stopped_in_checkpoint else 'max_epochs'
    val_loader = make_loader(val_dataset, config['batch_size'], config['seed'], workers=config['workers'], device=device)
    epoch_limit = start_epoch if stopped_in_checkpoint else config['max_epochs']
    for epoch in range(start_epoch, epoch_limit):
        synchronize(device)
        epoch_started = time.perf_counter()
        model.train()
        loader = make_loader(train_dataset, config['batch_size'], config['seed'], epoch, True, config['workers'], device)
        stats, train_samples = [], 0
        learning_rate = optimizer.param_groups[0]['lr']
        for batch in loader:
            stats.append((len(batch['image']), update_batch(model, optimizer, criterion, batch, device)))
            global_step += 1
            train_samples += len(batch['image'])
        synchronize(device)
        train_seconds = time.perf_counter() - epoch_started
        if train_samples != len(train_dataset):
            raise RuntimeError('Training sample count differs from manifest')
        val = evaluate(model, val_loader, criterion, device)
        improved = val['dice'] > best_dice
        if improved:
            best_dice, best_epoch = val['dice'], epoch
        if scheduler is not None:
            scheduler.step(epoch + 1)
        elapsed = previous_elapsed + time.perf_counter() - started
        summary = {'epoch': epoch + 1, 'global_step': global_step, 'learning_rate': learning_rate,
                   'train_loss': sum(n * s['loss'] for n, s in stats) / train_samples,
                   'train_sample_count': train_samples, 'train_seconds': train_seconds,
                   'train_samples_per_second': train_samples / train_seconds,
                   'text_gradient_l2_min':min(s['text_gradient_l2'] for _,s in stats),
                   'text_gradient_l2_max':max(s['text_gradient_l2'] for _,s in stats),
                   'text_gate':model.conditioning_gate(),
                   'gradient_l2_min': min(s['gradient_l2'] for _, s in stats),
                   'gradient_l2_max': max(s['gradient_l2'] for _, s in stats),
                   'prediction_min': min(s['prediction_min'] for _, s in stats),
                   'prediction_max': max(s['prediction_max'] for _, s in stats),
                   'train_prediction_mean': sum(n * s['prediction_mean'] for n, s in stats) / train_samples,
                   'target_foreground_fraction': sum(n * s['target_foreground_fraction'] for n, s in stats) / train_samples,
                   'validation': {k: v for k, v in val.items() if k != 'samples'},
                   'best_epoch': best_epoch + 1, 'best_validation_dice': best_dice,
                   'epoch_seconds_before_checkpoint': time.perf_counter() - epoch_started, **gpu_memory(device)}
        payload = checkpoint_payload(model, optimizer, scheduler, config, identity, epoch, best_dice, best_epoch, global_step, elapsed)
        if improved:
            atomic_torch_save(output/'best.pt', payload)
            atomic_json(output/'best_validation.json', val)
        atomic_torch_save(output/'last.pt', payload)
        with (output/'history.jsonl').open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(summary, allow_nan=False) + '\n')
            handle.flush()
        atomic_json(output/'status.json', {'status': 'running', **summary})
        print(json.dumps(summary, allow_nan=False), flush=True)
        if epoch + 1 >= config['min_epochs'] and epoch - best_epoch >= config['patience']:
            stop_reason = 'validation_patience'
            break
    total = previous_elapsed + time.perf_counter() - started
    result = {'status': 'completed', 'stop_reason': stop_reason, 'epochs_completed': epoch + 1 if start_epoch < epoch_limit else start_epoch,
              'best_epoch': best_epoch + 1, 'best_validation_dice': best_dice,
              'global_step': global_step, 'training_elapsed_seconds': total, 'test_evaluated': False,
              'best_checkpoint': str(output/'best.pt'), 'last_checkpoint': str(output/'last.pt'), **gpu_memory(device)}
    atomic_json(output/'result.json', result)
    atomic_json(output/'status.json', result)
    print(json.dumps(result), flush=True)
    return result


def evaluate_checkpoint(checkpoint_path, split, output_dir, device='cuda', save_predictions=False):
    checkpoint = torch.load(project_path(checkpoint_path), map_location='cpu', weights_only=False)
    config = configure(checkpoint['config'])
    assert split=='val','Test evaluation uses the separate gated phase46 finalizer'
    if split == 'test' and checkpoint['epoch'] != checkpoint['best_epoch']:
        raise ValueError('Test evaluation requires the checkpoint selected by validation, normally best.pt')
    seed_everything(config['seed'], config['deterministic'])
    rows, hashes, features = setup_data(config)
    # Fixed-protocol evaluation rejects model/metric code drift as well as assets.
    identity = run_identity(config, hashes, features)
    if checkpoint['identity'] != identity:
        raise ValueError('Evaluation identity differs from frozen run (code/metrics/config/cache/protocol/manifest)')
    model = build_model(config).to(device)
    load_checkpoint(checkpoint_path, model)
    dataset = ManifestDataset(rows[split], features, config['preload'])
    loader = make_loader(dataset, config['batch_size'], config['seed'], workers=config['workers'], device=device)
    criterion = WeightedDiceBCE(config['dice_weight'], config['bce_weight'])
    output = project_path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if torch.device(device).type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    result = evaluate(model, loader, criterion, device, prediction_dir=output/'predictions' if save_predictions else None)
    result.update({'checkpoint': str(project_path(checkpoint_path)), 'checkpoint_sha256': sha256_file(project_path(checkpoint_path)),
                   'checkpoint_epoch': checkpoint['epoch'] + 1, 'checkpoint_best_epoch': checkpoint['best_epoch'] + 1,
                   'dataset': config['dataset'], 'source': config['source'], 'split': split,
                   'seed': config['seed'], 'manifest_sha256': hashes[split],
                   'code_matches_training': checkpoint['identity']['code_sha256'] == identity['code_sha256'],
                   'unit': 'fractions plus explicit percent fields', **gpu_memory(device)})
    atomic_json(output/(split+'_metrics.json'), result)
    print(json.dumps({k: v for k, v in result.items() if k != 'samples'}), flush=True)
    return result

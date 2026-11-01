"""CPU-only, read-only source-artifact audit of the completed stage 2--4 runs.

Only this script's requested report is written. It never launches generation,
training, evaluation or a GPU process. Test predictions are not opened until all
six formal training artifacts have independently passed their training checks.
"""
import argparse
import ast
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ.setdefault('OMP_NUM_THREADS', '8')
import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.protocol_metrics import binary_sample_scores

SEEDS = (666, 667, 668)
SOURCES = {'legacy': 'provided_legacy_text', 'qwen': 'qwen_predicted_text'}
COUNTS = {'train': 24, 'val': 6, 'test': 14}


class Pending(Exception):
    pass


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def near(actual, expected, message, tolerance=1e-11):
    require(math.isfinite(float(actual)) and abs(float(actual) - float(expected)) <= tolerance,
            f'{message}: actual={actual}, expected={expected}')


def finite_positive(value, label, allow_zero=False):
    require(isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
            and (value >= 0 if allow_zero else value > 0), f'Invalid {label}: {value}')


class Audit:
    def __init__(self, asset_scope):
        self.asset_scope = asset_scope
        self.checks = []
        self.values = {}
        self.digests = {}
        self.training = {}
        self.tests = {}

    @staticmethod
    def path(value):
        path = (ROOT / value).resolve()
        require(path.is_relative_to(ROOT.resolve()), f'Artifact path escapes project: {value}')
        return path

    def need(self, value):
        path = self.path(value)
        if not path.is_file():
            raise Pending(f'Artifact not yet available: {path.relative_to(ROOT)}')
        return path

    def read(self, value):
        return json.loads(self.need(value).read_text(encoding='utf-8'))

    def lines(self, value):
        return [json.loads(line) for line in self.need(value).read_text(encoding='utf-8').splitlines() if line.strip()]

    def sha(self, value):
        path = self.need(value)
        stat = path.stat()
        key = (str(path), stat.st_size, stat.st_mtime_ns)
        if key not in self.digests:
            digest = hashlib.sha256()
            with path.open('rb') as handle:
                for chunk in iter(lambda: handle.read(8 * 1024**2), b''):
                    digest.update(chunk)
            after = path.stat()
            require((stat.st_size, stat.st_mtime_ns) == (after.st_size, after.st_mtime_ns),
                    f'Artifact changed while hashing: {path}')
            self.digests[key] = digest.hexdigest()
        return self.digests[key]

    def check(self, name, operation):
        try:
            detail = operation()
            result = {'id': name, 'status': 'passed', 'details': detail}
            self.values[name] = detail
        except Pending as exc:
            result = {'id': name, 'status': 'pending', 'reason': str(exc)}
        except Exception as exc:
            result = {'id': name, 'status': 'failed', 'error': f'{type(exc).__name__}: {exc}'}
        self.checks.append(result)

    def manifests(self):
        if hasattr(self, 'rows'):
            return self.rows
        protocol = self.read('configs/data_protocol_v1.json')
        require(protocol['protocol_id'] == 'legacy_subset_v1' and protocol['counts']['monuseg'] == COUNTS,
                'Unexpected frozen MoNuSeg protocol/counts')
        require(protocol['image_input']['lvit_size'] == [224, 224] and protocol['image_input']['lvit_color_order'] == 'BGR', 'Image protocol differs')
        require(protocol['mask_input']['foreground_rule'] == 'grayscale_gt_0'
                and protocol['mask_input']['binarize_before_resize'] is True
                and protocol['mask_input']['resize'] == 'nearest', 'Mask protocol differs')
        metrics = protocol['metrics']
        require(metrics['probability_threshold'] == 0.5 and metrics['comparison'] == 'greater_equal'
                and metrics['denominator'] == 'actual_manifest_length'
                and metrics['checkpoint_tie'] == 'earliest' and metrics['evaluation_grid'] == [224, 224], 'Metric/selection protocol differs')
        self.rows, self.manifest_hashes = {}, {}
        ids = set()
        for split, count in COUNTS.items():
            name = f'{protocol["manifest_root"]}/monuseg/{split}.csv'
            with self.need(name).open(encoding='utf-8-sig', newline='') as handle:
                rows = list(csv.DictReader(handle))
            require(len(rows) == count, f'{split} manifest length differs')
            for row in rows:
                require(row['dataset'] == 'monuseg' and row['split'] == split and row['sample_id'] not in ids,
                        'Manifest has wrong source/split or duplicate ID')
                ids.add(row['sample_id'])
            self.rows[split] = rows
            self.manifest_hashes[split] = self.sha(name)
        self.protocol = protocol
        self.protocol_sha = self.sha('configs/data_protocol_v1.json')
        return self.rows

    def stage2(self):
        rows = self.manifests()
        data = self.read('outputs/lvit_checks/legacy_data_cpu/checks.json')
        full = self.read('outputs/lvit_checks/legacy_full/checks.json')
        require(data['status'] == full['status'] == 'passed', 'Real-data checks did not pass')
        for check in (data, full):
            require(check['test_masks_used_for_training'] is False, 'Check claims training used test masks')
            require({s: check['data'][s]['sample_count'] for s in COUNTS} == COUNTS, 'Real check sample counts differ')
            require({item['case'] for item in check['loss_parity']} == {'mixed', 'empty', 'full'}, 'Loss edge cases missing')
            for item in check['loss_parity']:
                require(item['loss_absolute_difference'] == item['gradient_max_absolute_difference'] == 0,
                        'Loss or gradient differs from uploaded source')
        batch = full['single_batch']
        require(batch['image_shape'] == [2, 3, 224, 224] and batch['mask_shape'] == [2, 1, 224, 224]
                and batch['text_shape'] == [2, 10, 768] and full['parameter_count'] == 39933345, 'Real forward interface differs')
        finite_positive(batch['gradient_l2'], 'real-batch gradient')
        require(batch['gradient_tensors'] > 0 and 0 <= batch['prediction_min'] <= batch['prediction_max'] <= 1, 'Invalid real-batch outputs/gradients')
        overfit = full['overfit']
        require(overfit['steps'] >= 100 and overfit['learning_confirmed'] is True, 'Insufficient/failed real overfit')
        require(overfit['final']['loss'] < overfit['initial']['loss'] and overfit['final']['dice'] > overfit['initial']['dice'], 'Recorded overfit does not actually improve')
        require(len(full['real_training_sample_ids']) == 2 and set(full['real_training_sample_ids']) <= {r['sample_id'] for r in rows['train']}, 'Overfit sample is outside train split')
        resume = full['resume']
        for key in ('model_max_absolute_difference', 'optimizer_max_absolute_difference', 'scheduler_max_absolute_difference', 'loss_absolute_difference'):
            require(resume[key] == 0, f'Resume mismatch: {key}')
        require(resume['passed'] is True and resume['rng_dropout_restored'] is True, 'RNG/dropout recovery unverified')
        smoke = self.read('outputs/lvit_checks/engine_smoke/verification.json')
        result = self.read('outputs/lvit_checks/engine_smoke/run/result.json')
        history = self.lines('outputs/lvit_checks/engine_smoke/run/history.jsonl')
        require(smoke['status'] == 'passed' and smoke['terminal_resume_no_extra_update'] is True
                and smoke['test_evaluated'] is False and result['test_evaluated'] is False, 'Engine/resume smoke failed')
        require([h['epoch'] for h in history] == [1, 2] and [h['global_step'] for h in history] == [12, 24]
                and result['global_step'] == 24 and result['epochs_completed'] == 2, 'Engine history/steps differ')
        for row in history:
            require(row['train_sample_count'] == 24 and row['validation']['sample_count'] == 6, 'Engine smoke denominators differ')
        for name, checksum in smoke['checkpoint_hashes'].items():
            require(self.sha('outputs/lvit_checks/engine_smoke/run/' + name) == checksum, 'Terminal resume changed smoke checkpoint')
        return {'real_parameter_count': 39933345, 'data_counts': COUNTS, 'overfit_initial_loss': overfit['initial']['loss'],
                'overfit_final_loss': overfit['final']['loss'], 'overfit_final_dice_percent': 100 * overfit['final']['dice'],
                'next_update_model_optimizer_scheduler_loss_exact': True, 'engine_epochs': 2, 'engine_steps': 24}

    def text_contract(self):
        rows = self.manifests()
        all_rows = [r for split in COUNTS for r in rows[split]]
        expected_ids = [r['sample_id'] for r in all_rows]
        configs = {arm: self.read(f'configs/training_monuseg_{arm}_v1.json') for arm in SOURCES}
        require({k: v for k, v in configs['legacy'].items() if k not in ('source', 'feature_cache')}
                == {k: v for k, v in configs['qwen'].items() if k not in ('source', 'feature_cache')}, 'Arms differ beyond source/cache')
        require(configs['legacy']['seed'] == 666 and configs['legacy']['max_epochs'] == 2000
                and configs['legacy']['min_epochs'] == 50 and configs['legacy']['patience'] == 50
                and configs['legacy']['batch_size'] == 2 and configs['legacy']['learning_rate'] == 1e-3
                and configs['legacy']['optimizer'] == 'Adam' and configs['legacy']['precision'] == 'float32'
                and configs['legacy']['amp'] is False and configs['legacy']['deterministic'] is True
                and configs['legacy']['unet_pretrained'] is False and configs['legacy']['geometric_augmentation'] is False,
                'Frozen training budget/precision/pretraining contract differs')
        stores, feature_hashes, detail = {}, {}, {}
        for arm, source in SOURCES.items():
            config = configs[arm]
            require(config['source'] == source, 'Incorrect text arm source')
            path = self.need(config['feature_cache'])
            payload = torch.load(path, map_location='cpu', weights_only=False)
            require(payload['schema_version'] == 1 and payload['sample_ids'] == expected_ids, 'Text cache membership/order differs')
            tensor, records, metadata = payload['features'], payload['records'], payload['metadata']
            require(tuple(tensor.shape) == (44, 10, 768) and tensor.dtype == torch.float32
                    and bool(torch.isfinite(tensor).all()) and tensor.std().item() > 0, 'Text cache tensor invalid')
            require(metadata['source'] == source and metadata['dataset'] == 'monuseg'
                    and metadata['sample_count'] == 44 and metadata['manifest_sha256'] == self.manifest_hashes
                    and metadata['protocol_sha256'] == self.protocol_sha, 'Text cache provenance differs')
            record_map = {r['sample_id']: r for r in records}
            require(len(records) == len(record_map) == 44 and set(record_map) == set(expected_ids), 'Text record IDs differ')
            for row in all_rows:
                record = record_map[row['sample_id']]
                require(record['image_sha256'] == row['image_sha256'] and record['split'] == row['split'], 'Text record image/split differs')
                require(hashlib.sha256(record['raw_text'].encode()).hexdigest() == record['raw_text_sha256'], 'Cached raw text hash differs')
                require(0 <= record['zero_padded_positions'] <= 10 and len(record['selected_tokens']) + record['zero_padded_positions'] == 10, 'Merged-word positions/padding invalid')
                require(len(record['wordpiece_tokens']) <= 23, 'Encoder exceeded 25 positions including CLS/SEP')
                if arm == 'legacy':
                    require(record['raw_text_sha256'] == row['description_sha256'], 'Legacy workbook text differs from frozen description')
            feature_hashes[arm] = self.sha(config['feature_cache'])
            companion = self.read(config['feature_cache'] + '.json')
            require(companion['cache_sha256'] == feature_hashes[arm] and companion['cache_bytes'] == path.stat().st_size,
                    'Feature cache receipt hash/size differs')
            stores[arm] = payload
            detail[arm] = {'cache_sha256': feature_hashes[arm], 'cache_bytes': path.stat().st_size,
                           'encode_seconds': metadata['encode_seconds'], 'total_cache_seconds': metadata['total_seconds'],
                           'device': metadata['device'], 'padded_sample_count': sum(r['zero_padded_positions'] > 0 for r in records),
                           'selected_tokens_example': records[0]['selected_tokens']}
        encoder = stores['legacy']['metadata']['encoder']
        require(encoder == stores['qwen']['metadata']['encoder'], 'Text encoder/policy differs between arms')
        require(encoder['version'] == 'hf_bert_uncased_last_wordpiece_mean25_first10_eofpad_v1'
                and encoder['max_sequence_length_including_special'] == 25 and encoder['positions'] == 10
                and encoder['aggregation'] == 'last_hidden_state_mean_contiguous_hashhash'
                and encoder['legacy_checkpoint_equivalence'] is False, 'Encoder semantics/equivalence disclosure differs')
        for name, checksum in encoder['model_files_sha256'].items():
            require(self.sha(Path(encoder['model_path']) / name) == checksum, 'BERT encoder file changed: ' + name)
        self.configs, self.stores, self.feature_hashes, self.encoder = configs, stores, feature_hashes, encoder
        return {'settings_difference': ['source', 'feature_cache'], 'common_encoder': encoder,
                'sample_count_per_cache': 44, 'arms': detail, 'legacy_provenance': self.protocol['legacy_text_provenance'],
                'comparison_scope': 'text input conditions; neither image-only ablation nor legacy-checkpoint numerical reproduction'}

    def descriptions(self):
        if not hasattr(self, 'stores'):
            raise Pending('Text contract prerequisite did not pass')
        freeze = self.read('docs/stage234_freeze.json')
        require(freeze['frozen_before_test_generation'] is True and freeze['revisions_after_v2'] == 0,
                'Generation was not frozen before test')
        config = self.read('configs/qwen_generation_v2.json')
        config_hash = hashlib.sha256(json.dumps(config, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        require(config_hash == freeze['generation_config_sha256'], 'Generation config changed after freeze')
        metadata = self.read('cache/qwen/monuseg_plain_v2.metadata.json')
        receipt = self.read('cache/qwen/monuseg_plain_v2.receipt.json')
        raw = self.lines('cache/qwen/monuseg_plain_v2.jsonl')
        require(receipt['status'] == 'complete' and receipt['count'] == len(raw) == 44
                and receipt['config_sha256'] == metadata['config_sha256'] == config_hash, 'Plain cache receipt/count/config invalid')
        require(receipt['cache_sha256'] == self.sha('cache/qwen/monuseg_plain_v2.jsonl'), 'Plain JSONL differs from committed receipt')
        model_files = metadata['model_files']
        model_hash = hashlib.sha256(json.dumps(model_files, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        require(model_hash == metadata['model_sha256'] == receipt['model_sha256'], 'Qwen model fingerprint inconsistent')
        for name, checksum in model_files.items():
            require(self.sha(Path(config['model_directory']) / name) == checksum, 'Qwen model/processor file changed: ' + name)
        rows = {r['sample_id']: r for split in self.rows.values() for r in split}
        require(len({r['sample_id'] for r in raw}) == 44 and {r['sample_id'] for r in raw} == set(rows), 'Plain cache membership differs')
        features = {r['sample_id']: r for r in self.stores['qwen']['records']}
        reuses = {}; generated = reused = 0
        for record in raw:
            row = rows[record['sample_id']]
            for key in ('image_path', 'image_sha256', 'dataset', 'split'):
                require(record[key] == row[key], f'Plain cache image provenance differs: {key}')
            require(not (set(record) & {'mask_path', 'mask', 'label', 'legacy_text', 'text_key', 'Description'}), 'A GT/legacy field entered generation cache')
            require(record['config_sha256'] == config_hash and record['model_sha256'] == model_hash, 'Mixed generation model/config')
            require(record['plain_text'] == features[record['sample_id']]['raw_text'], 'Cached BERT text differs from Qwen plain text')
            if 'plain_reused_from' in record:
                path = record['plain_reused_from']
                if path not in reuses:
                    previous = self.lines(path)
                    reuses[path] = {r['sample_id']: r for r in previous}
                previous = reuses[path][record['sample_id']]
                for key in ('plain_text', 'image_sha256', 'config_sha256', 'model_sha256'):
                    require(previous[key] == record[key], 'Reused description does not match its provenance')
                reused += 1
            else:
                generated += 1
            archive = record.get('generator_source_archive')
            if archive:
                require(self.sha(archive) == record['generator_sha256'], 'Per-record generation source archive changed')
            for key in ('generate_seconds', 'preprocess_seconds'):
                finite_positive(record['timing']['plain'][key], 'plain ' + key, allow_zero=True)
        require(receipt['records_existing_on_entry'] + receipt['records_written_this_invocation'] == 44
                and receipt['plain_generated_this_invocation'] + receipt['plain_reused_this_invocation'] == receipt['records_written_this_invocation'],
                'Generation/reuse invocation cost accounting differs')
        if receipt['records_existing_on_entry'] == 0:
            require(generated == receipt['plain_generated_this_invocation'] and reused == receipt['plain_reused_this_invocation'], 'Full-cache generation origin counts differ')
        require(self.stores['qwen']['metadata']['generation_model_sha256'] == [model_hash]
                and self.stores['qwen']['metadata']['generation_config_sha256'] == [config_hash]
                and self.stores['qwen']['metadata']['descriptions_sha256'] == self.sha('cache/qwen/monuseg_plain_v2.jsonl'), 'BERT cache is not built from this frozen Qwen cache')
        self.freeze, self.qwen_model_hash, self.qwen_config_hash = freeze, model_hash, config_hash
        return {'sample_count': 44, 'generation_config_sha256': config_hash, 'generation_model_sha256': model_hash,
                'full_cache_newly_generated': generated, 'full_cache_reused_from_pilot': reused,
                'cache_bytes': self.need('cache/qwen/monuseg_plain_v2.jsonl').stat().st_size,
                'new_full_cache_process_wall_seconds': receipt['wall_seconds'],
                'new_full_cache_model_load_seconds': receipt['model_load_seconds'],
                'all_44_plain_generation_seconds_including_reused_original_calls': sum(r['timing']['plain']['generate_seconds'] for r in raw),
                'newly_generated_plain_seconds': sum(r['timing']['plain']['generate_seconds'] for r in raw if 'plain_reused_from' not in r),
                'cost_limitation': 'Reused pilot generation timings are not new full-cache wall cost; pilot structured generation is separate.',
                'hit_plain_token_limit_count': sum(bool(r['timing']['plain']['hit_token_limit']) for r in raw),
                'gpu_sampler': receipt.get('gpu')}

    def existing_cost_analysis(self):
        if not hasattr(self, 'stores') or not hasattr(self, 'qwen_model_hash'):
            raise Pending('Text/generation prerequisites did not pass')
        cost = self.read('outputs/stage234/cache_cost_analysis.json')
        from transformers import BertTokenizer
        tokenizer = BertTokenizer.from_pretrained(self.path(self.encoder['model_path']), local_files_only=True, do_lower_case=True)
        for arm in SOURCES:
            payload, recorded = self.stores[arm], cost[arm]
            records, metadata = payload['records'], payload['metadata']
            texts = [r['raw_text'] for r in records]
            prepared = [(text + ' EOF XXX' * max(0, 9 - len(text.split()))).split('\n')[0] for text in texts]
            raw_lengths = [len(tokenizer(text.split('\n')[0], add_special_tokens=True, truncation=False)['input_ids']) for text in texts]
            prepared_lengths = [len(tokenizer(text, add_special_tokens=True, truncation=False)['input_ids']) for text in prepared]
            require(recorded['sample_count'] == 44 and recorded['feature_shape'] == list(payload['features'].shape)
                    and recorded['cache_bytes'] == self.need(self.configs[arm]['feature_cache']).stat().st_size, 'Existing cache cost dimensions/bytes differ')
            near(recorded['encode_seconds'], metadata['encode_seconds'], 'Feature encode cost')
            near(recorded['cache_build_seconds'], metadata['total_seconds'], 'Feature build cost')
            require(recorded['raw_before_filler_wordpiece_length_max'] == max(raw_lengths)
                    and recorded['raw_before_filler_max25_truncated_count'] == sum(n > 25 for n in raw_lengths)
                    and recorded['wordpiece_length_with_special_min'] == min(prepared_lengths)
                    and recorded['wordpiece_length_with_special_max'] == max(prepared_lengths)
                    and recorded['max25_truncated_count'] == sum(n > 25 for n in prepared_lengths), 'Existing raw/filler WordPiece truncation analysis differs')
            require(recorded['first10_merged_truncated_count'] == sum(r['available_merged_tokens'] > 10 for r in records)
                    and recorded['zero_padded_count'] == sum(r['zero_padded_positions'] > 0 for r in records)
                    and recorded['legacy_eof_filler_input_count'] == sum(len(text.split()) < 9 for text in texts), 'Existing merged-word/padding analysis differs')
        plain = self.lines('cache/qwen/monuseg_plain_v2.jsonl')
        generation = cost['qwen_generation']
        require(generation['sample_count'] == 44 and generation['description_cache_bytes'] == self.need('cache/qwen/monuseg_plain_v2.jsonl').stat().st_size, 'Existing description cost size/count differs')
        near(generation['generation_seconds_sum_unique_images'], sum(r['timing']['plain']['generate_seconds'] for r in plain), 'Existing generation-only cost')
        near(generation['preprocess_seconds_sum_unique_images'], sum(r['timing']['plain']['preprocess_seconds'] for r in plain), 'Existing preprocessing cost')
        require(generation['output_tokens_sum'] == sum(r['timing']['plain']['output_tokens'] for r in plain)
                and generation['max_token_limit_hits'] == sum(bool(r['timing']['plain']['hit_token_limit']) for r in plain)
                and generation['reused_pilot_descriptions'] == sum('plain_reused_from' in r for r in plain), 'Existing token/reuse cost analysis differs')
        return {'evidence': 'outputs/stage234/cache_cost_analysis.json', 'generation_seconds_unique_44': generation['generation_seconds_sum_unique_images'],
                'raw_before_filler_max_length': {arm: cost[arm]['raw_before_filler_wordpiece_length_max'] for arm in SOURCES},
                'raw_description_max25_truncation_count': {arm: cost[arm]['raw_before_filler_max25_truncated_count'] for arm in SOURCES},
                'prepared_after_filler_max25_truncation_count': {arm: cost[arm]['max25_truncated_count'] for arm in SOURCES},
                'first10_merged_positions_truncation_count': {arm: cost[arm]['first10_merged_truncated_count'] for arm in SOURCES},
                'interpretation': '25-position truncation affects filler tails here; the subsequent 10 merged-position LViT interface still truncates token sequences.'}

    def training_run(self, arm, seed):
        base = f'outputs/lvit/monuseg/{arm}_seed{seed}'
        result = self.read(base + '/result.json')
        if result.get('status') != 'completed':
            raise Pending(f'{arm}/{seed} training is not complete')
        if not hasattr(self, 'stores'):
            raise Pending('Common text contract prerequisite did not pass')
        config = dict(self.configs[arm]); config['seed'] = seed
        saved = self.read(base + '/run_config.json')
        require(saved['config'] == config and saved['text_encoder'] == self.encoder
                and saved['counts'] == COUNTS and saved['parameter_count'] == 39933345, 'Effective run config/encoder/counts differs')
        identity = saved['identity']
        require(identity['config_sha256'] == hashlib.sha256(json.dumps(config, sort_keys=True, separators=(',', ':')).encode()).hexdigest(), 'Effective config fingerprint differs')
        require(identity['protocol_sha256'] == self.protocol_sha and identity['manifest_sha256'] == self.manifest_hashes
                and identity['feature_cache_sha256'] == self.feature_hashes[arm], 'Run assets/protocol/text source differs')
        for path, checksum in identity['code_sha256'].items():
            require(self.sha(path) == checksum, 'Frozen code changed: ' + path)
        require(identity['metrics_code_sha256'] == self.sha('scripts/protocol_metrics.py'), 'Canonical metric code changed')
        history = self.lines(base + '/history.jsonl')
        count = result['epochs_completed']
        require([r['epoch'] for r in history] == list(range(1, count + 1)), 'History has missing/duplicate/out-of-order epochs; cannot audit earliest best')
        best, best_epoch, first_stop = -1.0, -1, None
        for row in history:
            epoch = row['epoch']; val = row['validation']
            require(row['global_step'] == epoch * 12 and row['train_sample_count'] == 24 and val['sample_count'] == 6, 'Epoch step/sample denominators differ')
            require(0 <= val['dice'] <= 1 and 0 <= val['iou'] <= 1, 'Validation metric outside [0,1]')
            near(val['dice_percent'], 100 * val['dice'], 'Validation percent/fraction Dice')
            near(val['iou_percent'], 100 * val['iou'], 'Validation percent/fraction IoU')
            expected_lr = config['scheduler_eta_min'] + (config['learning_rate'] - config['scheduler_eta_min']) * (1 + math.cos(math.pi * ((epoch - 1) % 10) / 10)) / 2
            near(row['learning_rate'], expected_lr, 'Original cosine restart schedule', 1e-12)
            for key in ('train_loss', 'train_seconds', 'gradient_l2_min', 'gradient_l2_max'):
                finite_positive(row[key], key)
            require(0 <= row['prediction_min'] <= row['prediction_max'] <= 1, 'Training probability range invalid')
            if val['dice'] > best:
                best, best_epoch = val['dice'], epoch
            near(row['best_validation_dice'], best, 'Running best score')
            require(row['best_epoch'] == best_epoch, 'Running best did not retain earliest validation tie')
            if first_stop is None and epoch >= config['min_epochs'] and epoch - best_epoch >= config['patience']:
                first_stop = epoch
        require(1 <= count <= config['max_epochs'] and count >= config['min_epochs'], 'Training budget bounds violated')
        expected_end = first_stop if first_stop is not None else config['max_epochs']
        require(count == expected_end, 'Run stopped before/after the frozen early-stop rule')
        require(result['stop_reason'] == ('validation_patience' if first_stop is not None else 'max_epochs'), 'Stop reason differs')
        require(result['best_epoch'] == best_epoch and result['global_step'] == count * 12
                and result['test_evaluated'] is False, 'Final train state/best/test-access differs')
        near(result['best_validation_dice'], best, 'Final selected validation Dice')
        checkpoint_hashes = {}
        for name, epoch in (('best.pt', best_epoch), ('last.pt', count)):
            checkpoint = torch.load(self.need(base + '/' + name), map_location='cpu', weights_only=False)
            require(checkpoint['schema_version'] == 1 and checkpoint['model_version'] == 'uploaded_lvit_binary_224_v1'
                    and checkpoint['identity'] == identity and checkpoint['config'] == config
                    and checkpoint['epoch'] + 1 == epoch, 'Checkpoint contract/selected epoch differs')
            require(checkpoint['global_step'] == epoch * 12 and checkpoint['resume_granularity'] == 'completed_epoch', 'Checkpoint update count/resume scope differs')
            require({'python', 'numpy', 'torch', 'cuda'} <= checkpoint['rng_state'].keys()
                    and checkpoint['optimizer_state']['state'] and checkpoint['scheduler_state'] is not None, 'Checkpoint lacks RNG/Adam/scheduler recovery state')
            require(len(checkpoint['model_state']) > 0 and all(bool(torch.isfinite(v).all()) for v in checkpoint['model_state'].values() if torch.is_floating_point(v)), 'Checkpoint weights nonfinite')
            if name == 'best.pt':
                require(checkpoint['best_epoch'] + 1 == best_epoch, 'best.pt was not selected at its own epoch')
                near(checkpoint['best_validation_dice'], best, 'Best checkpoint score')
            else:
                require(checkpoint['best_epoch'] + 1 == best_epoch, 'last.pt lost selected-best tracking')
            del checkpoint
            checkpoint_hashes[name] = self.sha(base + '/' + name)
        selected = self.read(base + '/best_validation.json')
        require(selected['sample_count'] == len(selected['samples']) == 6
                and {r['sample_id'] for r in selected['samples']} == {r['sample_id'] for r in self.rows['val']}, 'Best validation samples differ')
        near(sum(r['dice'] for r in selected['samples']) / 6, best, 'Best validation macro reaggregation')
        finite_positive(result['training_elapsed_seconds'], 'training wall time')
        detail = {'arm': arm, 'source': SOURCES[arm], 'seed': seed, 'epochs': count, 'optimizer_steps': count * 12,
                  'best_epoch': best_epoch, 'best_validation_dice': best, 'stop_reason': result['stop_reason'],
                  'tie_policy_independently_recomputed': 'earliest', 'checkpoint_sha256': checkpoint_hashes,
                  'training_elapsed_seconds': result['training_elapsed_seconds'],
                  'train_seconds_sum': sum(r['train_seconds'] for r in history),
                  'validation_seconds_sum': sum(r['validation']['seconds'] for r in history),
                  'steady_train_samples_per_second_median': statistics.median(r['train_samples_per_second'] for r in history[1:] or history),
                  'peak_allocated_bytes': result['peak_allocated_bytes'], 'peak_reserved_bytes': result['peak_reserved_bytes']}
        self.training[(arm, seed)] = detail
        return detail

    def test_run(self, arm, seed):
        if len(self.training) != 6:
            raise Pending('Test predictions deliberately not opened until all six training audits pass')
        base = f'outputs/lvit/monuseg/{arm}_seed{seed}'
        metrics = self.read(base + '/test/test_metrics.json')
        train = self.training[(arm, seed)]
        require(metrics['dataset'] == 'monuseg' and metrics['source'] == SOURCES[arm] and metrics['seed'] == seed
                and metrics['sample_count'] == 14 and metrics['manifest_sha256'] == self.manifest_hashes['test'], 'Test source/seed/denominator differs')
        require(metrics['checkpoint_sha256'] == train['checkpoint_sha256']['best.pt']
                and metrics['checkpoint_epoch'] == metrics['checkpoint_best_epoch'] == train['best_epoch']
                and metrics['code_matches_training'] is True, 'Test used a different/non-best checkpoint or code')
        records = {r['sample_id']: r for r in metrics['samples']}
        require(len(metrics['samples']) == len(records) == 14 and set(records) == {r['sample_id'] for r in self.rows['test']}, 'Test sample IDs duplicate/missing')
        prediction_dir = self.need(base + '/test/test_metrics.json').parent / 'predictions'
        require({p.stem for p in prediction_dir.glob('*.png')} == set(records), 'Predicted PNG membership differs')
        scores = []
        for row in self.rows['test']:
            sid = row['sample_id']
            prediction = cv2.imread(str(self.need(prediction_dir / (sid + '.png'))), cv2.IMREAD_GRAYSCALE)
            require(prediction is not None and prediction.shape == (224, 224)
                    and set(np.unique(prediction).tolist()) <= {0, 255}, 'Prediction PNG is not binary 224x224')
            gray = cv2.imread(str(self.need(row['mask_path'])), cv2.IMREAD_GRAYSCALE)
            require(gray is not None, 'Cannot decode canonical GT')
            # Independent preprocessing, deliberately not experiment.data.load_mask.
            target = cv2.resize((gray > 0).astype(np.uint8), (224, 224), interpolation=cv2.INTER_NEAREST)
            score = binary_sample_scores(prediction.astype(np.float32) / 255.0, target, threshold=0.5)
            require(records[sid]['image_path'] == row['image_path'], 'Test image path mismatch')
            for key in ('dice', 'iou'):
                near(records[sid][key], score[key], f'{arm}/{seed}/{sid}/{key}')
            scores.append({'sample_id': sid, **score, 'prediction_sha256': self.sha(prediction_dir / (sid + '.png'))})
        # Do not call macro_scores: aggregate the actual 14 per-image values here.
        dice, iou = sum(r['dice'] for r in scores) / len(scores), sum(r['iou'] for r in scores) / len(scores)
        for key, value in (('dice', dice), ('iou', iou), ('dice_percent', 100 * dice), ('iou_percent', 100 * iou)):
            near(metrics[key], value, 'Independent test macro ' + key, 1e-9)
        detail = {'arm': arm, 'seed': seed, 'source': SOURCES[arm], 'sample_count': len(scores),
                  'dice_percent': 100 * dice, 'iou_percent': 100 * iou, 'checkpoint_sha256': metrics['checkpoint_sha256'],
                  'independent_png_gt_recompute': True, 'scores': scores}
        self.tests[(arm, seed)] = detail
        return detail

    def ordering_and_summary(self):
        if len(self.tests) != 6:
            raise Pending('Six independently recomputed test runs not yet available')
        latest_train_end, earliest_test_start = 0.0, float('inf')
        for arm, seed in self.tests:
            train_base = f'outputs/stage234/formal_experiments/train_{arm}_{seed}'
            test_base = f'outputs/stage234/formal_experiments/test_{arm}_{seed}'
            process, receipt = self.read(train_base + '/process.json'), self.read(train_base + '/receipt.json')
            require(receipt['exit_code'] == 0, 'Training subprocess failed')
            latest_train_end = max(latest_train_end, process['started_at'] + receipt['wall_seconds'])
            process, receipt = self.read(test_base + '/process.json'), self.read(test_base + '/receipt.json')
            require(receipt['exit_code'] == 0, 'Test subprocess failed')
            earliest_test_start = min(earliest_test_start, process['started_at'])
        require(earliest_test_start + 1 >= latest_train_end, 'A test subprocess began before all six training runs ended')
        summary = self.read('outputs/stage234/formal_experiments/summary.json')
        require(summary['status'] == 'complete' and summary['test_count_per_run'] == 14 and summary['seeds'] == list(SEEDS), 'Formal summary denominator/seeds invalid')
        records = {(r['arm'], r['seed']): r for r in summary['records']}
        require(set(records) == set(self.tests) and len(summary['records']) == 6, 'Summary records missing/duplicate')
        aggregates = {}
        for arm in SOURCES:
            subset = [self.tests[(arm, seed)] for seed in SEEDS]
            aggregates[arm] = {}
            for key in ('dice_percent', 'iou_percent'):
                values = [r[key] for r in subset]
                aggregates[arm][key] = {'mean': statistics.mean(values), 'sample_std': statistics.stdev(values)}
                for stat, value in aggregates[arm][key].items():
                    near(summary['aggregate'][arm][key][stat], value, 'Independent summary ' + arm + '/' + key + '/' + stat, 1e-9)
            for seed in SEEDS:
                for key in ('dice_percent', 'iou_percent'):
                    near(records[(arm, seed)][key], self.tests[(arm, seed)][key], 'Summary per-run ' + key)
        differences = []
        for seed in SEEDS:
            reported = next(r for r in summary['paired_differences'] if r['seed'] == seed)
            difference = {'seed': seed}
            for key in ('dice', 'iou'):
                value = self.tests[('qwen', seed)][key + '_percent'] - self.tests[('legacy', seed)][key + '_percent']
                near(reported[key + '_percentage_points'], value, 'Paired difference', 1e-9)
                difference[key + '_percentage_points'] = value
            differences.append(difference)
        return {'all_six_training_finished_before_first_test': True, 'latest_training_end_unix': latest_train_end,
                'earliest_test_start_unix': earliest_test_start, 'aggregate_independently_recomputed': aggregates,
                'paired_differences': differences, 'interpretation': summary['comparison_limitation']}

    def joint(self):
        if len(self.tests) != 6 or not hasattr(self, 'qwen_model_hash'):
            raise Pending('Joint audit waits for six test audits and frozen description contract')
        base = 'outputs/stage234/joint_inference'
        result = self.read(base + '/pipeline_result.json')
        segmentation = self.read(base + '/segmentation/inference.json')
        generated = self.lines(base + '/description.jsonl')
        receipt = self.read(base + '/description.receipt.json')
        selected = sorted(self.rows['test'], key=lambda r: r['sample_id'])[0]
        require(result['status'] == 'complete' and result['mask_input'] is False and result['selected_seed'] == 666,
                'Joint pipeline is incomplete/claims a mask/different preselected seed')
        require(Path(result['input_image']).resolve() == self.need(selected['image_path'])
                and result['input_image_sha256'] == self.sha(selected['image_path']) == selected['image_sha256'], 'Joint input is not the preselected held-out image')
        require(result['checkpoint_sha256'] == self.training[('qwen', 666)]['checkpoint_sha256']['best.pt'], 'Joint used a test-selected/different seed checkpoint')
        require(len(generated) == 1 and generated[0]['split'] == 'new_image'
                and generated[0]['image_sha256'] == selected['image_sha256'], 'Joint generator did not consume the selected image alone')
        require(result['description'] == generated[0]['plain_text']
                and result['qwen_model_sha256'] == generated[0]['model_sha256'] == self.qwen_model_hash
                and result['generation_config_sha256'] == generated[0]['config_sha256'] == self.qwen_config_hash, 'Joint Qwen contract differs from trained descriptions')
        require(receipt['status'] == 'complete' and receipt['count'] == 1
                and receipt['cache_sha256'] == self.sha(base + '/description.jsonl'), 'Joint generated-cache receipt differs')
        require(result['segmentation'] == segmentation and segmentation['text_encoder'] == self.encoder
                and segmentation['training_source'] == 'qwen_predicted_text'
                and segmentation['checkpoint_sha256'] == result['checkpoint_sha256']
                and segmentation['image_sha256'] == selected['image_sha256'], 'Joint BERT/LViT/input checkpoint differs')
        require(segmentation['mask_grid'] == [224, 224] and segmentation['threshold'] == 0.5, 'Joint probability grid/threshold differs')
        probability = np.load(self.need(base + '/segmentation/probability_224.npy'), allow_pickle=False)
        binary = cv2.imread(str(self.need(base + '/segmentation/mask_224.png')), cv2.IMREAD_GRAYSCALE)
        require(probability.shape == (224, 224) and np.isfinite(probability).all() and probability.min() >= 0 and probability.max() <= 1, 'Joint probability array invalid')
        require(binary is not None and binary.shape == (224, 224) and set(np.unique(binary).tolist()) <= {0, 255}
                and np.array_equal(binary, (probability >= 0.5).astype(np.uint8) * 255), 'Joint PNG differs from probability>=0.5')
        near(segmentation['foreground_fraction'], float((binary > 0).mean()), 'Joint foreground fraction')
        process = self.read('outputs/stage234/formal_experiments/joint_inference/process.json')
        forbidden = {'--mask', '--mask-file', '--mask-path', '--label', '--gt', '--ground-truth', '--target-mask'}
        require(not (set(process['argv']) & forbidden), 'Joint invocation contains a GT/mask parameter')
        source = ast.parse(self.need('scripts/infer_qwen_lvit.py').read_text())
        arguments = {call.args[0].value for call in ast.walk(source) if isinstance(call, ast.Call)
                     and isinstance(call.func, ast.Attribute) and call.func.attr == 'add_argument'
                     and call.args and isinstance(call.args[0], ast.Constant) and isinstance(call.args[0].value, str)}
        require(arguments == {'--image', '--dataset', '--checkpoint', '--generation-config', '--output-dir'}, 'Joint CLI has an unreviewed input surface')
        generator = ast.parse(self.need('scripts/generate_qwen_descriptions.py').read_text())
        fields = next(ast.literal_eval(node.value) for node in generator.body if isinstance(node, ast.Assign)
                      and any(isinstance(target, ast.Name) and target.id == 'IMAGE_FIELDS' for target in node.targets))
        require(set(fields) == {'sample_id', 'dataset', 'split', 'image_path', 'image_sha256', 'width', 'height'}, 'Generator admitted mask/legacy fields')
        return {'selected_seed_before_test': 666, 'held_out_demo_sample_id': selected['sample_id'],
                'input_sha256': selected['image_sha256'], 'mask_parameter_absent': True,
                'mask_contract_verification': 'actual queue argv + CLI/IMAGE_FIELDS static surface, not a filesystem syscall trace',
                'probability_png_threshold_exact': True, 'foreground_fraction': segmentation['foreground_fraction'],
                'qwen_process_wall_seconds': result['qwen_process_wall_seconds'],
                'qwen_generate_seconds_recorded': result['qwen_generation_seconds'],
                'qwen_new_records_this_invocation': result['qwen_records_generated_this_invocation'],
                'bert_lvit_process_wall_seconds': result['bert_lvit_process_wall_seconds'],
                'end_to_end_wall_seconds': result['end_to_end_wall_seconds'],
                'generalization_limit': 'Demonstration uses a preselected held-out dataset image, not a newly acquired external clinical image.'}

    def vendor_and_assets(self):
        snapshot = self.read('src/lvit/source_snapshot.json')
        for row in snapshot['files']:
            path = 'src/lvit/' + row['path']
            require(self.need(path).stat().st_size == row['bytes'] and self.sha(path) == row['sha256'], 'Uploaded vendor snapshot changed: ' + row['path'])
        inventory = self.read('manifests/legacy_subset_v1/dataset_inventory.json')
        with self.need('manifests/legacy_subset_v1/asset_hashes.csv').open(newline='') as handle:
            assets = list(csv.DictReader(handle))
        require(len(assets) == inventory['data_files'] == 1845, 'Frozen asset inventory length differs')
        if self.asset_scope == 'monuseg':
            assets = [r for r in assets if r['path'].startswith('monuseg/')]
        source_root = Path(inventory['data_source'])
        if not source_root.is_dir():
            raise Pending('Original source dataset root is unavailable for final byte comparison')
        byte_count = 0
        for row in assets:
            relative = 'datasets/' + row['path']; path = self.need(relative)
            require(not (ROOT / relative).is_symlink() and path.stat().st_size == int(row['bytes'])
                    and self.sha(relative) == row['sha256'], 'Copied canonical data asset changed: ' + row['path'])
            original_raw = source_root / row['path']
            original = original_raw.resolve()
            require(original.is_relative_to(source_root.resolve()) and original.is_file() and not original_raw.is_symlink(), 'Original source asset missing/unsafe: ' + row['path'])
            digest = hashlib.sha256()
            with original.open('rb') as handle:
                for chunk in iter(lambda: handle.read(8 * 1024**2), b''):
                    digest.update(chunk)
            require(original.stat().st_size == int(row['bytes']) and digest.hexdigest() == row['sha256'], 'Original uploaded dataset changed: ' + row['path'])
            byte_count += int(row['bytes'])
        require(snapshot['kind'] == 'user_uploaded_modified_copy', 'Vendor provenance was overstated')
        return {'vendor_snapshot_files_verified': len(snapshot['files']), 'vendor_kind': snapshot['kind'],
                'dataset_asset_scope': self.asset_scope, 'canonical_assets_rehashed': len(assets),
                'original_source_assets_rehashed': len(assets), 'bytes_per_copy_verified': byte_count,
                'all_1845_assets_verified': self.asset_scope == 'all',
                'clinical_annotation_validity': 'Byte identity verifies preservation; it does not certify clinical annotation correctness.'}

    def repository_hygiene(self):
        ignored = ['weights/Qwen2.5-VL-3B-Instruct/model.safetensors', 'datasets/monuseg/Train_Folder/img/example.png',
                   'cache/text_features/example.pt', 'outputs/stage234/final_audit.json', 'src/lvit/LV_loss/fig1.png', '.env']
        visible = ['scripts/audit_stage234_results.py', 'scripts/train_lvit.py', 'src/experiment/engine.py',
                   'src/lvit/LICENSE', 'src/lvit/source_snapshot.json', 'configs/data_protocol_v1.json',
                   'configs/training_monuseg_legacy_v1.json', 'manifests/legacy_subset_v1/monuseg/test.csv',
                   'datasets/README.md', 'weights/README.md', 'cache/README.md', 'outputs/README.md']
        process = subprocess.run(['git', '-C', str(ROOT), 'check-ignore', '--no-index', '-z', '--stdin'],
                                 input='\0'.join(ignored + visible).encode() + b'\0', capture_output=True)
        require(process.returncode in (0, 1), 'git check-ignore failed')
        actual = {p.decode() for p in process.stdout.split(b'\0') if p}
        require(actual == set(ignored), 'Heavy artifacts ignored/source visible rules differ')
        tracked = subprocess.run(['git', '-C', str(ROOT), 'ls-files', '-z'], capture_output=True, check=True)
        allow = {'datasets/README.md', 'weights/README.md', 'cache/README.md', 'outputs/README.md'}
        forbidden = [p.decode() for p in tracked.stdout.split(b'\0') if p and
                     (p.decode().split('/')[0] in {'datasets', 'weights', 'cache', 'outputs'} and p.decode() not in allow)]
        require(not forbidden, 'Large/private runtime artifacts are tracked: ' + repr(forbidden[:10]))
        return {'ignore_rules_verified': ignored, 'source_paths_visible': visible, 'tracked_runtime_artifact_count': 0,
                'git_mutations_performed': False}

    def report(self):
        counts = {s: sum(check['status'] == s for check in self.checks) for s in ('passed', 'pending', 'failed')}
        status = 'failed' if counts['failed'] else 'pending' if counts['pending'] else 'passed'
        return {'schema_version': 1, 'created_at_utc': datetime.now(timezone.utc).isoformat(), 'status': status,
                'audit_script_sha256': self.sha('scripts/audit_stage234_results.py'), 'dataset': 'monuseg',
                'read_only_source_artifacts': True, 'gpu_tasks_started': 0,
                'test_prediction_read_gate': 'all six training audits passed', 'check_counts': counts,
                'checks': self.checks, 'verified_training_runs': len(self.training), 'independently_recomputed_test_runs': len(self.tests),
                'limits': ['This audit reads artifacts; it does not rerun a model forward pass or create new test predictions.',
                           'Visual review and clinical description/localization quality remain separate human/evidence reviews.',
                           'Legacy text provenance is unverified; compare text-input conditions without claiming image-only benefit.']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='outputs/stage234/final_audit.json')
    parser.add_argument('--asset-scope', choices=['all', 'monuseg'], default='all')
    args = parser.parse_args()
    torch.set_num_threads(8)
    audit = Audit(args.asset_scope)
    audit.check('stage2_real_data_learning_and_exact_resume', audit.stage2)
    audit.check('common_text_encoder_manifest_and_budget_contract', audit.text_contract)
    audit.check('frozen_qwen_44_description_cache_and_cost', audit.descriptions)
    audit.check('existing_cache_cost_and_truncation_analysis', audit.existing_cost_analysis)
    for seed in SEEDS:
        for arm in SOURCES:
            audit.check(f'training_{arm}_{seed}', lambda arm=arm, seed=seed: audit.training_run(arm, seed))
    for seed in SEEDS:
        for arm in SOURCES:
            audit.check(f'independent_test_png_gt_{arm}_{seed}', lambda arm=arm, seed=seed: audit.test_run(arm, seed))
    audit.check('all_training_before_test_and_independent_summary', audit.ordering_and_summary)
    audit.check('joint_preselected_input_no_gt_and_probability_threshold', audit.joint)
    audit.check('vendor_snapshot_and_original_dataset_hashes', audit.vendor_and_assets)
    audit.check('repository_ignore_and_tracked_artifact_hygiene', audit.repository_hygiene)
    report = audit.report()
    output = audit.path(args.output)
    require(output.is_relative_to((ROOT / 'outputs').resolve()), 'Audit report must stay inside outputs/')
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + '.tmp')
    with temporary.open('w', encoding='utf-8') as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write('\n');handle.flush();os.fsync(handle.fileno())
    os.replace(temporary, output)
    print(json.dumps({'status': report['status'], 'check_counts': report['check_counts'], 'output': str(output),
                      'verified_training_runs': len(audit.training), 'independently_recomputed_test_runs': len(audit.tests)}, ensure_ascii=False), flush=True)
    raise SystemExit(1 if report['status'] == 'failed' else 0)


if __name__ == '__main__':
    main()

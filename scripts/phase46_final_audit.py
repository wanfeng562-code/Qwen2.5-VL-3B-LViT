"""Independent CPU audit of locked final PNGs; never runs training or inference."""
import csv
import json
import statistics
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.experiment.common import atomic_json, sha256_file

ARMS = ('candidate', 'constant', 'shuffled', 'legacy')
SEEDS = (666, 667, 668)


def read(path):
    return json.loads((ROOT / path).read_text())


def near(a, b):
    assert abs(a - b) < 1e-10, (a, b)


def png_scores(directory, rows, truth, saved):
    expected = {r['sample_id'] for r in rows}
    assert {p.stem for p in directory.glob('*.png')} == expected
    assert {r['sample_id'] for r in saved['samples']} == expected
    scores = []
    for row in rows:
        sid = row['sample_id']
        path = directory / (sid + '.png')
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        assert image is not None and image.shape == (224, 224)
        assert image.dtype == np.uint8 and set(np.unique(image)) <= {0, 255}
        prediction, target = image > 0, truth[sid]
        tp = int(np.count_nonzero(prediction & target))
        fp = int(np.count_nonzero(prediction & ~target))
        fn = int(np.count_nonzero(~prediction & target))
        dice = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 1.0
        iou = tp / (tp + fp + fn) if tp + fp + fn else 1.0
        item = next(x for x in saved['samples'] if x['sample_id'] == sid)
        near(dice, item['dice'])
        near(iou, item['iou'])
        scores.append(dict(sample_id=sid, dice=dice, iou=iou, tp=tp, fp=fp, fn=fn,
                           prediction_sha256=sha256_file(path)))
    for metric in ('dice', 'iou'):
        near(statistics.mean(x[metric] for x in scores), saved[metric])
    assert saved['sample_count'] == 14
    return scores


def main():
    torch.set_num_threads(8)
    auth = read('outputs/phase46/test_authorization.json')
    assert auth['validation_gate_passed']
    audit = read('outputs/phase46/development_audit.json')
    assert audit['status'] == 'passed' and audit['experiment_status'] == 'validation_passed'
    assert sha256_file(ROOT / 'outputs/phase46/development_audit.json') == auth['development_audit_sha256']
    assert sha256_file(ROOT / 'outputs/phase46/confirmation/summary.json') == auth['validation_summary_sha256']
    for name, expected in auth['evaluation_scripts'].items():
        assert sha256_file(ROOT / name) == expected, name
    for name, expected in read('docs/phase46_start.json')['old_tracked_sha256'].items():
        assert sha256_file(ROOT / name) == expected, name
    for name, expected in read('docs/phase46_core_freeze.json').items():
        assert sha256_file(ROOT / name) == expected, name
    assert sha256_file(ROOT / 'configs/phase46/contract.json') == auth['contract_sha256']
    checks = ['development_gate_and_frozen_sources']

    manifest = ROOT / 'manifests/legacy_subset_v1/monuseg/test.csv'
    assert sha256_file(manifest) == auth['test_manifest_sha256']
    with manifest.open(encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 14 and all(r['split'] == 'test' for r in rows)
    truth = {}
    for row in rows:
        for kind in ('image', 'mask'):
            path = (ROOT / row[kind + '_path']).resolve()
            assert path.is_relative_to(ROOT.resolve())
            assert sha256_file(path) == row[kind + '_sha256']
        gray = cv2.imread(str(ROOT / row['mask_path']), cv2.IMREAD_GRAYSCALE)
        assert gray is not None
        truth[row['sample_id']] = cv2.resize((gray > 0).astype(np.uint8), (224, 224),
                                           interpolation=cv2.INTER_NEAREST) > 0
    checks.append('all_test_image_and_mask_hashes_and_frozen_preprocessing')

    dev_path = ROOT / 'cache/phase46/V_mid_development.pt'
    assert sha256_file(dev_path) == auth['development_cache_sha256']
    dev = torch.load(dev_path, map_location='cpu', weights_only=False)
    held_path = ROOT / 'cache/phase46/V_mid_test.pt'
    held = torch.load(held_path, map_location='cpu', weights_only=False)
    assert held['sample_ids'] == [r['sample_id'] for r in rows]
    assert held['metadata']['authorization_sha256'] == sha256_file(ROOT / 'outputs/phase46/test_authorization.json')
    assert held['metadata']['encoder'] == dev['metadata']['encoder']
    assert held['metadata']['extractor_sha256'] == auth['evaluation_scripts']['scripts/phase46_extract_final.py']
    assert not (set(held['sample_ids']) & set(dev['sample_ids']))
    # Independently derive the training subset and means, then compare the shared constructor.
    train = [i for i, r in enumerate(dev['records']) if r['split'] == 'train']
    assert len(train) == 24 and len(dev['records']) == 30
    assert [r['sample_id'] for r in dev['records']] == dev['sample_ids']
    mean_v = dev['features'][train].float().mean(dim=0)
    mean_t = dev['legacy_text_features'][train].float().mean(dim=0)
    from scripts.phase46_final_data import conditions
    for seed in SEEDS:
        for mode in ('correct', 'constant', 'legacy', 'shuffled'):
            lookup, mapping = conditions(dev, held, mode, seed)
            shift = 1 + seed % 13
            for i, sid in enumerate(held['sample_ids']):
                j = (i + shift) % 14 if mode == 'shuffled' else i
                v = mean_v if mode in ('constant', 'legacy') else held['features'][j].float()
                t = held['legacy_text_features'][i].float() if mode == 'legacy' else mean_t
                assert torch.equal(lookup[sid]['vision'], v) and torch.equal(lookup[sid]['text'], t)
                if mode == 'shuffled':
                    assert mapping[sid]['source_sample_id'] != sid
    checks.append('all_conditions_use_train_only_means_and_within_test_derangements')

    final = read('outputs/phase46/final_test/summary.json')
    assert final['status'] == 'completed' and len(final['results']) == 12
    records = {}
    for frozen in auth['records']:
        arm, seed = frozen['arm'], frozen['seed']
        checkpoint = frozen['run'] + '/best.pt'
        assert sha256_file(ROOT / checkpoint) == auth['checkpoints'][checkpoint]
        base = ROOT / 'outputs/phase46/final_test' / f'{arm}_seed{seed}'
        saved = json.loads((base / 'metrics.json').read_text())
        assert saved['arm'] == arm and saved['seed'] == seed
        identity = saved['final_identity']
        assert identity['checkpoint_sha256'] == auth['checkpoints'][checkpoint]
        assert identity['test_cache_sha256'] == sha256_file(held_path)
        assert identity['authorization_sha256'] == sha256_file(ROOT / 'outputs/phase46/test_authorization.json')
        assert identity['test_manifest_sha256'] == auth['test_manifest_sha256']
        cfg = read(frozen['run'] + '/run_config.json')['config']
        assert identity['mapping'] == conditions(dev, held, cfg['feature_mode'], seed)[1]
        scores = png_scores(base / 'predictions', rows, truth, saved)
        summary_record = next(x for x in final['results'] if x['arm'] == arm and x['seed'] == seed)
        assert summary_record == saved
        records[f'{arm}_{seed}'] = scores
    checks.append('all_168_new_prediction_pngs_independently_rescored')

    old = {}
    for seed in SEEDS:
        base = ROOT / f'outputs/lvit/monuseg/legacy_seed{seed}/test'
        saved = json.loads((base / 'test_metrics.json').read_text())
        old[seed] = png_scores(base / 'predictions', rows, truth, saved)
    checks.append('all_42_old_baseline_prediction_pngs_independently_rescored')

    aggregates = {}
    for arm in ARMS:
        aggregates[arm] = {}
        for metric in ('dice', 'iou'):
            per_seed = [100 * statistics.mean(x[metric] for x in records[f'{arm}_{seed}']) for seed in SEEDS]
            aggregates[arm][metric] = dict(mean_percent=statistics.mean(per_seed), sample_sd_percent=statistics.stdev(per_seed))
            for key, value in aggregates[arm][metric].items():
                near(value, final['aggregates'][arm][metric][key])
    deltas = {str(seed): 100 * (statistics.mean(x['dice'] for x in records[f'candidate_{seed}']) -
                              statistics.mean(x['dice'] for x in old[seed])) for seed in SEEDS}
    for key, value in deltas.items():
        near(value, final['seed_deltas_vs_old_pp'][key])
    d, i = aggregates['candidate']['dice']['mean_percent'], aggregates['candidate']['iou']['mean_percent']
    thresholds = auth['success']
    gates = dict(dice_target=d >= thresholds['mean_test_dice_percent_at_least'],
                 iou_target=i >= thresholds['mean_test_iou_percent_at_least'],
                 beats_fixed=d - aggregates['constant']['dice']['mean_percent'] >= thresholds['beats_fixed_and_shuffled_pp'],
                 beats_shuffled=d - aggregates['shuffled']['dice']['mean_percent'] >= thresholds['beats_fixed_and_shuffled_pp'],
                 not_below_matched_legacy=d >= aggregates['legacy']['dice']['mean_percent'],
                 two_seeds_improved=sum(x > 0 for x in deltas.values()) >= thresholds['improves_old_baseline_at_least_seeds'])
    assert gates == final['success_gates'] and all(gates.values()) == final['target_achieved']
    checks.append('three_seed_aggregates_and_all_six_success_gates_recomputed')
    per_image = []
    for row in rows:
        sid = row['sample_id']
        item = dict(sample_id=sid)
        for arm in ARMS:
            for metric in ('dice', 'iou', 'fp', 'fn'):
                value = statistics.mean(next(x[metric] for x in records[f'{arm}_{seed}'] if x['sample_id'] == sid) for seed in SEEDS)
                item[arm + '_' + metric] = value
        per_image.append(item)
    out = ROOT / 'outputs/phase46/final_test'
    with (out / 'audited_per_image.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(per_image[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(per_image)
    result = dict(status='passed', check_count=len(checks), checks=checks, new_pngs=168, old_pngs=42,
                  aggregates=aggregates, seed_deltas_vs_old_pp=deltas, success_gates=gates,
                  target_achieved=all(gates.values()), records=records, per_image=per_image,
                  scope=auth['scope'], script_sha256=sha256_file(Path(__file__)))
    atomic_json(out / 'independent_audit.json', result)
    print(json.dumps({k: v for k, v in result.items() if k not in ('records', 'per_image')}))


if __name__ == '__main__':
    main()

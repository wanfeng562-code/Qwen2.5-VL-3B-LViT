#!/usr/bin/env python3
"""Deterministic coverage-based train/val pilot; GT excluded from generator file."""
import csv
import json
from pathlib import Path

import numpy as np
import cv2
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
FIELDS = ('sample_id', 'dataset', 'split', 'image_path', 'image_sha256', 'width', 'height')


def load_rows(dataset, split):
    path = ROOT / f'manifests/legacy_subset_v1/{dataset}/{split}.csv'
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def measures(r):
    image = np.asarray(Image.open(ROOT/r['image_path']).convert('L').resize((224,224), Image.Resampling.BILINEAR), dtype=np.float32)
    # Evaluate foreground in the original mask to avoid eliminating tiny components.
    gt = np.asarray(Image.open(ROOT/r['mask_path']).convert('L')) > 0
    count = cv2.connectedComponents(gt.astype(np.uint8), connectivity=8)[0] - 1
    return {'foreground_fraction': float(gt.mean()), 'components_8connected': int(count),
            'gray_mean_224': float(image.mean()), 'laplacian_variance_224': float(cv2.Laplacian(image, cv2.CV_32F).var())}


def average_ranks(values):
    values = np.asarray(values)
    order = np.argsort(values, kind='stable')
    ranked = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranked[order[start:end]] = (start + end - 1) / 2
        start = end
    return ranked


def choose(rows, count):
    rows = sorted(rows, key=lambda x: x['sample_id'])
    metrics = [measures(r) for r in rows]
    keys = list(metrics[0])
    # Average ranks handle ties without arbitrary assignment of distinct percentiles.
    vectors = np.column_stack([average_ranks([m[k] for m in metrics]) / max(1, len(rows)-1) for k in keys])
    if count >= len(rows):
        chosen = list(range(len(rows)))
    else:
        # Start farthest from the percentile-space center; then greedily maximize
        # minimum distance to selected samples. Stable sorted ID breaks exact ties.
        chosen = [int(np.argmax(np.sum((vectors - .5)**2, axis=1)))]
        while len(chosen) < count:
            distances = ((vectors[:,None,:]-vectors[chosen][None,:,:])**2).sum(axis=2).min(axis=1)
            distances[chosen] = -1
            chosen.append(int(np.argmax(distances)))
    result = []
    for order, i in enumerate(chosen):
        result.append({'row': rows[i], 'metrics': metrics[i], 'percentiles': dict(zip(keys, map(float, vectors[i]))), 'selection_order':order+1})
    return result


def main():
    selected = []
    for dataset in ('monuseg', 'cov19', 'mosmeddata'):
        for split in ('train', 'val'):
            rows = load_rows(dataset, split)
            n = len(rows) if dataset == 'monuseg' else {'train':12, 'val':3}[split]
            selected.extend(choose(rows, n))
    ids = [x['row']['sample_id'] for x in selected]
    assert len(ids) == 60 and len(set(ids)) == 60
    records = []
    for x in selected:
        r = {key: x['row'][key] for key in FIELDS}
        r['width'], r['height'] = int(r['width']), int(r['height'])
        records.append(r)
    config = {'schema_version':1, 'protocol':'legacy_subset_v1', 'records':records}
    selection = {'protocol':'legacy_subset_v1', 'scope':'train/val only; purposefully sampled, not population estimates',
                 'algorithm':'Within each dataset/split rank four metrics; start farthest from 0.5 in rank space, then maximize minimum squared Euclidean distance; ID breaks ties. Monuseg includes all.',
                 'measurements':'Original mask >0, 8-connected components; image grayscale 224 bilinear mean and Laplacian variance. Metrics describe selected samples and are not model inputs.',
                 'records':[{'sample_id':x['row']['sample_id'], 'dataset':x['row']['dataset'], 'split':x['row']['split'], 'selection_order':x['selection_order'], 'metrics':x['metrics'], 'percentiles':x['percentiles']} for x in selected]}
    config_path = ROOT/'configs/qwen_pilot_v1.json'
    out = ROOT/'outputs/visual_review/pilot_selection.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    for p in (config_path, out):
        if p.exists():
            raise FileExistsError(f'Refusing to overwrite: {p}')
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    out.write_text(json.dumps(selection, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({'selected':len(records), 'selection':str(config_path), 'audit':str(out)}))


if __name__ == '__main__':
    main()

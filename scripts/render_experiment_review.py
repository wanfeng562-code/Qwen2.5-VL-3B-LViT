#!/usr/bin/env python3
"""Create readable, labeled image/GT and prediction comparison sheets (CPU only)."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import cv2
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def inside(path):
    p = (ROOT / path).resolve()
    if not p.is_relative_to(ROOT):
        raise ValueError(f'Path escapes project: {path}')
    return p


def mask(path, size):
    gray=cv2.imread(str(inside(path)),cv2.IMREAD_GRAYSCALE)
    if gray is None: raise ValueError(f'Cannot decode mask: {path}')
    return cv2.resize((gray>0).astype(np.uint8),size,interpolation=cv2.INTER_NEAREST)>0


def prediction(path, size):
    im = Image.open(inside(path)).convert('L')
    if im.size != (224,224):
        raise ValueError(f'Expected 224x224 binary prediction: {path}, got {im.size}')
    values = set(np.unique(np.asarray(im)).tolist())
    if not (values <= {0,1} or values <= {0,255}):
        raise ValueError(f'Prediction must be a binary mask: {path}, values={sorted(values)}')
    return cv2.resize(np.asarray(im),size,interpolation=cv2.INTER_NEAREST)>0


def dice(gt, pred):
    n = int(gt.sum()) + int(pred.sum())
    return 2 * int((gt & pred).sum()) / n if n else 1.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--manifest', required=True, help='CSV with sample_id,dataset,split,image_path,mask_path')
    p.add_argument('--output-dir', required=True)
    p.add_argument('--predictions', help='CSV keyed by sample_id, with baseline_path,qwen_path; paths are binary masks')
    p.add_argument('--per-sheet', type=int, default=6)
    p.add_argument('--select-comparison', action='store_true', help='select baseline worst, median and best by 224 Dice, ties by sample_id')
    args = p.parse_args()
    out = inside(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(inside(args.manifest), encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    ids = [r['sample_id'] for r in rows]
    if len(set(ids)) != len(ids):
        raise ValueError('Duplicate sample_id')
    predictions = {}
    if args.predictions:
        with open(inside(args.predictions), encoding='utf-8-sig', newline='') as f:
            pred_rows = list(csv.DictReader(f))
        predictions = {r['sample_id']: r for r in pred_rows}
        if len(predictions) != len(pred_rows) or set(ids) - predictions.keys():
            raise ValueError('Duplicate or missing prediction ids')
    if args.select_comparison and not predictions:
        raise ValueError('--select-comparison requires --predictions')
    ranking = None
    if args.select_comparison:
        ranking = []
        for row in rows:
            r = predictions[row['sample_id']]
            gt224 = mask(row['mask_path'], (224,224))
            base = dice(gt224, prediction(r['baseline_path'], (224,224)))
            qwen = dice(gt224, prediction(r['qwen_path'], (224,224)))
            ranking.append({'sample_id':row['sample_id'], 'baseline_dice':base, 'qwen_dice':qwen})
        ranking.sort(key=lambda r:(r['baseline_dice'],r['sample_id']))
        keep = {ranking[i]['sample_id'] for i in (0, (len(ranking)-1)//2, len(ranking)-1)}
        selected_order = [x['sample_id'] for x in ranking if x['sample_id'] in keep]
        row_map = {r['sample_id']:r for r in rows}
        rows = [row_map[sid] for sid in selected_order]
    if args.per_sheet <= 0:
        raise ValueError('per-sheet must be positive')
    for start in range(0, len(rows), args.per_sheet):
        group = rows[start:start + args.per_sheet]
        cols = 4 if predictions else 2
        fig, axes = plt.subplots(len(group), cols, figsize=(3.6 * cols, 3.8 * len(group)), squeeze=False)
        for axrow, row in zip(axes, group):
            image = Image.open(inside(row['image_path'])).convert('RGB')
            size = image.size
            gt = mask(row['mask_path'], size)
            label = f"{row['dataset']} / {row['split']} / {row['sample_id']}"
            for ax in axrow:
                ax.axis('off')
                ax.imshow(image)
            axrow[0].set_title(label + '\nImage', fontsize=9)
            def overlay(ax, m, title, color):
                rgba = np.zeros((*m.shape, 4), dtype=float)
                rgba[m] = color
                ax.imshow(rgba, interpolation='nearest')
                ax.set_title(title, fontsize=9)
            overlay(axrow[1], gt, 'GT (>0), green', [0, 1, 0, .42])
            if predictions:
                r = predictions[row['sample_id']]
                for col, key, title in [(2, 'baseline_path', 'Baseline prediction'), (3, 'qwen_path', 'Qwen-text prediction')]:
                    pred = prediction(r[key], size)
                    overlay(axrow[col], pred, title + ', magenta', [1, 0, 1, .42])
                    if ranking is not None:
                        rank = next(item for item in ranking if item['sample_id'] == row['sample_id'])
                        value = rank['baseline_dice'] if key == 'baseline_path' else rank['qwen_dice']
                        axrow[col].set_title(f'{title}, magenta\nDice@224={value:.4f}', fontsize=9)
        fig.tight_layout(pad=1.0)
        fig.savefig(out / f'sheet_{start//args.per_sheet+1:03d}.png', dpi=120, bbox_inches='tight')
        plt.close(fig)
    (out / 'index.json').write_text(json.dumps({'manifest': args.manifest, 'predictions': args.predictions, 'count': len(rows), 'ids': [r['sample_id'] for r in rows], 'per_sheet': args.per_sheet,
        'selection': 'baseline Dice@224 ascending: worst, lower median, best; ties by sample_id' if ranking is not None else 'all', 'ranking': ranking}, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'samples':len(rows), 'sheets':(len(rows)+args.per_sheet-1)//args.per_sheet, 'out': str(out)}))


if __name__ == '__main__':
    main()

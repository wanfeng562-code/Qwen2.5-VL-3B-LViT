"""Render audited final predictions without selecting or updating a model."""
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.experiment.common import atomic_json


def main():
    out = ROOT / 'outputs/phase46/final_test'
    audit = json.loads((out / 'independent_audit.json').read_text())
    assert audit['status'] == 'passed'
    with (ROOT / 'manifests/legacy_subset_v1/monuseg/test.csv').open(encoding='utf-8-sig', newline='') as f:
        rows = {r['sample_id']: r for r in csv.DictReader(f)}
    ranked = sorted(audit['per_image'], key=lambda x: (x['candidate_dice'], x['sample_id']))
    picks = [ranked[i]['sample_id'] for i in (0, 6, 13)]
    delta = sorted(audit['per_image'], key=lambda x: (x['candidate_dice'] - x['constant_dice'], x['sample_id']))
    failure_picks = [delta[0]['sample_id'], delta[-1]['sample_id']]
    arms = ('candidate', 'constant', 'shuffled', 'legacy')
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 18)
    small = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 15)

    def panel(ids, filename, caption):
        canvas = Image.new('RGB', (1920, 70 + 405 * len(ids)), 'white')
        draw = ImageDraw.Draw(canvas)
        draw.text((12, 12), caption, font=font, fill='black')
        draw.text((12, 39), 'Seed 666 predictions; IDs selected by 3-seed means. Green=GT; magenta=prediction. All counts are 224-grid pixels.', font=small, fill='black')
        for index, sid in enumerate(ids):
            row = rows[sid]
            rgb = np.asarray(Image.open(ROOT / row['image_path']).convert('RGB'))
            gt = cv2.resize((cv2.imread(str(ROOT / row['mask_path']), 0) > 0).astype(np.uint8), (224, 224), interpolation=cv2.INTER_NEAREST) > 0
            y = 78 + index * 405
            draw.text((12, y), sid, font=font, fill='black')
            for col, arm in enumerate(('original', 'GT', *arms)):
                pic = rgb.astype(float).copy()
                if arm != 'original':
                    mask = gt if arm == 'GT' else cv2.imread(str(out / f'{arm}_seed666/predictions/{sid}.png'), 0) > 0
                    mask = cv2.resize(mask.astype(np.uint8), (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
                    color = np.array([0, 255, 0] if arm == 'GT' else [255, 0, 255])
                    pic[mask] = pic[mask] * .55 + color * .45
                canvas.paste(Image.fromarray(pic.astype(np.uint8)).resize((308, 308), Image.Resampling.LANCZOS), (6 + 320 * col, y + 55))
                draw.text((8 + 320 * col, y + 30), arm, font=font, fill='black')
                if arm in arms:
                    record = next(x for x in audit['records'][f'{arm}_666'] if x['sample_id'] == sid)
                    label = f"Dice {100*record['dice']:.2f}% | FP {record['fp']} | FN {record['fn']}"
                    draw.text((8 + 320 * col, y + 369), label, font=small, fill='black')
        canvas.save(out / filename)

    panel(picks, 'comparison_final.png', 'Original-split final re-evaluation | lowest / lower median / highest candidate Dice')
    panel(failure_picks, 'delta_cases.png', 'Post-evaluation error analysis | largest loss / gain versus constant control | no model tuning')
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for arm in arms:
        values = [100 * np.mean([x['dice'] for x in audit['records'][f'{arm}_{seed}']]) for seed in (666, 667, 668)]
        axes[0].plot((666, 667, 668), values, '-o', label=arm)
    axes[0].set_xticks((666, 667, 668)); axes[0].set_xlabel('Seed'); axes[0].set_ylabel('Test Dice (%)')
    axes[0].grid(alpha=.2); axes[0].legend()
    values = [100 * (r['candidate_dice'] - r['constant_dice']) for r in delta]
    axes[1].bar(range(14), values, color=['#ba4444' if v < 0 else '#248574' for v in values])
    axes[1].axhline(0, color='black', linewidth=.7)
    axes[1].set_xlabel('All 14 test images, sorted by paired difference')
    axes[1].set_ylabel('Candidate minus constant Dice (pp)')
    fig.suptitle('Final re-evaluation on the original split; no test-based selection')
    fig.tight_layout(); fig.savefig(out / 'final_scores.png', dpi=150); plt.close(fig)
    atomic_json(out / 'visuals.json', dict(status='rendered_pending_human_or_agent_visual_inspection',
        comparison_ids=picks, delta_ids=failure_picks, selection='3-seed candidate Dice rank 0,6,13; extreme differences versus constant',
        displayed_seed=666, files=['comparison_final.png', 'delta_cases.png', 'final_scores.png']))
    print('FINAL_VISUALS_RENDERED')


if __name__ == '__main__':
    main()

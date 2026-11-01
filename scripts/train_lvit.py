#!/usr/bin/env python
"""Train the preserved LViT or evaluate a selected checkpoint once."""
import argparse
import json
import os
from pathlib import Path
import sys

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.experiment.common import project_path
from src.experiment.engine import evaluate_checkpoint, train


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config')
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--max-epochs', type=int)
    parser.add_argument('--device', default=None)
    parser.add_argument('--resume', help='Completed-epoch last.pt from the same output directory')
    parser.add_argument('--evaluate-only', action='store_true')
    parser.add_argument('--checkpoint')
    parser.add_argument('--split', choices=['train', 'val', 'test'], default='test')
    parser.add_argument('--save-predictions', action='store_true')
    args = parser.parse_args()
    if args.evaluate_only:
        if not args.checkpoint:
            parser.error('--evaluate-only requires --checkpoint (select best.pt using validation only)')
        evaluate_checkpoint(args.checkpoint, args.split, args.output_dir, args.device or 'cuda', args.save_predictions)
    else:
        if not args.config:
            parser.error('Training requires --config')
        config = json.loads(project_path(args.config).read_text(encoding='utf-8'))
        for flag, key in ((args.seed, 'seed'), (args.max_epochs, 'max_epochs'), (args.device, 'device')):
            if flag is not None:
                config[key] = flag
        train(config, args.output_dir, args.resume)


if __name__ == '__main__':
    main()

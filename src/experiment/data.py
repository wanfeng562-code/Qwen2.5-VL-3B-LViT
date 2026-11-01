"""Immutable manifest loading and protocol-specific LViT input transforms."""
import csv
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from .common import ROOT, project_path, sha256_file


def read_protocol(path='configs/data_protocol_v1.json'):
    path = project_path(path)
    protocol = json.loads(path.read_text(encoding='utf-8'))
    if protocol['image_input']['lvit_size'] != [224, 224]:
        raise ValueError('The preserved LViT architecture requires 224x224 inputs')
    if protocol['image_input']['lvit_color_order'] != 'BGR':
        raise ValueError('Expected frozen BGR input protocol')
    if not protocol['mask_input']['binarize_before_resize'] or protocol['mask_input']['resize'] != 'nearest':
        raise ValueError('Mask protocol must binarize before nearest resize')
    return protocol


def manifest_rows(dataset, protocol_path='configs/data_protocol_v1.json', verify_assets=False):
    protocol = read_protocol(protocol_path)
    if dataset not in protocol['counts']:
        raise ValueError(f'Unknown dataset: {dataset}')
    result, hashes, seen = {}, {}, set()
    for split in ('train', 'val', 'test'):
        path = project_path(protocol['manifest_root']) / dataset / (split + '.csv')
        with path.open(encoding='utf-8-sig', newline='') as handle:
            rows = list(csv.DictReader(handle))
        if len(rows) != protocol['counts'][dataset][split]:
            raise ValueError(f'Frozen manifest count mismatch: {path}')
        for row in rows:
            if row['dataset'] != dataset or row['split'] != split or row['sample_id'] in seen:
                raise ValueError(f'Invalid/duplicate manifest row: {row}')
            seen.add(row['sample_id'])
            for key in ('image_path', 'mask_path'):
                asset = project_path(row[key]).resolve(strict=True)
                if not asset.is_relative_to(ROOT.resolve()):
                    raise ValueError(f'Manifest path escapes project: {asset}')
                if verify_assets and sha256_file(asset) != row[key.replace('_path', '_sha256')]:
                    raise ValueError(f'Immutable asset hash mismatch: {asset}')
        result[split] = rows
        hashes[split] = sha256_file(path)
    return result, hashes


def load_image(path):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f'Cannot decode image: {path}')
    image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_LINEAR)
    return torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))).float().div_(255.0)


def load_mask(path):
    gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise ValueError(f'Cannot decode mask: {path}')
    # Crucially, do not interpolate a grayscale/instance mask and threshold later.
    binary = (gray > 0).astype(np.uint8)
    binary = cv2.resize(binary, (224, 224), interpolation=cv2.INTER_NEAREST)
    return torch.from_numpy(np.ascontiguousarray(binary[None])).float()


class FeatureStore:
    def __init__(self, path, all_rows, expected_source=None, expected_manifests=None):
        self.path = project_path(path)
        payload = torch.load(self.path, map_location='cpu', weights_only=False)
        if payload.get('schema_version') != 1:
            raise ValueError('Unsupported text feature cache schema')
        self.metadata = payload['metadata']
        self.source = self.metadata['source']
        if expected_source is not None and self.source != expected_source:
            raise ValueError('Text source mismatch between configuration and cache')
        if expected_manifests is not None and self.metadata['manifest_sha256'] != expected_manifests:
            raise ValueError('Text feature cache was built with different manifest bytes')
        ids, tensor = payload['sample_ids'], payload['features']
        if len(ids) != len(set(ids)) or tuple(tensor.shape) != (len(ids), 10, 768):
            raise ValueError('Feature cache IDs/shapes invalid')
        expected = {r['sample_id'] for r in all_rows}
        if set(ids) != expected or not torch.isfinite(tensor).all():
            raise ValueError('Feature cache must contain every manifest sample exactly once and finite values')
        if tensor.dtype != torch.float32:
            raise ValueError('Expected float32 frozen text features')
        record_lookup = {r['sample_id']: r for r in payload['records']}
        if set(record_lookup) != expected or len(payload['records']) != len(expected):
            raise ValueError('Feature provenance records do not match samples')
        for row in all_rows:
            record = record_lookup[row['sample_id']]
            if record['image_sha256'] != row['image_sha256'] or record['split'] != row['split']:
                raise ValueError('Feature provenance does not match image/split')
            if self.source == 'provided_legacy_text' and record['raw_text_sha256'] != row['description_sha256']:
                raise ValueError('Legacy text provenance mismatch')
        self.lookup = dict(zip(ids, tensor.unbind(0)))
        self.sha256 = sha256_file(self.path)


class ManifestDataset(Dataset):
    def __init__(self, rows, feature_store, preload=True):
        self.rows = list(rows)
        self.features = feature_store
        self.loaded = [self._load(row) for row in self.rows] if preload else None

    @staticmethod
    def _load(row):
        return load_image(project_path(row['image_path'])), load_mask(project_path(row['mask_path']))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        image, mask = self.loaded[index] if self.loaded is not None else self._load(row)
        return {'image': image, 'mask': mask, 'text': self.features.lookup[row['sample_id']],
                'sample_id': row['sample_id'], 'image_path': row['image_path']}


def make_loader(dataset, batch_size, seed, epoch=0, shuffle=False, workers=0, device='cpu'):
    generator = torch.Generator().manual_seed(seed + epoch * 100003)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=workers,
                      pin_memory=torch.device(device).type == 'cuda', drop_last=False,
                      generator=generator, persistent_workers=False)

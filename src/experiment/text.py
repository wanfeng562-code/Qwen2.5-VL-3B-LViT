"""Frozen BERT features with explicit legacy-inspired WordPiece aggregation.

This preserves the old bert-embedding *interface*, not numerical equivalence
between GluonNLP and Hugging Face weights/tokenizers. Both experimental arms
must share this encoder and its policy/fingerprint.
"""
import json
import posixpath
import re
import time
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import torch

from .common import project_path, sha256_file, sha256_text, synchronize

ENCODER_VERSION = 'hf_bert_uncased_last_wordpiece_mean25_first10_eofpad_v1'
UPSTREAM_SOURCE = 'https://github.com/imgarylai/bert-embedding/blob/master/bert_embedding/bert.py'


def read_legacy_workbook(path):
    ns = {'s': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
    with zipfile.ZipFile(path) as archive:
        strings = []
        if 'xl/sharedStrings.xml' in archive.namelist():
            strings = [''.join(item.itertext()) for item in ET.fromstring(archive.read('xl/sharedStrings.xml'))]
        workbook = ET.fromstring(archive.read('xl/workbook.xml'))
        first_sheet = workbook.find('s:sheets/s:sheet', ns)
        rel_id = first_sheet.attrib['{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id']
        rels = ET.fromstring(archive.read('xl/_rels/workbook.xml.rels'))
        target = next(item.attrib['Target'] for item in rels if item.attrib['Id'] == rel_id)
        target = target.lstrip('/') if target.startswith('/') else posixpath.normpath('xl/' + target)
        sheet = ET.fromstring(archive.read(target))
        rows = []
        for row in sheet.findall('s:sheetData/s:row', ns):
            cells = {}
            for cell in row.findall('s:c', ns):
                letters = re.match('[A-Z]+', cell.attrib.get('r', ''))
                if not letters:
                    continue
                column = 0
                for letter in letters.group():
                    column = 26 * column + ord(letter) - 64
                value = cell.find('s:v', ns)
                kind = cell.attrib.get('t')
                if kind == 'inlineStr':
                    text = ''.join(cell.find('s:is', ns).itertext())
                elif kind == 's' and value is not None:
                    text = strings[int(value.text)]
                else:
                    text = value.text if value is not None else ''
                cells[column - 1] = text or ''
            if cells:
                rows.append(cells)
    if not rows:
        raise ValueError(f'Empty workbook: {path}')
    headers = rows[0]
    records = [{header: row.get(index, '') for index, header in headers.items()} for row in rows[1:]]
    result = {}
    for record in records:
        if record.get('Image') and record.get('Description'):
            if record['Image'] in result:
                raise ValueError(f'Duplicate workbook key: {record["Image"]}')
            result[record['Image']] = record['Description']
    return result


def load_descriptions(all_rows, source, jsonl_path=None, return_metadata=False):
    if source == 'provided_legacy_text':
        books, result = {}, {}
        for row in all_rows:
            path = row['text_workbook']
            if path not in books:
                books[path] = read_legacy_workbook(project_path(path))
            text = books[path][row['text_key']]
            if sha256_text(text) != row['description_sha256']:
                raise ValueError(f'Frozen legacy description hash mismatch: {row["sample_id"]}')
            result[row['sample_id']] = text
        metadata = {'generation_model_sha256': [], 'generation_config_sha256': []}
        return (result, metadata) if return_metadata else result
    if source not in ('qwen_predicted_text', 'qwen_description') or jsonl_path is None:
        raise ValueError('Qwen source requires its generated descriptions JSONL')
    expected = {row['sample_id']: row for row in all_rows}
    result = {}
    configuration_hashes = set()
    model_hashes = set()
    for line in project_path(jsonl_path).read_text(encoding='utf-8').splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        sample_id = record['sample_id']
        if sample_id in result or sample_id not in expected:
            raise ValueError(f'Unexpected/duplicate Qwen sample: {sample_id}')
        row = expected[sample_id]
        for key in ('dataset', 'split', 'image_path', 'image_sha256'):
            if record[key] != row[key]:
                raise ValueError(f'Qwen provenance mismatch: {sample_id}/{key}')
        if record.get('source', 'qwen_predicted_text') not in ('qwen_predicted_text', 'qwen_description'):
            raise ValueError('Description source must be model-predicted text')
        text = record['plain_text']
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f'Empty Qwen description: {sample_id}')
        configuration_hashes.add(record['config_sha256'])
        if not isinstance(record.get('model_sha256'), str) or not record['model_sha256']:
            raise ValueError(f'Missing Qwen model SHA-256 provenance: {sample_id}')
        model_hashes.add(record['model_sha256'])
        result[sample_id] = text
    if set(result) != set(expected) or len(configuration_hashes) != 1 or len(model_hashes) != 1:
        raise ValueError('Qwen cache must cover every sample exactly once with one generation configuration and model')
    metadata = {'generation_model_sha256': sorted(model_hashes), 'generation_config_sha256': sorted(configuration_hashes)}
    return (result, metadata) if return_metadata else result


def prepare_text(text):
    # Match uploaded utils.read_text filler and first-line behavior in ImageToImage2D.
    if not isinstance(text, str) or not text.strip() or not text.split('\n')[0].strip():
        raise ValueError('Text must be nonempty and its first line must not be blank')
    count = len(text.split())
    padded = text + ' EOF XXX' * max(0, 9 - count)
    return padded.split('\n')[0]


def encoder_fingerprint(model_path):
    path = project_path(model_path).resolve(strict=True)
    names = {'config.json', 'vocab.txt', 'tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json'}
    files = sorted(p for p in path.iterdir() if p.is_file() and (p.name in names or p.suffix in {'.bin', '.safetensors'}))
    if not any(p.suffix in {'.bin', '.safetensors'} for p in files):
        raise ValueError('Local BERT directory has no weights')
    return {'version': ENCODER_VERSION, 'model_path': str(path),
            'model_files_sha256': {p.name: sha256_file(p) for p in files},
            'max_sequence_length_including_special': 25, 'aggregation': 'last_hidden_state_mean_contiguous_hashhash',
            'positions': 10, 'short_feature_padding': 'zero',
            'text_preparation': 'legacy_EOF_XXX_filler_then_first_line',
            'upstream_reference': UPSTREAM_SOURCE,
            'legacy_checkpoint_equivalence': False}


class FrozenBertEncoder:
    def __init__(self, model_path='weights/bert-base-uncased', device='cpu'):
        from transformers import BertModel, BertTokenizer
        self.device = torch.device(device)
        path = project_path(model_path)
        self.tokenizer = BertTokenizer.from_pretrained(path, local_files_only=True, do_lower_case=True)
        self.model = BertModel.from_pretrained(path, local_files_only=True, add_pooling_layer=False)
        self.model.eval().requires_grad_(False).to(self.device)
        if self.model.config.hidden_size != 768:
            raise ValueError('LViT requires 768-dimensional BERT features')
        self.fingerprint = encoder_fingerprint(path)

    @torch.inference_mode()
    def encode(self, texts):
        prepared = [prepare_text(text) for text in texts]
        batch = self.tokenizer(prepared, padding='max_length', truncation=True, max_length=25,
                               return_tensors='pt', return_special_tokens_mask=True)
        special = batch.pop('special_tokens_mask')
        token_ids = batch['input_ids']
        hidden = self.model(**{key: value.to(self.device) for key, value in batch.items()}).last_hidden_state
        output, records = [], []
        for index, sentence in enumerate(prepared):
            tokens = self.tokenizer.convert_ids_to_tokens(token_ids[index].tolist())
            groups, words = [], []
            for position, token in enumerate(tokens):
                if not batch['attention_mask'][index, position] or special[index, position]:
                    continue
                if token.startswith('##') and groups:
                    groups[-1].append(position)
                    words[-1] += token[2:]
                else:
                    groups.append([position])
                    words.append(token)
            features = torch.zeros(10, 768, dtype=torch.float32)
            for target_index, positions in enumerate(groups[:10]):
                features[target_index] = hidden[index, positions].float().mean(0).cpu()
            output.append(features)
            records.append({'prepared_text_sha256': sha256_text(sentence), 'selected_tokens': words[:10],
                            'available_merged_tokens': len(words), 'zero_padded_positions': max(0, 10 - len(words)),
                            'wordpiece_tokens': [token for position, token in enumerate(tokens)
                                                 if batch['attention_mask'][index, position] and not special[index, position]]})
        return torch.stack(output), records

"""Download only BERT inference files, checking ModelScope's SHA-256 manifest."""
import hashlib
import json
import os
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / 'weights/bert-base-uncased'
REPO = 'google-bert/bert-base-uncased'
REVISION = 'fbb17953beb5f9627a4342b49b0f1ce9fa10596a'
WANTED = {'config.json', 'model.safetensors', 'vocab.txt', 'tokenizer.json', 'tokenizer_config.json', 'LICENSE', 'README.md'}


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    session = requests.Session()
    base = f'https://modelscope.cn/api/v1/models/{REPO}/repo'
    response = session.get(base + '/files', params={'Revision': REVISION, 'Recursive': 'true'}, timeout=30)
    response.raise_for_status()
    records = [f for f in response.json()['Data']['Files'] if f['Path'] in WANTED]
    assert {f['Path'] for f in records} == WANTED
    for entry in records:
        path = DEST / entry['Path']
        if path.is_file() and digest(path) == entry['Sha256']:
            print('VERIFIED_EXISTING', path.name, flush=True)
            continue
        temporary = path.with_suffix(path.suffix + '.partial')
        with session.get(base, params={'Revision': REVISION, 'FilePath': entry['Path']}, stream=True, timeout=(30, 120)) as download:
            download.raise_for_status()
            with temporary.open('wb') as f:
                for chunk in download.iter_content(4 * 1024 * 1024):
                    f.write(chunk)
        assert temporary.stat().st_size == entry['Size'], entry['Path']
        assert digest(temporary) == entry['Sha256'], entry['Path']
        os.replace(temporary, path)
        print('DOWNLOADED_VERIFIED', path.name, entry['Size'], flush=True)
    provenance = {'source': 'ModelScope mirror', 'repo': REPO, 'revision': REVISION, 'files': {e['Path']: {'bytes': e['Size'], 'sha256': e['Sha256']} for e in records}, 'download_seconds': time.perf_counter()-start}
    (DEST/'download_provenance.json').write_text(json.dumps(provenance,indent=2))
    from transformers import BertTokenizer, BertModel
    tokenizer=BertTokenizer.from_pretrained(DEST,local_files_only=True)
    model=BertModel.from_pretrained(DEST,local_files_only=True,add_pooling_layer=False).eval().requires_grad_(False)
    import torch
    with torch.inference_mode():
        out=model(**tokenizer('Numerous dark purple nuclei in pink tissue.',return_tensors='pt'))
    assert out.last_hidden_state.shape[-1]==768 and torch.isfinite(out.last_hidden_state).all()
    print('BERT_CPU_LOAD_OK',list(out.last_hidden_state.shape),flush=True)


if __name__ == '__main__':
    main()

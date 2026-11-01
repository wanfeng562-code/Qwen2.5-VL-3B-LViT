"""Package small, audited evidence; exclude weights, datasets and feature caches."""
import hashlib
import json
import shutil
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.experiment.common import atomic_json, sha256_file


def main():
    finalization = json.loads((ROOT/'outputs/phase46/finalization/status.json').read_text())
    assert finalization['status']=='ready_for_primary_agent_review'
    audit = json.loads((ROOT/'outputs/phase46/development_audit.json').read_text())
    assert audit['status']=='passed'
    for name, expected in json.loads((ROOT/'docs/phase46_core_freeze.json').read_text()).items():
        assert sha256_file(ROOT/name)==expected
    evidence = ROOT/'docs/phase46_evidence'
    evidence.mkdir(exist_ok=True)
    sources = {
        'registry.json':'outputs/phase46/registry.json',
        'development_audit.json':'outputs/phase46/development_audit.json',
        'confirmation_summary.json':'outputs/phase46/confirmation/summary.json',
        'seed666_sensitivity.json':'outputs/phase46/confirmation/seed666_sensitivity.json',
        'preflight_checks.json':'outputs/phase46/preflight/checks.json',
        'final_data_check.json':'outputs/phase46/final_data_check.json',
        'benchmark.json':'outputs/phase46/benchmark.json',
        'fewshot_scores.json':'cache/phase46/fewshot_scores.json',
        'fewshot_attributes.jsonl':'cache/phase46/fewshot_attributes.jsonl',
        'extraction_identity.json':'cache/phase46/extraction_identity.json',
        'extraction_receipt.json':'cache/phase46/extraction_receipt.json',
        'candidate_gate.json':'cache/phase46/candidate_gate.json',
        'spatial_resolution_diagnostic.json':'outputs/phase46/spatial_resolution_diagnostic.json',
        'validation_paired_per_image.csv':'outputs/phase46/validation_paired_per_image.csv',
        'validation_review.json':'outputs/phase46/visual_review/review.json',
        'comparison_validation.png':'outputs/phase46/visual_review/comparison_validation.png',
        'training_curves.png':'outputs/phase46/visual_review/training_curves.png',
    }
    if finalization['test_evaluated']:
        final_audit=json.loads((ROOT/'outputs/phase46/final_test/independent_audit.json').read_text())
        assert final_audit['status']=='passed'
        for name in ('summary.json','independent_audit.json','audited_per_image.csv','paired_per_image.csv','visuals.json',
                     'comparison_final.png','delta_cases.png','final_scores.png'):
            sources['final_'+name if name.endswith(('.json','.csv')) else name]='outputs/phase46/final_test/'+name
        sources['test_authorization.json']='outputs/phase46/test_authorization.json'
        sources['test_extraction_receipt.json']='outputs/phase46/test_extraction_receipt.json'
    for entry in json.loads((ROOT/'outputs/phase46/registry.json').read_text())['runs']:
        name=entry['run_id']
        for file in ('run_config.json','result.json','best_validation.json','history.jsonl'):
            sources['runs/'+name+'/'+file]='outputs/phase46/runs/'+name+'/'+file
    copied=[]
    for destination, source in sources.items():
        src=(ROOT/source).resolve(); dst=(evidence/destination).resolve()
        assert src.is_relative_to(ROOT.resolve()) and dst.is_relative_to(evidence.resolve())
        assert src.is_file() and src.suffix in ('.json','.jsonl','.csv','.png') and src.stat().st_size<12_000_000
        dst.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(src,dst)
        assert sha256_file(src)==sha256_file(dst)
        copied.append(dict(source=source,destination=str(dst.relative_to(ROOT)),bytes=dst.stat().st_size,sha256=sha256_file(dst)))
    atomic_json(evidence/'provenance.json',dict(status='copied',artifacts=copied,
        visual_inspection='See docs/phase46_visual_inspection.json when available; rendering alone is not visual acceptance.'))
    paths=[]
    for prefix in ('src/phase46','configs/phase46'):
        paths.extend(p for p in (ROOT/prefix).rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix in ('.py','.json'))
    paths.extend((ROOT/'scripts').glob('phase46_*.py'))
    paths.extend(p for p in (ROOT/'docs').iterdir() if p.is_file() and (p.name.startswith('phase46_') or p.name.startswith('阶段4.6')))
    paths.extend(p for p in evidence.rglob('*') if p.is_file())
    paths=sorted(set(paths))
    archive=ROOT/'outputs/phase46/delivery.zip'
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in paths:
            assert p.suffix in ('.py','.md','.json','.jsonl','.csv','.png') and p.stat().st_size<12_000_000
            z.write(p,p.relative_to(ROOT).as_posix())
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for p in paths:assert hashlib.sha256(z.read(p.relative_to(ROOT).as_posix())).hexdigest()==sha256_file(p)
    receipt=dict(status='ready',files=len(paths),bytes=archive.stat().st_size,sha256=sha256_file(archive),archive=str(archive.relative_to(ROOT)))
    atomic_json(ROOT/'outputs/phase46/delivery_receipt.json',receipt)
    print(json.dumps(receipt))


if __name__=='__main__':
    main()

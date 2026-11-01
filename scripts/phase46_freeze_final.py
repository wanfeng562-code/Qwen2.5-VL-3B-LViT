"""Create final-test authorization only from audited, completed validation gates."""
import sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.experiment.common import atomic_json,sha256_file

def read(p):return json.loads((ROOT/p).read_text())

def main():
    summary=read('outputs/phase46/confirmation/summary.json')
    assert summary['status']=='validation_passed' and summary['validation_gate']['passed']
    assert summary['selection']['candidate']=='V_mid' and len(summary['records'])==12
    audit=read('outputs/phase46/development_audit.json');assert audit['status']=='passed' and audit['experiment_status']=='validation_passed'
    check=read('outputs/phase46/final_data_check.json');assert check['status']=='passed' and check['check_count']==12
    assert check['script_sha256']==sha256_file(ROOT/'scripts/phase46_final_data.py')
    replay=read('outputs/phase46/validation_replay_extraction_receipt.json')
    assert replay['status']=='completed' and replay['count']==1 and all(x['original_cache_bitwise_equal'] for x in replay['records'])
    assert replay['script_sha256']==sha256_file(ROOT/'scripts/phase46_extract_final.py')
    checkpoints={r['run']+'/best.pt':sha256_file(ROOT/r['run']/'best.pt') for r in summary['records']}
    for name,expected in read('docs/phase46_core_freeze.json').items():assert sha256_file(ROOT/name)==expected
    paths=['scripts/phase46_final_data.py','scripts/phase46_extract_final.py','scripts/phase46_evaluate_final.py','scripts/phase46_freeze_final.py','scripts/phase46_final_audit.py','scripts/phase46_final_visuals.py','scripts/protocol_metrics.py']
    value={'validation_gate_passed':True,'scope':'original_split_final_re_evaluation_not_blind_test',
           'validation_summary_sha256':sha256_file(ROOT/'outputs/phase46/confirmation/summary.json'),
           'development_audit_sha256':sha256_file(ROOT/'outputs/phase46/development_audit.json'),
           'development_cache_sha256':sha256_file(ROOT/'cache/phase46/V_mid_development.pt'),
           'contract_sha256':sha256_file(ROOT/'configs/phase46/contract.json'),
           'test_manifest_sha256':sha256_file(ROOT/'manifests/legacy_subset_v1/monuseg/test.csv'),
           'checkpoints':checkpoints,'evaluation_scripts':{name:sha256_file(ROOT/name) for name in paths},
           'success':read('configs/phase46/contract.json')['success'],'records':summary['records'],
           'prohibited':'No selection or tuning based on final test outcomes; no test-derived feature means.'}
    p=ROOT/'outputs/phase46/test_authorization.json'
    if p.exists():assert read(str(p.relative_to(ROOT)))==value
    else:atomic_json(p,value)
    atomic_json(ROOT/'docs/phase46_final_freeze.json',value)
    print('FINAL_TEST_AUTHORIZED',len(checkpoints),'frozen checkpoints')

if __name__=='__main__':main()

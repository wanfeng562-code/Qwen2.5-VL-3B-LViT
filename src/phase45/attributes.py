"""Auditable area attributes. Reference labels never enter generation records."""
import csv,json,re
from pathlib import Path
import cv2,numpy as np
from src.experiment.common import ROOT,atomic_json,sha256_file,sha256_text

ORDER=['upper_left','upper_right','lower_left','lower_right']
COVERAGE=['low','medium','high','unknown']
LOCATION=ORDER+['uniform','unknown']

def rows_for(splits=('train','val'),verify=False):
    rows=[]
    for split in splits:
        if split=='test':
            auth=ROOT/'outputs/phase45/test_authorization.json'
            assert auth.exists() and json.loads(auth.read_text())['validation_gate_passed'], 'Test access is gated'
        p=ROOT/f'manifests/legacy_subset_v1/monuseg/{split}.csv'
        with p.open(encoding='utf-8-sig',newline='') as f:part=list(csv.DictReader(f))
        assert len(part)=={'train':24,'val':6,'test':14}[split]
        if verify:
            for row in part:
                assert sha256_file(ROOT/row['image_path'])==row['image_sha256']
                assert sha256_file(ROOT/row['mask_path'])==row['mask_sha256']
        rows.extend(part)
    return rows

def mask_stats(mask):
    mask=np.asarray(mask)>0;h,w=mask.shape
    assert h>=2 and w>=2
    q=[mask[:h//2,:w//2],mask[:h//2,w//2:],mask[h//2:,:w//2],mask[h//2:,w//2:]]
    return float(mask.mean()),[float(x.mean()) for x in q]

def label(mask,contract):
    area,quadrants=mask_stats(mask);a,b=contract['coverage_quantiles']
    return {'coverage':'low' if area<a else 'medium' if area<b else 'high',
            'location':'uniform' if max(quadrants)-min(quadrants)<.05 else ORDER[int(np.argmax(quadrants))],
            'foreground_fraction':area,'quadrant_fractions':quadrants}

def sentence(value):
    return f"Nuclei coverage {value['coverage']} concentration {value['location'].replace('_',' ')}"

def parse(raw):
    text=raw.strip();actions=[]
    try:json.loads(text);strict_json=True
    except ValueError:strict_json=False
    m=re.fullmatch(r'```(?:json)?\s*\n(.*?)\n```',text,flags=re.S|re.I)
    if m:text=m.group(1).strip();actions=['strip_single_complete_fence']
    try:
        obj=json.loads(text)
        assert isinstance(obj,dict) and set(obj)=={'coverage','location'}
        assert obj['coverage'] in COVERAGE and obj['location'] in LOCATION
        valid=True
    except (ValueError,TypeError,AssertionError):
        obj={'coverage':'unknown','location':'unknown'};valid=False
    return {**obj,'valid':valid,'strict_json':strict_json,'parse_actions':actions,'plain_text':sentence(obj)}

def prepare():
    path=ROOT/'configs/phase45/attribute_contract.json'
    rows=rows_for(verify=True)
    masks={x['sample_id']:cv2.imread(str(ROOT/x['mask_path']),cv2.IMREAD_GRAYSCALE) for x in rows}
    values=[mask_stats(masks[x['sample_id']])[0] for x in rows if x['split']=='train']
    q=np.quantile(values,[1/3,2/3],method='linear').tolist()
    contract={'version':'phase45_area_attributes_v1','coverage_quantiles':q,'quantile_method':'numpy_linear',
              'threshold_source':'24_train_full_images_only','location_order':ORDER,'uniform_range_threshold':.05,
              'coverage_values':COVERAGE,'location_values':LOCATION,'model_path':'weights/Qwen2.5-VL-3B-Instruct',
              'min_pixels':200704,'max_pixels':1003520,'max_new_tokens':64,'do_sample':False,
              'output_contract':'strict fields and enums; strip at most one complete JSON fence; no semantic repairs'}
    contract['prompt']=(f"Examine the cell nuclei in this H&E histology image. Estimate the fraction of image pixels occupied by nuclei, not the number of nuclei. "
        f"Set coverage to low if this fraction is below {q[0]:.8f}, medium if from {q[0]:.8f} to below {q[1]:.8f}, or high otherwise. "
        "Divide the displayed image into four equal quadrants. Set location to upper_left, upper_right, lower_left, or lower_right for the quadrant with the highest fraction of nuclear foreground pixels. "
        "If the largest and smallest quadrant fractions differ by less than 0.05, set location to uniform. If you cannot determine a field, use unknown. "
        "Return only a JSON object with exactly two string fields named coverage and location. Do not include Markdown, explanations, diagnoses, or additional fields.")
    if path.exists():assert json.loads(path.read_text())==contract,'Frozen contract changed'
    else:atomic_json(path,contract)
    refs=[{**x,**label(masks[x['sample_id']],contract)} for x in rows]
    atomic_json(ROOT/'outputs/phase45/attribute_references.json',{'contract_sha256':sha256_file(path),'records':refs})
    atomic_json(ROOT/'configs/phase45/development_images.json',{'records':[{k:x[k] for k in ('sample_id','dataset','split','image_path','image_sha256','width','height')} for x in rows]})
    return contract

def score(cache):
    refs={x['sample_id']:x for x in json.loads((ROOT/'outputs/phase45/attribute_references.json').read_text())['records']}
    records=[json.loads(s) for s in Path(cache).read_text().splitlines() if s.strip()]
    result={}
    for split in ('train','val'):
        part=[x for x in records if x['split']==split]
        assert len(part)=={'train':24,'val':6}[split]
        counts={key:sum(x[key]==refs[x['sample_id']][key] for x in part) for key in ('coverage','location')}
        result[split]={'count':len(part),'correct':counts,'field_accuracy':sum(counts.values())/(2*len(part)),
                       'exact_match':sum(all(x[k]==refs[x['sample_id']][k] for k in counts) for x in part)/len(part),
                       'valid':sum(x['valid'] for x in part),'strict_json':sum(x['strict_json'] for x in part),
                       'unique_sentences':len({x['plain_text'] for x in part})}
    return result

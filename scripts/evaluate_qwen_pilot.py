#!/usr/bin/env python3
"""Strict offline assessment; annotations never enter the Qwen generator."""
import argparse
import csv
import hashlib
import json
import math
import re
import textwrap
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parents[1]


def parse_raw(raw, extent=(1000,1000)):
    errors = []
    try:
        obj = json.loads(raw, parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))
    except (TypeError, ValueError) as e:
        return None, ['invalid_json:' + str(e)]
    if not isinstance(obj, dict) or set(obj) != {'regions'} or not isinstance(obj.get('regions'), list):
        return None, ['expected_object_with_regions_list']
    regions = obj['regions']
    if len(regions) > 3:
        errors.append('too_many_regions')
    seen = set()
    for i, region in enumerate(regions):
        tag = f'region_{i}'
        if not isinstance(region, dict):
            errors.append(tag + ':not_object'); continue
        if set(region) != {'id','description','box'}:
            errors.append(tag + ':invalid_keys')
        ident = region.get('id')
        if not isinstance(ident, int) or isinstance(ident, bool):
            errors.append(tag + ':invalid_id')
        elif str(ident) in seen:
            errors.append(tag + ':duplicate_id')
        else:
            seen.add(str(ident))
        if not isinstance(region.get('description'), str) or not region['description'].strip():
            errors.append(tag + ':invalid_description')
        box = region.get('box')
        if not isinstance(box, list) or len(box) != 4:
            errors.append(tag + ':invalid_box_shape'); continue
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in box):
            errors.append(tag + ':invalid_box_number'); continue
        if any(v < 0 or v > limit for v, limit in zip(box, (extent[0],extent[1],extent[0],extent[1]))):
            errors.append(tag + ':out_of_bounds')
        if not (box[0] < box[2] and box[1] < box[3]):
            errors.append(tag + ':degenerate_box')
    return (regions if not errors else None), errors


def parse_fence_recovery(raw, extent=(1000,1000)):
    """Only remove one complete outer JSON markdown fence; no data repair."""
    if not isinstance(raw, str):
        return None, ['non_string_raw'], False
    match = re.fullmatch(r'\s*```(?:json)?\s*\n(.*?)\n```\s*', raw, flags=re.DOTALL | re.IGNORECASE)
    if not match:
        return None, ['not_single_complete_json_fence'], False
    regions, errors = parse_raw(match.group(1), extent)
    return regions, errors, True


def is_json(raw):
    try:
        json.loads(raw, parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))
        return True
    except (TypeError, ValueError):
        return False


def parse_adapter(raw, extent=(1000,1000)):
    """Only unwrap one entire JSON fence and/or wrap a top-level array."""
    if not isinstance(raw,str): return None,['non_string_raw'],[]
    actions=[]
    match=re.fullmatch(r'\s*```(?:json)?\s*\n(.*?)\n```\s*',raw,flags=re.DOTALL|re.IGNORECASE)
    source=match.group(1) if match else raw
    if match: actions.append('strip_single_full_json_code_fence')
    try:
        obj=json.loads(source,parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))
    except (TypeError,ValueError) as exc:
        return None,['invalid_json:'+str(exc)],actions
    if isinstance(obj,list):
        obj={'regions':obj}; actions.append('wrap_top_level_array_in_regions')
    regions, errors=parse_raw(json.dumps(obj,ensure_ascii=False,allow_nan=False),extent)
    return regions,errors,actions


def normalize_regions(regions, extent):
    if regions is None:
        return None
    return [{**r, 'box':[v * 1000 / limit for v, limit in zip(r['box'], (extent[0],extent[1],extent[0],extent[1]))]} for r in regions]


def equivalent_regions(expected, actual):
    if not isinstance(actual, list) or len(expected) != len(actual): return False
    for a,b in zip(expected,actual):
        if not isinstance(b,dict) or a['id']!=b.get('id') or a['description']!=b.get('description'): return False
        b_box=b.get('box')
        if not isinstance(b_box,list) or len(b_box)!=4: return False
        if any(isinstance(y,bool) or not isinstance(y,(int,float)) or not math.isfinite(y) or abs(x-y)>1e-6 for x,y in zip(a['box'],b_box)): return False
    return True


def box_pixels(b, w, h):
    return (math.floor(b[0]*w/1000), math.floor(b[1]*h/1000),
            math.ceil(b[2]*w/1000), math.ceil(b[3]*h/1000))


def gt_mask(path):
    gray=cv2.imread(str(path),cv2.IMREAD_GRAYSCALE)
    if gray is None: raise ValueError(f'Cannot decode mask: {path}')
    return gray>0


def resize_gt(gt):
    return cv2.resize(gt.astype(np.uint8),(224,224),interpolation=cv2.INTER_NEAREST)>0


def metrics(gt, regions):
    h, w = gt.shape
    union = np.zeros((h,w), dtype=bool)
    box_hits = 0
    box_areas = []
    for region in regions:
        x1,y1,x2,y2 = box_pixels(region['box'], w, h)
        union[y1:y2, x1:x2] = True
        box_hits += bool(gt[y1:y2, x1:x2].any())
        box_areas.append((x2-x1)*(y2-y1)/(w*h))
    tp = int((gt & union).sum()); ng = int(gt.sum()); npred = int(union.sum())
    count, labels = cv2.connectedComponents(gt.astype(np.uint8), connectivity=8)
    touched = np.unique(labels[union & gt]); touched = touched[touched != 0]
    return {'gt_pixels':ng, 'pred_pixels':npred, 'intersection_pixels':tp,
            'coverage':tp/ng if ng else None,
            'precision':tp/npred if npred else None,
            'iou':tp/(ng+npred-tp) if ng+npred-tp else None,
            'gt_components':count-1, 'components_touched':len(touched), 'components_missed':count-1-len(touched),
            'component_touch_rate':len(touched)/(count-1) if count>1 else None,
            'box_count':len(regions), 'boxes_touching_gt':int(box_hits), 'boxes_not_touching_gt':len(regions)-int(box_hits),
            'box_touch_rate':box_hits/len(regions) if regions else None,
            'union_area_fraction':npred/(w*h), 'max_box_area_fraction':max(box_areas, default=0),
            'huge_box_ge_50pct':any(a >= .5 for a in box_areas)}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--selection', required=True)
    p.add_argument('--records', required=True)
    p.add_argument('--generation-config', required=True)
    p.add_argument('--output-contract', help='Fixed envelope adapter contract; reports adapter separately')
    p.add_argument('--calibration-selection', help='Subset used for prompt calibration; report separately from the other cases')
    p.add_argument('--output-dir', required=True)
    p.add_argument('--render', action='store_true', help='render one full-size labeled card per case')
    a=p.parse_args()
    gen_config=json.loads((ROOT/a.generation_config).read_text(encoding='utf-8'))
    config_hash=hashlib.sha256(json.dumps(gen_config,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
    limit=gen_config.get('plain_word_limit')
    if limit is None:
        match=re.search(r'at most\s+(\d+)\s+English words', gen_config.get('plain_prompt',''), re.IGNORECASE)
        if not match: raise ValueError('Cannot determine plain-word limit from generation config')
        limit=int(match.group(1))
    if isinstance(limit,bool) or not isinstance(limit,int) or limit<=0: raise ValueError('Invalid plain_word_limit')
    mode=gen_config.get('coordinate_system')
    if mode not in ('normalized_0_1000_xyxy','absolute_resized_pixels_xyxy'):
        raise ValueError(f'Unsupported coordinate_system: {mode}')
    contract=json.loads((ROOT/a.output_contract).read_text(encoding='utf-8')) if a.output_contract else None
    contract_hash=hashlib.sha256(json.dumps(contract,sort_keys=True,ensure_ascii=False).encode()).hexdigest() if contract else None
    if contract is not None:
        expected={'version':'qwen_output_contract_v1','strip_single_full_json_code_fence':True,
                  'wrap_top_level_array_in_regions':True,'reject_missing_or_extra_fields':True,
                  'reject_any_invalid_region':True,'clip_coordinates':False,'fill_descriptions':False,
                  'max_regions':3,'retain_raw_output':True,'coordinate_transform':'declared_frame_to_normalized_0_1000_without_rounding'}
        if contract != expected: raise ValueError('Unsupported output contract contents')
    selected=json.loads((ROOT/a.selection).read_text(encoding='utf-8'))['records']
    with (ROOT/a.records).open(encoding='utf-8') as f:
        generated=[json.loads(line) for line in f if line.strip()]
    lookup={r['sample_id']:r for r in generated}
    ids={r['sample_id'] for r in selected}
    if len(lookup)!=len(generated) or set(lookup)!=ids:
        raise ValueError(f'ID mismatch or duplicates: selection={len(ids)}, unique generated={len(lookup)}, lines={len(generated)}, missing={sorted(ids-set(lookup))}, extra={sorted(set(lookup)-ids)}')
    manifests={}
    for ds, split in {(r['dataset'],r['split']) for r in selected}:
        if split not in ('train','val'): raise ValueError('Pilot assessment restricted to train/val')
        with (ROOT/f'manifests/legacy_subset_v1/{ds}/{split}.csv').open(newline='',encoding='utf-8-sig') as f:
            manifests.update({r['sample_id']:r for r in csv.DictReader(f)})
    out=ROOT/a.output_dir
    out.mkdir(parents=True,exist_ok=True)
    details=[]
    for sample in selected:
        sid=sample['sample_id']; row=manifests[sid]; rec=lookup[sid]
        if rec.get('config_sha256') != config_hash: raise ValueError(f'{sid}: generation config hash mismatch')
        if contract is not None and rec.get('output_contract_sha256') not in (None,contract_hash):
            raise ValueError(f'{sid}: output contract hash mismatch')
        for k in ('dataset','split','image_path','image_sha256','width','height'):
            if str(sample[k]) != str(rec[k]) or str(sample[k]) != str(row[k]):
                raise ValueError(f'{sid}: manifest or generated mismatch {k}')
        if mode == 'absolute_resized_pixels_xyxy':
            if rec.get('coordinate_system') != mode: raise ValueError(f'{sid}: coordinate system mismatch')
            extent=(rec.get('coordinate_width'),rec.get('coordinate_height'))
            if any(isinstance(x,bool) or not isinstance(x,int) or x<=0 for x in extent):
                raise ValueError(f'{sid}: missing or invalid resized image dimensions')
        else:
            extent=(1000,1000)
        raw_regions,errors=parse_raw(rec.get('structured_raw'),extent)
        raw_recovered,recovery_errors,had_fence=parse_fence_recovery(rec.get('structured_raw'),extent)
        regions=normalize_regions(raw_regions,extent)
        recovered=normalize_regions(raw_recovered,extent)
        adapter_raw,adapter_errors,adapter_actions=parse_adapter(rec.get('structured_raw'),extent) if contract is not None else (None,['not_requested'],[])
        adapter=normalize_regions(adapter_raw,extent)
        raw_text=rec.get('structured_raw')
        fence_match=re.fullmatch(r'\s*```(?:json)?\s*\n(.*?)\n```\s*', raw_text, flags=re.DOTALL | re.IGNORECASE) if isinstance(raw_text,str) else None
        # Diagnostic group consists of strict JSON plus exactly one complete fence.
        diagnostic = regions if not errors else recovered
        diagnostic_valid = not errors or (had_fence and not recovery_errors)
        descriptions = [r['description'].strip() for r in diagnostic] if diagnostic_valid else []
        placeholder = sum(s.casefold() == 'short visible evidence' for s in descriptions)
        words = re.findall(r"\b[\w]+(?:['’\-][\w]+)*\b", str(rec.get('plain_text') or ''))
        detail={'sample_id':sid, 'dataset':sample['dataset'], 'split':sample['split'], 'raw_json_parseable':is_json(raw_text), 'valid':not errors,
                'errors':errors, 'generator_valid':rec.get('structure_valid'), 'plain_text':rec.get('plain_text'),
                'plain_word_count':len(words), 'plain_word_limit':limit, 'plain_over_word_limit':len(words)>limit,
                'coordinate_system':mode, 'coordinate_extent':extent,
                'generator_normalized_regions_match':equivalent_regions(regions,rec.get('regions')) if not errors else None,
                'raw_regions':raw_regions, 'regions_normalized_0_1000':regions, 'original':None,'224':None,
                'fence_recovery':{'had_complete_fence':had_fence, 'json_parseable_after_recovery':bool(fence_match and is_json(fence_match.group(1))), 'valid_after_recovery':had_fence and not recovery_errors,
                  'errors':recovery_errors, 'diagnostic_valid':diagnostic_valid, 'raw_regions':raw_regions if not errors else raw_recovered, 'regions':diagnostic, 'original':None,'224':None,
                  'placeholder_region_count':placeholder, 'duplicate_descriptions':len(set(s.casefold() for s in descriptions)) < len(descriptions)}}
        if contract is not None:
            adapter_desc=[r['description'].strip() for r in adapter] if not adapter_errors else []
            detail['adapter']={'valid':not adapter_errors,'errors':adapter_errors,'actions':adapter_actions,
                'raw_regions':adapter_raw,'regions':adapter,'original':None,'224':None,
                'generator_adapter_valid':rec.get('adapter_valid'),
                'record_has_contract_hash':rec.get('output_contract_sha256') is not None,
                'generator_actions_match':rec.get('adapter_actions') == adapter_actions if rec.get('adapter_actions') is not None else None,
                'generator_raw_regions_match':rec.get('regions_raw') == adapter_raw if not adapter_errors and rec.get('adapter_valid') is True else None,
                'generator_normalized_regions_match':equivalent_regions(adapter,rec.get('regions')) if not adapter_errors and rec.get('adapter_valid') is True else None,
                'placeholder_region_count':sum(s.casefold()=='short visible evidence' for s in adapter_desc),
                'duplicate_descriptions':len(set(s.casefold() for s in adapter_desc)) < len(adapter_desc)}
        im=Image.open(ROOT/row['image_path']).convert('RGB')
        gt=gt_mask(ROOT/row['mask_path'])
        if gt.shape != (im.height, im.width): raise ValueError(f'mask/image size mismatch {sid}')
        if not errors:
            gt224=resize_gt(gt)
            detail['original']=metrics(gt,regions); detail['224']=metrics(gt224,regions)
        if diagnostic_valid:
            gt224=resize_gt(gt)
            detail['fence_recovery']['original']=metrics(gt,diagnostic); detail['fence_recovery']['224']=metrics(gt224,diagnostic)
        if contract is not None and not adapter_errors:
            gt224=resize_gt(gt)
            detail['adapter']['original']=metrics(gt,adapter); detail['adapter']['224']=metrics(gt224,adapter)
        if a.render:
            case_dir=out/'cases'; case_dir.mkdir(exist_ok=True)
            fig, axes=plt.subplots(1,3,figsize=(16,6),gridspec_kw={'width_ratios':[1,1,1.15]})
            for ax in axes[:2]: ax.imshow(im); ax.axis('off')
            rgba=np.zeros((*gt.shape,4),dtype=np.uint8); rgba[gt]=[0,255,0,105]
            axes[1].imshow(rgba,interpolation='nearest')
            status='strict JSON' if not errors else ('FENCE RECOVERY ONLY' if diagnostic_valid else ('ADAPTER: ENVELOPE REPAIRED' if contract is not None and not adapter_errors else 'INVALID; no boxes drawn'))
            displayed=diagnostic if diagnostic_valid else (adapter if contract is not None and not adapter_errors else None)
            axes[0].set_title('Image + boxes (yellow)\n'+status)
            axes[1].set_title('GT mask (green) + predicted boxes')
            if displayed is not None:
                for region in displayed:
                    x1,y1,x2,y2=box_pixels(region['box'],im.width,im.height)
                    for ax in axes[:2]:
                        ax.add_patch(Rectangle((x1,y1),x2-x1,y2-y1,fill=False,edgecolor='yellow',linewidth=1.5))
                        ax.text(x1,y1,str(region['id']),fontsize=9,color='black',bbox={'facecolor':'yellow','pad':1})
            axes[2].axis('off')
            paragraphs=['Plain: '+str(rec.get('plain_text',''))]
            if displayed is None:
                paragraphs.append('INVALID: '+', '.join(errors))
                paragraphs.append('Raw: '+str(rec.get('structured_raw','')))
            elif not displayed:
                paragraphs.append('No regions returned')
            else:
                paragraphs.append(status)
                paragraphs.extend(f"{r['id']}: {r['description']} | normalized box={[round(x,2) for x in r['box']]}" for r in displayed)
                m=detail['fence_recovery']['original'] if diagnostic_valid else detail['adapter']['original']
                paragraphs.append(f"GT coverage={m['coverage']} | precision={m['precision']} | IoU={m['iou']}\nGT component touches={m['components_touched']}/{m['gt_components']} | box area max={m['max_box_area_fraction']:.3f}")
            content='\n\n'.join('\n'.join(textwrap.wrap(s,width=49,break_long_words=True)) for s in paragraphs)
            fig.set_size_inches(16, max(6, 1.1 + .17 * (content.count('\n')+1)))
            axes[2].text(0,1,content,va='top',ha='left',fontsize=8,transform=axes[2].transAxes)
            fig.suptitle(f"{sample['dataset']} / {sample['split']} / {sid}",fontsize=12)
            fig.tight_layout(rect=[0,0,1,.95]); fig.savefig(case_dir/f'{sid}.png',dpi=120,bbox_inches='tight'); plt.close(fig)
        details.append(detail)
    groups=defaultdict(list)
    calibration_ids=set()
    if a.calibration_selection:
        calibration=json.loads((ROOT/a.calibration_selection).read_text(encoding='utf-8'))['records']
        calibration_ids={r['sample_id'] for r in calibration}
        if len(calibration_ids)!=len(calibration) or not calibration_ids.issubset(ids):
            raise ValueError('Calibration IDs must be unique and belong to selection')
    for d in details:
        groups['all'].append(d); groups[d['dataset']].append(d); groups[d['dataset']+'/'+d['split']].append(d)
        if a.calibration_selection:
            groups['calibration' if d['sample_id'] in calibration_ids else 'noncalibration'].append(d)
    summary={}
    for group, ds in groups.items():
        valid=[d for d in ds if d['valid']]
        entry={'total':len(ds),'strict_raw_json_parseable':sum(d['raw_json_parseable'] for d in ds),
               'strict_raw_json_rate':sum(d['raw_json_parseable'] for d in ds)/len(ds),
               'valid':len(valid),'valid_rate':len(valid)/len(ds),
               'generator_normalized_regions_mismatches':sum(d['generator_normalized_regions_match'] is False for d in valid),
               'invalid_errors':dict(Counter(e.split(':')[1] if ':' in e else e for d in ds for e in d['errors']))}
        for scale in ('original','224'):
            ms=[d[scale] for d in valid]
            vals={}
            for key in ('coverage','precision','iou','component_touch_rate','box_touch_rate','union_area_fraction','max_box_area_fraction'):
                v=[m[key] for m in ms if m[key] is not None]
                vals[key]={'mean':float(np.mean(v)) if v else None,'n_defined':len(v)}
            vals['huge_box_images']=sum(m['huge_box_ge_50pct'] for m in ms)
            vals['union_area_ge_50pct_images']=sum(m['union_area_fraction']>=.5 for m in ms)
            vals['component_total']=sum(m['gt_components'] for m in ms)
            vals['component_touched_total']=sum(m['components_touched'] for m in ms)
            vals['component_missed_total']=sum(m['components_missed'] for m in ms)
            vals['boxes_total']=sum(m['box_count'] for m in ms)
            vals['boxes_touching_gt_total']=sum(m['boxes_touching_gt'] for m in ms)
            vals['empty_gt_images']=sum(m['gt_pixels']==0 for m in ms)
            vals['empty_box_images']=sum(m['box_count']==0 for m in ms)
            entry[scale]=vals
        summary[group]=entry
        diagnostics=[d for d in ds if d['fence_recovery']['diagnostic_valid']]
        diag={'total':len(ds), 'strict_raw_valid':len(valid),
              'single_complete_fence_count':sum(d['fence_recovery']['had_complete_fence'] for d in ds),
              'json_parseable_after_fence_removal':sum(d['fence_recovery']['json_parseable_after_recovery'] for d in ds),
              'strict_or_fence_json_parseable_rate':sum(d['raw_json_parseable'] or d['fence_recovery']['json_parseable_after_recovery'] for d in ds)/len(ds),
              'recovered_from_fence_count':sum(d['fence_recovery']['valid_after_recovery'] for d in ds),
              'strict_or_fence_recovered_valid':len(diagnostics),
              'strict_or_fence_recovered_valid_rate':len(diagnostics)/len(ds),
              'plain_word_limit':limit,
              'plain_over_word_limit':sum(d['plain_over_word_limit'] for d in ds),
              'plain_over_word_limit_rate':sum(d['plain_over_word_limit'] for d in ds)/len(ds),
              'placeholder_images':sum(d['fence_recovery']['placeholder_region_count']>0 for d in diagnostics),
              'placeholder_regions':sum(d['fence_recovery']['placeholder_region_count'] for d in diagnostics),
              'diagnostic_region_count':sum(len(d['fence_recovery']['regions']) for d in diagnostics),
              'duplicate_description_images':sum(d['fence_recovery']['duplicate_descriptions'] for d in diagnostics)}
        for scale in ('original','224'):
            ms=[d['fence_recovery'][scale] for d in diagnostics]
            vals={}
            for key in ('coverage','precision','iou','component_touch_rate','box_touch_rate','union_area_fraction','max_box_area_fraction'):
                v=[m[key] for m in ms if m[key] is not None]
                vals[key]={'mean':float(np.mean(v)) if v else None,'n_defined':len(v)}
            vals['huge_box_images']=sum(m['huge_box_ge_50pct'] for m in ms)
            vals['union_area_ge_50pct_images']=sum(m['union_area_fraction']>=.5 for m in ms)
            vals['component_total']=sum(m['gt_components'] for m in ms)
            vals['component_touched_total']=sum(m['components_touched'] for m in ms)
            vals['component_missed_total']=sum(m['components_missed'] for m in ms)
            vals['boxes_total']=sum(m['box_count'] for m in ms)
            vals['boxes_touching_gt_total']=sum(m['boxes_touching_gt'] for m in ms)
            vals['empty_gt_images']=sum(m['gt_pixels']==0 for m in ms)
            vals['empty_box_images']=sum(m['box_count']==0 for m in ms)
            diag[scale]=vals
        entry['fence_recovery_diagnostic']=diag
        if contract is not None:
            adapted=[d for d in ds if d['adapter']['valid']]
            ae={'total':len(ds), 'adapter_valid':len(adapted),'adapter_valid_rate':len(adapted)/len(ds),
                'records_with_contract_hash':sum(d['adapter']['record_has_contract_hash'] for d in ds),
                'actions':dict(Counter(action for d in ds for action in d['adapter']['actions'])),
                'errors':dict(Counter(error for d in ds for error in d['adapter']['errors'])),
                'generator_validity_mismatch':sum(d['adapter']['generator_adapter_valid'] is not None and d['adapter']['generator_adapter_valid'] != d['adapter']['valid'] for d in ds),
                'generator_actions_mismatch':sum(d['adapter']['generator_actions_match'] is False for d in ds),
                'generator_raw_regions_mismatch':sum(d['adapter']['generator_raw_regions_match'] is False for d in adapted),
                'generator_normalized_regions_mismatch':sum(d['adapter']['generator_normalized_regions_match'] is False for d in adapted),
                'placeholder_images':sum(d['adapter']['placeholder_region_count']>0 for d in adapted),
                'placeholder_regions':sum(d['adapter']['placeholder_region_count'] for d in adapted),
                'region_count':sum(len(d['adapter']['regions']) for d in adapted),
                'duplicate_description_images':sum(d['adapter']['duplicate_descriptions'] for d in adapted)}
            for scale in ('original','224'):
                ms=[d['adapter'][scale] for d in adapted];vals={}
                for key in ('coverage','precision','iou','component_touch_rate','box_touch_rate','union_area_fraction','max_box_area_fraction'):
                    v=[m[key] for m in ms if m[key] is not None]
                    vals[key]={'mean':float(np.mean(v)) if v else None,'n_defined':len(v)}
                vals['huge_box_images']=sum(m['huge_box_ge_50pct'] for m in ms)
                vals['union_area_ge_50pct_images']=sum(m['union_area_fraction']>=.5 for m in ms)
                vals['component_total']=sum(m['gt_components'] for m in ms); vals['component_touched_total']=sum(m['components_touched'] for m in ms)
                vals['component_missed_total']=sum(m['components_missed'] for m in ms); vals['boxes_total']=sum(m['box_count'] for m in ms)
                vals['boxes_touching_gt_total']=sum(m['boxes_touching_gt'] for m in ms)
                vals['empty_gt_images']=sum(m['gt_pixels']==0 for m in ms); vals['empty_box_images']=sum(m['box_count']==0 for m in ms)
                ae[scale]=vals
            entry['adapter_diagnostic']=ae
    (out/'details.json').write_text(json.dumps(details,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (out/'summary.json').write_text(json.dumps({'generation_config':a.generation_config,'output_contract':a.output_contract,'output_contract_sha256':contract_hash,'config_sha256':config_hash,'plain_word_limit':limit,'coordinate_system':mode,'groups':summary,'notes':['conditional metrics only on valid outputs; undefined denominators excluded and n shown','adapter is a separate group allowing only one full JSON fence removal and top-level array wrapping; strict raw validity unchanged','absolute resized coordinates independently normalized using the recorded input extent; no clipping or GT-derived adjustments','fence recovery is separate diagnostic group: strict valid plus valid after removing one complete outer json fence; no coordinate correction','placeholder exactly short visible evidence, case insensitive; duplicate descriptions case insensitive','8-connected GT regions are not instance labels','purposeful train/val selection, not population estimates','huge: any single box >=50% of image','half-open boxes floor min and ceil max']},ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False))


if __name__=='__main__': main()

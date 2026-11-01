"""Validation-image deployment timing, with both full Qwen and LViT resident."""
import os
os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false', CUBLAS_WORKSPACE_CONFIG=':4096:8')
import hashlib
import json
import statistics
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.experiment.common import atomic_json, sha256_file, sha256_text, seed_everything
from src.experiment.data import load_image
from src.phase46.engine import setup_data, run_identity, load_checkpoint
from src.phase46.model import build_model


def main():
    torch.set_num_threads(8)
    # This does not open any test manifest or GT mask.
    directory = ROOT / 'outputs/phase46/runs/V_mid_seed666'
    cfg = json.loads((directory / 'run_config.json').read_text())['config']
    contract = json.loads((ROOT / 'configs/phase46/contract.json').read_text())
    seed_everything(666, cfg['deterministic'])
    rows, hashes, store = setup_data(cfg)
    row = rows['val'][0]
    assert sha256_file(ROOT / row['image_path']) == row['image_sha256']
    files = {p.name: sha256_file(p) for p in sorted((ROOT / contract['model_path']).iterdir())
             if p.is_file() and p.suffix in ('.json', '.safetensors', '.txt')}
    identity = json.loads((ROOT / 'cache/phase46/extraction_identity.json').read_text())
    assert sha256_text(json.dumps(files, sort_keys=True)) == identity['model_sha256']
    startup = time.perf_counter()
    processor = AutoProcessor.from_pretrained(ROOT / contract['model_path'], local_files_only=True,
        use_fast=False, min_pixels=contract['min_pixels'], max_pixels=contract['max_pixels'])
    qwen = Qwen2_5_VLForConditionalGeneration.from_pretrained(ROOT / contract['model_path'], local_files_only=True,
        torch_dtype=torch.bfloat16, attn_implementation='sdpa', device_map={'': 0}).eval().requires_grad_(False)
    segmentation = build_model(cfg).cuda().eval()
    load_checkpoint(str((directory / 'best.pt').relative_to(ROOT)), segmentation,
                    expected_identity=run_identity(cfg, hashes, store))
    text_mean = store.lookup[row['sample_id']]['text'][None].cuda()
    capture = {}
    hook = qwen.visual.blocks[15].register_forward_hook(lambda m, a, out: capture.update(mid=out.detach()))
    torch.cuda.synchronize()
    load_seconds = time.perf_counter() - startup
    original = torch.load(ROOT / 'cache/phase46/visual_samples' / (row['sample_id'] + '.pt'),
                          map_location='cpu', weights_only=False)

    @torch.inference_mode()
    def once():
        start = time.perf_counter()
        messages = [{'role': 'user', 'content': [
            {'type': 'image', 'image': Image.open(ROOT / row['image_path']).convert('RGB'),
             'min_pixels': contract['min_pixels'], 'max_pixels': contract['max_pixels']},
            {'type': 'text', 'text': 'Describe the image.'}]}]
        images, videos = process_vision_info(messages)
        batch = processor(text=[processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)],
                          images=images, videos=videos, return_tensors='pt').to('cuda')
        image = load_image(ROOT / row['image_path'])[None].cuda()
        torch.cuda.synchronize()
        prepared = time.perf_counter()
        grid = batch.image_grid_thw
        h, w = map(int, grid[0, 1:].tolist())
        qwen.visual(batch.pixel_values.to(qwen.visual.dtype), grid_thw=grid)
        window, _ = qwen.visual.get_window_index(grid)
        mid = capture.pop('mid').reshape(-1, 4, 1280)[torch.argsort(window)]
        mid = mid.reshape(h // 2, w // 2, 2, 2, 1280).permute(4, 0, 2, 1, 3).reshape(1, 1280, h, w).float()
        feature = F.adaptive_avg_pool2d(mid, (28, 28)).half().float()
        torch.cuda.synchronize()
        extracted = time.perf_counter()
        probability = segmentation(image, {'vision': feature, 'text': text_mean})
        binary = (probability >= .5).cpu()
        torch.cuda.synchronize()
        end = time.perf_counter()
        return dict(preprocess_seconds=prepared - start, vision_seconds=extracted - prepared,
                    segmentation_and_binary_copy_seconds=end - extracted, total_seconds=end - start), feature, probability, binary

    first, feature, probability, binary = once()
    assert torch.equal(feature[0].half().cpu(), original['features']['mid'])
    with torch.inference_mode():
        image = load_image(ROOT / row['image_path'])[None].cuda()
        cached = segmentation(image, {k: v[None].cuda() for k, v in store.lookup[row['sample_id']].items()})
    assert torch.equal(probability, cached)
    del feature, probability, binary, cached, image
    once()  # Second unmeasured warmup; no optimizer or gradient work.
    torch.cuda.reset_peak_memory_stats()
    samples = []
    for _ in range(10):
        timing, feature, probability, binary = once()
        samples.append(timing)
        del feature, probability, binary
    result = dict(status='completed', split='val', sample_id=row['sample_id'], batch_size=1,
                  warmup_runs=2, measured_runs=10, model_load_seconds=load_seconds, first_run=first,
                  timings={key: dict(mean=statistics.mean(s[key] for s in samples), median=statistics.median(s[key] for s in samples),
                                    minimum=min(s[key] for s in samples), maximum=max(s[key] for s in samples)) for key in samples[0]},
                  measured_samples=samples, peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                  peak_reserved_bytes=torch.cuda.max_memory_reserved(), validation_feature_bitwise_equal=True,
                  cached_vs_live_probability_bitwise_equal=True, test_accessed=False,
                  checkpoint_sha256=sha256_file(directory / 'best.pt'), script_sha256=sha256_file(Path(__file__)),
                  scope='Full Qwen weights plus LViT resident; only Qwen visual tower executes. Constant training-mean text is preloaded. Includes image preprocessing, visual extraction, segmentation and binary CPU transfer; excludes file output, model loading and initial hashes from warm timings.')
    hook.remove()
    atomic_json(ROOT / 'outputs/phase46/benchmark.json', result)
    print(json.dumps({k: v for k, v in result.items() if k != 'measured_samples'}))


if __name__ == '__main__':
    main()

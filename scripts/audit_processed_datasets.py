"""Read-only audit of processed medical datasets against the uploaded LViT loader."""

import argparse
import collections
import concurrent.futures
import csv
import hashlib
import json
import os
import posixpath
import re
import statistics
import time
import traceback
import xml.etree.ElementTree as ET
import zipfile
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

SPLITS = ["Train_Folder", "Val_Folder", "Test_Folder"]
EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
NS = {"s": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}


def emit(message):
    print(f"[{datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds')}] {message}", flush=True)


def digest_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def digest_array(array):
    digest = hashlib.sha256()
    digest.update(str(array.shape).encode())
    digest.update(str(array.dtype).encode())
    digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def read_xlsx(path):
    """Read the first worksheet without installing pandas/openpyxl."""
    with zipfile.ZipFile(path) as archive:
        shared = []
        if "xl/sharedStrings.xml" in archive.namelist():
            tree = ET.fromstring(archive.read("xl/sharedStrings.xml"))
            shared = ["".join(item.itertext()) for item in tree]
        workbook = ET.fromstring(archive.read("xl/workbook.xml"))
        sheet = workbook.find("s:sheets/s:sheet", NS)
        relation_id = sheet.attrib["{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"]
        relationships = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
        target = next(item.attrib["Target"] for item in relationships if item.attrib["Id"] == relation_id)
        target = target.lstrip("/") if target.startswith("/") else posixpath.normpath("xl/" + target)
        tree = ET.fromstring(archive.read(target))
        rows = []
        for row in tree.findall("s:sheetData/s:row", NS):
            cells = {}
            for cell in row.findall("s:c", NS):
                address = cell.attrib.get("r", "")
                letters = re.match(r"[A-Z]+", address)
                if not letters:
                    continue
                column = 0
                for letter in letters.group():
                    column = column * 26 + ord(letter) - 64
                kind = cell.attrib.get("t")
                value = cell.find("s:v", NS)
                if kind == "inlineStr":
                    text = "".join(cell.find("s:is", NS).itertext())
                elif kind == "s" and value is not None:
                    text = shared[int(value.text)]
                else:
                    text = value.text if value is not None else ""
                cells[column - 1] = {"value": text or "", "kind": kind}
            if cells:
                rows.append((int(row.attrib.get("r", len(rows) + 1)), cells))
    if not rows:
        return {"headers": [], "records": [], "error": "empty_workbook"}
    headers = [rows[0][1].get(i, {}).get("value", "") for i in range(max(rows[0][1]) + 1)]
    records = []
    for row_number, cells in rows[1:]:
        record = {headers[i]: cells.get(i, {}).get("value", "") for i in range(len(headers)) if headers[i]}
        record["_row_number"] = row_number
        records.append(record)
    return {"headers": headers, "records": records}


def infer_case(task, name):
    if task == "monuseg":
        match = re.match(r"(TCGA-[^-]+-[^-]+)", name)
        return (match.group(1), "patient_identifier") if match else (None, None)
    if task == "cov19":
        match = re.match(r"(sub-S\d+)", name)
        return (match.group(1), "patient_identifier") if match else (None, None)
    match = re.match(r"(.*)_\d+\.[^.]+$", name)
    return (match.group(1), "filename_case_or_series_identifier") if match else (None, None)


def mask_name(image_name, masks):
    prefixed = "mask_" + image_name
    return prefixed if prefixed in masks else image_name[:-3] + "png"


def inspect_pair(job):
    task, split, image_path, mask_path, reference_candidates = job
    row = {"task": task, "split": split, "image": image_path.name, "mask": mask_path.name,
           "image_path": str(image_path), "mask_path": str(mask_path), "errors": [], "warnings": []}
    try:
        with Image.open(image_path) as image:
            image.load()
            row.update({"image_mode": image.mode, "width": image.width, "height": image.height,
                        "image_frames": getattr(image, "n_frames", 1)})
            rgb = np.array(image.convert("RGB"))
            row["image_pixel_hash"] = digest_array(rgb)
            row["image_file_hash"] = digest_file(image_path)
            row["image_std"] = round(float(rgb.std()), 5)
        if not mask_path.is_file():
            row["errors"].append("missing_mask")
            return row
        with Image.open(mask_path) as mask:
            mask.load()
            raw = np.array(mask)
            gray = np.array(mask.convert("L"))
            row.update({"mask_mode": mask.mode, "mask_dtype": str(raw.dtype), "mask_width": mask.width,
                        "mask_height": mask.height, "mask_raw_values": np.unique(raw).tolist()[:300]})
        unique, counts = np.unique(gray, return_counts=True)
        row["mask_values"] = unique.tolist()
        row["mask_value_counts"] = dict(zip(map(str, unique.tolist()), counts.tolist()))
        binary = gray > 0
        row["foreground_ratio"] = round(float(binary.mean()), 8)
        row["mask_binary_hash"] = digest_array(binary)
        row["mask_file_hash"] = digest_file(mask_path)
        row["nonbinary_pixel_count"] = int(np.count_nonzero((gray != 0) & (gray != 1) & (gray != 255)))
        if (row["width"], row["height"]) != (row["mask_width"], row["mask_height"]):
            row["errors"].append("image_mask_size_mismatch")
        if len(unique) > 2 or not set(unique.tolist()).issubset({0, 1, 255}):
            row["warnings"].append("mask_not_strict_binary_encoding")
        if not binary.any():
            row["warnings"].append("empty_mask")
        if binary.all():
            row["warnings"].append("full_foreground_mask")
        if row["image_std"] == 0:
            row["warnings"].append("constant_image")
        if raw.dtype != np.uint8:
            row["warnings"].append("mask_not_uint8")
        if len(unique) == 2 and 0 not in unique:
            row["warnings"].append("all_nonzero_values_become_foreground_in_lvit")
        case_id, kind = infer_case(task, image_path.name)
        row["case_id"], row["case_key_kind"] = case_id, kind
        row["reference_candidates"] = [candidate["split"] for candidate in reference_candidates]
        if reference_candidates:
            preferred = next((candidate for candidate in reference_candidates if candidate["split"] == split), reference_candidates[0])
            reference_image, reference_mask = preferred["image"], preferred["mask"]
            row["reference_split_used"] = preferred["split"]
            row["reference_image_file_identical"] = row["image_file_hash"] == digest_file(reference_image)
            if row["reference_image_file_identical"]:
                row["reference_image_pixels_identical"] = True
            else:
                with Image.open(reference_image) as original:
                    original_rgb = np.array(original.convert("RGB"))
                row["reference_image_pixels_identical"] = original_rgb.shape == rgb.shape and bool(np.array_equal(original_rgb, rgb))
            if reference_mask.is_file():
                row["reference_mask_file_identical"] = row["mask_file_hash"] == digest_file(reference_mask)
                with Image.open(reference_mask) as original:
                    original_mask = np.array(original.convert("L")) > 0
                row["reference_mask_same_shape"] = original_mask.shape == binary.shape
                if original_mask.shape == binary.shape:
                    changed = int(np.count_nonzero(original_mask != binary))
                    union = int(np.count_nonzero(original_mask | binary))
                    intersection = int(np.count_nonzero(original_mask & binary))
                    row["reference_binary_mask_changed_pixels"] = changed
                    row["reference_binary_mask_iou"] = round(intersection / union, 8) if union else 1.0
                    if changed:
                        row["warnings"].append("mask_differs_from_uploaded_reference")
                else:
                    row["warnings"].append("mask_shape_differs_from_uploaded_reference")
            else:
                row["warnings"].append("reference_mask_missing")
        else:
            row["warnings"].append("not_found_in_uploaded_reference")
    except Exception as error:
        row["errors"].append("decode_or_comparison_error")
        row["error_detail"] = type(error).__name__ + ": " + str(error)
    return row


def make_contact_sheet(task, rows, destination):
    width, panel, row_height = 840, 256, 302
    canvas = Image.new("RGB", (width, 64 + row_height * 3), "white")
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 13)
        title_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18)
    except OSError:
        font = title_font = ImageFont.load_default()
    draw.text((12, 8), f"{task}: representative image / mask / overlay", fill="black", font=title_font)
    chosen = []
    for index, split in enumerate(SPLITS):
        candidates = sorted([r for r in rows if r["split"] == split and not r["errors"]], key=lambda r: r["foreground_ratio"])
        if not candidates:
            continue
        row = candidates[len(candidates) // 2]
        chosen.append({"split": split, "image": row["image"], "mask": row["mask"], "foreground_ratio": row["foreground_ratio"]})
        with Image.open(row["image_path"]) as image:
            rgb = np.array(image.convert("RGB"))
        with Image.open(row["mask_path"]) as mask:
            binary = np.array(mask.convert("L")) > 0
        overlay = rgb.copy()
        if rgb.shape[:2] == binary.shape:
            overlay[binary] = (0.65 * overlay[binary] + 0.35 * np.array([0, 255, 0])).astype(np.uint8)
        native_mask = Image.fromarray((binary * 255).astype(np.uint8)).convert("RGB")
        top = 52 + index * row_height
        draw.text((12, top), f"{split} | foreground={row['foreground_ratio']:.3f}", fill="black", font=font)
        draw.text((12, top + 19), row["image"][:100], fill="black", font=font)
        for column, view in enumerate([Image.fromarray(rgb), native_mask, Image.fromarray(overlay)]):
            thumb = ImageOps.contain(view, (panel, panel), Image.Resampling.NEAREST if column == 1 else Image.Resampling.BILINEAR)
            canvas.paste(thumb, (12 + column * 278 + (panel - thumb.width) // 2, top + 42 + (panel - thumb.height) // 2))
    canvas.save(destination)
    return chosen


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    processed = Path("/root/autodl-tmp/datasets")
    reference = Path("/root/autodl-tmp/对比方法/datasets")
    code_root = Path("/root/autodl-tmp/对比方法/LViT-main")
    report = {
        "started_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "processed_root": str(processed), "reference_root": str(reference), "lvit_root": str(code_root),
        "scope": "All processed image/mask files and first-sheet Excel text mappings; byte and pixel comparison with corresponding uploaded reference files; static LViT interface audit.",
        "limitations": [
            "Consistency with uploaded reference does not independently certify clinical annotation correctness.",
            "Visual review covers representative samples, not every image.",
            "BERT embedding and the original DataLoader are not executed; required legacy dependencies are absent.",
            "Filename-derived patient/series overlap is reported only where identifiers can be inferred.",
        ],
        "splits": [], "tasks": {}, "code_findings": [],
    }
    rows = []
    text_lookup = {}
    for task in sorted(p.name for p in processed.iterdir() if p.is_dir()):
        reference_index = collections.defaultdict(list)
        reference_inventory = {}
        for split in SPLITS:
            folder = reference / task / split
            images = {p.name: p for p in (folder / "img").iterdir() if p.is_file() and p.suffix.lower() in EXTENSIONS} if (folder / "img").is_dir() else {}
            masks = {p.name: p for p in (folder / "labelcol").iterdir() if p.is_file()} if (folder / "labelcol").is_dir() else {}
            reference_inventory[split] = set(images)
            for name, path in images.items():
                reference_index[name].append({"split": split, "image": path, "mask": folder / "labelcol" / mask_name(name, masks)})
        task_rows = []
        for split in SPLITS:
            folder = processed / task / split
            image_folder, mask_folder = folder / "img", folder / "labelcol"
            images = sorted(p for p in image_folder.iterdir() if p.is_file() and p.suffix.lower() in EXTENSIONS) if image_folder.is_dir() else []
            all_image_entries = sorted(p.name for p in image_folder.iterdir()) if image_folder.is_dir() else []
            masks = {p.name: p for p in mask_folder.iterdir() if p.is_file()} if mask_folder.is_dir() else {}
            expected_masks = {mask_name(path.name, masks) for path in images}
            book_path = folder / split.replace("_Folder", "_text.xlsx")
            book = read_xlsx(book_path) if book_path.is_file() else {"headers": [], "records": [], "error": "missing_workbook"}
            lookup, key_rows = {}, collections.defaultdict(list)
            empty_descriptions, whitespace_keys, invalid_types = [], [], []
            for record in book["records"]:
                key = record.get("Image", "")
                description = record.get("Description", "")
                key_rows[key].append(record["_row_number"])
                if not key:
                    invalid_types.append(record["_row_number"])
                if not str(description).strip():
                    empty_descriptions.append(key)
                if key != key.strip():
                    whitespace_keys.append(key)
                lookup[key] = description
            text_lookup[(task, split)] = lookup
            duplicate_keys = {key: numbers for key, numbers in key_rows.items() if len(numbers) > 1}
            summary = {
                "task": task, "split": split, "image_count": len(images), "mask_count": len(masks),
                "required_folders_exist": image_folder.is_dir() and mask_folder.is_dir(),
                "unsupported_image_entries": [name for name in all_image_entries if Path(name).suffix.lower() not in EXTENSIONS],
                "orphan_masks": sorted(set(masks) - expected_masks),
                "text_workbook": str(book_path), "text_headers": book["headers"],
                "text_row_count": len(book["records"]), "text_unique_keys": len(lookup),
                "missing_text_keys": sorted(expected_masks - set(lookup)),
                "extra_text_keys_count": len(set(lookup) - expected_masks),
                "extra_text_key_examples": sorted(set(lookup) - expected_masks)[:10],
                "duplicate_text_keys": duplicate_keys,
                "empty_descriptions": empty_descriptions,
                "whitespace_text_keys": whitespace_keys,
                "blank_key_rows": invalid_types,
                "workbook_error": book.get("error"),
                "temporary_workbooks": [p.name for p in folder.glob("~$*.xlsx")],
                "description_examples": [{"Image": key, "Description": lookup[key]} for key in sorted(expected_masks & set(lookup))[:2]],
                "reference_count_same_split": len(reference_inventory.get(split, set())),
            }
            jobs = [(task, split, path, mask_folder / mask_name(path.name, masks), reference_index.get(path.name, [])) for path in images]
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
                batch_rows = list(pool.map(inspect_pair, jobs))
            for row in batch_rows:
                row["text_found_by_lvit_mask_key"] = row["mask"] in lookup
                row["text_words"] = len(lookup.get(row["mask"], "").split())
                if not row["text_found_by_lvit_mask_key"]:
                    row["errors"].append("missing_lvit_text_key")
                elif not lookup[row["mask"]].strip():
                    row["errors"].append("empty_description")
            summary["image_shapes"] = dict(collections.Counter(f"{r.get('width')}x{r.get('height')}:{r.get('image_mode')}" for r in batch_rows))
            summary["mask_shapes"] = dict(collections.Counter(f"{r.get('mask_width')}x{r.get('mask_height')}:{r.get('mask_mode')}" for r in batch_rows))
            summary["mask_encodings"] = dict(collections.Counter(str(r.get("mask_values")) for r in batch_rows))
            summary["error_counts"] = dict(collections.Counter(code for r in batch_rows for code in r["errors"]))
            summary["warning_counts"] = dict(collections.Counter(code for r in batch_rows for code in r["warnings"]))
            summary["foreground_ratio_min_median_max"] = [min(r["foreground_ratio"] for r in batch_rows if "foreground_ratio" in r), statistics.median(r["foreground_ratio"] for r in batch_rows if "foreground_ratio" in r), max(r["foreground_ratio"] for r in batch_rows if "foreground_ratio" in r)] if any("foreground_ratio" in r for r in batch_rows) else None
            summary["reference_image_pixels_identical_count"] = sum(r.get("reference_image_pixels_identical", False) for r in batch_rows)
            summary["reference_binary_masks_identical_count"] = sum(r.get("reference_binary_mask_changed_pixels") == 0 for r in batch_rows)
            report["splits"].append(summary)
            task_rows.extend(batch_rows)
            rows.extend(batch_rows)
            emit(json.dumps({"completed": task + "/" + split, "images": len(images), "errors": summary["error_counts"], "warnings": summary["warning_counts"], "missing_text": len(summary["missing_text_keys"])}, ensure_ascii=True))
        split_overlaps = []
        for i, first in enumerate(SPLITS):
            for second in SPLITS[i + 1:]:
                one = [r for r in task_rows if r["split"] == first]
                two = [r for r in task_rows if r["split"] == second]
                names = sorted(set(r["image"] for r in one) & set(r["image"] for r in two))
                hashes_one = collections.defaultdict(list)
                hashes_two = collections.defaultdict(list)
                for r in one:
                    if r.get("image_pixel_hash"):
                        hashes_one[r["image_pixel_hash"]].append(r["image"])
                for r in two:
                    if r.get("image_pixel_hash"):
                        hashes_two[r["image_pixel_hash"]].append(r["image"])
                duplicate_pixels = [{"first": hashes_one[key], "second": hashes_two[key]} for key in sorted(set(hashes_one) & set(hashes_two))]
                cases_one = {r["case_id"] for r in one if r.get("case_id")}
                cases_two = {r["case_id"] for r in two if r.get("case_id")}
                overlapping_cases = sorted(cases_one & cases_two)
                split_overlaps.append({
                    "first": first, "second": second, "same_names": names,
                    "identical_image_pixels": duplicate_pixels,
                    "case_or_series_overlap": overlapping_cases,
                    "first_identifiable_cases": len(cases_one), "second_identifiable_cases": len(cases_two),
                    "case_examples": [
                        {"case": case, "first_images": [r["image"] for r in one if r.get("case_id") == case][:5],
                         "second_images": [r["image"] for r in two if r.get("case_id") == case][:5],
                         "first_count": sum(r.get("case_id") == case for r in one),
                         "second_count": sum(r.get("case_id") == case for r in two)}
                        for case in overlapping_cases
                    ],
                    "reference_name_overlap_count": len(reference_inventory.get(first, set()) & reference_inventory.get(second, set())),
                })
        figure = output / (task + "_contact_sheet.png")
        representatives = make_contact_sheet(task, task_rows, figure)
        report["tasks"][task] = {
            "total_samples": len(task_rows), "split_overlaps": split_overlaps,
            "case_ids_by_split": {split: sorted({r["case_id"] for r in task_rows if r["split"] == split and r.get("case_id")}) for split in SPLITS},
            "cases_not_identifiable_count": sum(not r.get("case_id") for r in task_rows),
            "contact_sheet": str(figure), "representatives": representatives,
        }

    config_source = (code_root / "Config.py").read_text()
    loader_source = (code_root / "Load_Dataset.py").read_text()
    test_source = (code_root / "test_model.py").read_text()
    report["code_findings"] = [
        {"code": "configured_dataset_alias_and_path", "detail": "Config.py defaults to monuseg_2 and ./datasets/<task>/; processed data are at /root/autodl-tmp/datasets/{monuseg,cov19,mosmeddata}. Explicit dataset paths and aliases need alignment."},
        {"code": "uploaded_loader_accepts_mask_prefix", "confirmed": 'mask_with_prefix = "mask_" + image_filename' in loader_source},
        {"code": "mask_resize_default_bilinear_before_threshold", "confirmed": "mask = cv2.resize(mask, (self.image_size, self.image_size))" in loader_source, "detail": "No interpolation argument is supplied; nonzero binarization follows resize. Prefer nearest-neighbor for categorical masks."},
        {"code": "test_metric_denominator_hardcoded", "detail": "test_model.py contains fixed test_num values (Covid19 2113, MoNuSeg 14, MosMed 273); processed cov19 test count is 200, so the metric denominator must use actual dataset length."},
        {"code": "test_dataset_aliases_missing", "detail": "Training accepts monuseg and cov19, but test_model.py dataset_map does not include those aliases."},
        {"code": "bert_embedding_dependency_unavailable", "detail": "Original BERT encoding and complete DataLoader runtime were not executed; this audit validates files, mappings and static loader behavior."},
    ]
    report["totals"] = {
        "samples": len(rows), "samples_with_errors": sum(bool(r["errors"]) for r in rows),
        "samples_with_warnings": sum(bool(r["warnings"]) for r in rows),
        "image_pixels_identical_to_reference": sum(r.get("reference_image_pixels_identical", False) for r in rows),
        "binary_masks_identical_to_reference": sum(r.get("reference_binary_mask_changed_pixels") == 0 for r in rows),
    }
    report["finished_at"] = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()
    (output / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "samples.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    fields = sorted({key for row in rows for key in row})
    with (output / "samples.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict)) else value for key, value in row.items()})
    emit("FINISHED: " + json.dumps(report["totals"], ensure_ascii=True))


if __name__ == "__main__":
    main()

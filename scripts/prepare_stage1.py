"""Copy the user-confirmed historical dataset and freeze a reproducible protocol."""

import argparse
import collections
import csv
import hashlib
import importlib.metadata
import json
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
from PIL import Image
from audit_processed_datasets import infer_case, mask_name, read_xlsx

SPLITS = {"Train_Folder": "train", "Val_Folder": "val", "Test_Folder": "test"}
COUNTS = {"monuseg": {"train": 24, "val": 6, "test": 14},
          "cov19": {"train": 160, "val": 40, "test": 200},
          "mosmeddata": {"train": 160, "val": 40, "test": 273}}


def log(event, **fields):
    print(json.dumps({"event": event, **fields}, ensure_ascii=True), flush=True)


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def copy_verified(source, destination, exclude_caches=False):
    destination.mkdir(parents=True, exist_ok=True)
    copied, total_bytes = [], 0
    for original in sorted(source.rglob("*")):
        relative = original.relative_to(source)
        if exclude_caches and (set(relative.parts) & {".git", "__pycache__"} or original.suffix == ".pyc"):
            continue
        if original.is_symlink():
            raise RuntimeError("Source symlinks are not accepted: " + str(original))
        if not original.is_file():
            continue
        target = destination / relative
        if not target.resolve().is_relative_to(destination.resolve()):
            raise RuntimeError("Destination escapes project folder")
        target.parent.mkdir(parents=True, exist_ok=True)
        before = sha(original)
        if target.exists():
            if sha(target) != before:
                raise RuntimeError("Existing destination differs; not overwritten: " + str(target))
        else:
            shutil.copy2(original, target)
        if sha(target) != before or sha(original) != before:
            raise RuntimeError("Copy integrity/source stability check failed")
        if target.stat().st_ino == original.stat().st_ino and target.stat().st_dev == original.stat().st_dev:
            raise RuntimeError("Data was not independently copied")
        copied.append({"path": relative.as_posix(), "bytes": original.stat().st_size, "sha256": before})
        total_bytes += original.stat().st_size
    return copied, total_bytes


def save_csv(path, records, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def update_ignore(root):
    path = root / ".gitignore"
    current = path.read_text(encoding="utf-8") if path.exists() else ""
    block = "\n".join([
        "# BEGIN stage1 repository layout",
        "!/datasets/", "/datasets/*", "!/datasets/README.md",
        "!/weights/", "/weights/*", "!/weights/README.md",
        "!/cache/", "/cache/*", "!/cache/README.md",
        "!/outputs/", "/outputs/*", "!/outputs/README.md",
        "/src/lvit/LV_loss/",
        "云服务器账密.txt", "云服务器配置.txt",
        "# END stage1 repository layout",
    ])
    if "# BEGIN stage1 repository layout" not in current:
        path.write_text(current.rstrip() + "\n\n" + block + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--source-data", type=Path, default=Path("/root/autodl-tmp/datasets"))
    parser.add_argument("--source-lvit", type=Path, default=Path("/root/autodl-tmp/对比方法/LViT-main"))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.project_root.resolve(strict=True)
    output = args.output_dir.resolve()
    if not output.is_relative_to(root / "outputs"):
        raise RuntimeError("Report directory must be under project outputs")
    output.mkdir(parents=True, exist_ok=True)
    for name in ["docs", "manifests", "configs", "scripts", "requirements", "src/lvit",
                 "datasets", "weights/lvit", "cache/text_features"]:
        (root / name).mkdir(parents=True, exist_ok=True)
    update_ignore(root)
    log("COPY_DATA_BEGIN", source=str(args.source_data))
    assets, data_bytes = copy_verified(args.source_data, root / "datasets")
    log("COPY_DATA_OK", files=len(assets), bytes=data_bytes)
    source_files, code_bytes = copy_verified(args.source_lvit, root / "src/lvit", exclude_caches=True)
    log("COPY_LVIT_OK", files=len(source_files), bytes=code_bytes)
    frozen = root / "manifests/legacy_subset_v1"
    frozen.mkdir(parents=True, exist_ok=True)
    save_csv(frozen / "asset_hashes.csv", assets, ["path", "bytes", "sha256"])
    created_at = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()
    write_json(root / "src/lvit/source_snapshot.json", {
        "official_repository": "https://github.com/HUANGLIZI/LViT",
        "source": str(args.source_lvit), "kind": "user_uploaded_modified_copy",
        "copied_at": created_at, "files": source_files,
        "excluded": ["__pycache__", "*.pyc", ".git"],
    })
    fields = ["sample_id", "dataset", "split", "image_path", "mask_path", "text_workbook", "text_key",
              "image_sha256", "mask_sha256", "description_sha256", "width", "height",
              "patient_or_series_id", "identifier_kind", "mask_values"]
    all_rows, split_checks = [], {}
    image_hashes = {}
    for task, expected in COUNTS.items():
        task_rows = []
        for folder_name, split_name in SPLITS.items():
            directory = root / "datasets" / task / folder_name
            images = sorted(p for p in (directory / "img").iterdir() if p.is_file())
            masks = {p.name: p for p in (directory / "labelcol").iterdir() if p.is_file()}
            book = directory / folder_name.replace("_Folder", "_text.xlsx")
            parsed = read_xlsx(book)
            if parsed["headers"] != ["Image", "Description"]:
                raise RuntimeError("Unexpected text columns: " + str(book))
            lookup = {row["Image"]: row["Description"] for row in parsed["records"]}
            if len(images) != expected[split_name] or len(masks) != len(images):
                raise RuntimeError("Dataset counts differ from user-confirmed protocol")
            records = []
            used = set()
            for image in images:
                selected_mask = mask_name(image.name, masks)
                mask = directory / "labelcol" / selected_mask
                description = lookup.get(selected_mask)
                if not mask.is_file() or not description:
                    raise RuntimeError("Missing mask or text mapping")
                with Image.open(image) as original:
                    original.load()
                    width, height = original.size
                    image_pixels = np.array(original.convert("RGB"))
                with Image.open(mask) as original:
                    original.load()
                    if original.size != (width, height):
                        raise RuntimeError("Image/mask shape mismatch")
                    values = np.unique(np.array(original.convert("L"))).tolist()
                case_id, kind = infer_case(task, image.name)
                pixel_hash = hashlib.sha256(str(image_pixels.shape).encode() + image_pixels.tobytes()).hexdigest()
                image_hashes[(task, split_name, image.name)] = pixel_hash
                record = {
                    "sample_id": hashlib.sha256((task + "/" + image.name).encode()).hexdigest()[:20],
                    "dataset": task, "split": split_name,
                    "image_path": image.relative_to(root).as_posix(),
                    "mask_path": mask.relative_to(root).as_posix(),
                    "text_workbook": book.relative_to(root).as_posix(), "text_key": selected_mask,
                    "image_sha256": sha(image), "mask_sha256": sha(mask),
                    "description_sha256": hashlib.sha256(description.encode("utf-8")).hexdigest(),
                    "width": width, "height": height,
                    "patient_or_series_id": case_id or "", "identifier_kind": kind or "unknown",
                    "mask_values": json.dumps(values),
                }
                records.append(record)
                used.add(selected_mask)
            if set(lookup) != used or len(parsed["records"]) != len(used):
                raise RuntimeError("Text rows do not match sample keys one-to-one")
            save_csv(frozen / task / (split_name + ".csv"), records, fields)
            task_rows.extend(records)
            all_rows.extend(records)
            log("MANIFEST_OK", dataset=task, split=split_name, samples=len(records))
        checks = []
        for i, first in enumerate(["train", "val", "test"]):
            for second in ["train", "val", "test"][i + 1:]:
                a = [r for r in task_rows if r["split"] == first]
                b = [r for r in task_rows if r["split"] == second]
                ids_a = {r["patient_or_series_id"] for r in a if r["patient_or_series_id"]}
                ids_b = {r["patient_or_series_id"] for r in b if r["patient_or_series_id"]}
                duplicates = set(image_hashes[(task, first, Path(r["image_path"]).name)] for r in a) & set(image_hashes[(task, second, Path(r["image_path"]).name)] for r in b)
                if duplicates:
                    raise RuntimeError("Exact image duplicates across splits")
                checks.append({"first": first, "second": second,
                               "known_case_or_series_overlap": sorted(ids_a & ids_b),
                               "identical_image_count": len(duplicates)})
        split_checks[task] = {
            "comparisons": checks,
            "unknown_identifier_samples": sum(not r["patient_or_series_id"] for r in task_rows),
            "interpretation": "no_overlap_detected_on_available_patient_ids" if task == "monuseg" else "historical_reproduction_with_disclosed_group_overlap",
        }
    if len(all_rows) != 917:
        raise RuntimeError("Unexpected total sample count")
    write_json(frozen / "split_audit.json", split_checks)
    protocol = {
        "protocol_id": "legacy_subset_v1", "created_at": created_at,
        "user_confirmed": {"historical_paper2_splits": True, "retain_membership": True,
                           "execution_order": ["monuseg", "cov19", "mosmeddata"]},
        "historical_split_seed": None, "split_reproducibility": "immutable_lists_and_sha256",
        "training_seeds_planned": [666, 667, 668], "dataset_root": "datasets",
        "manifest_root": "manifests/legacy_subset_v1", "counts": COUNTS,
        "image_input": {"lvit_size": [224, 224], "lvit_color_order": "BGR", "qwen_color_order": "RGB",
                        "grayscale_to_three_channels": True, "image_resize": "bilinear", "lvit_scale": "divide_by_255", "patching": False},
        "mask_input": {"foreground_rule": "grayscale_gt_0", "binarize_before_resize": True,
                       "resize": "nearest", "raw_files_modified": False,
                       "legacy_comparison_policy": "legacy_loader_resize_v1"},
        "metrics": {"primary": "foreground_dice_per_image_macro", "secondary": "foreground_iou_per_image_macro",
                    "probability_threshold": 0.5, "comparison": "greater_equal", "evaluation_grid": [224, 224],
                    "both_empty": 1.0, "one_empty": 0.0, "report_unit": "percent",
                    "denominator": "actual_manifest_length", "checkpoint_selection": "best_validation_macro_dice",
                    "checkpoint_tie": "earliest", "test_used_for_tuning": False},
        "text_sources": ["provided_legacy_text", "qwen_predicted_text", "qwen_predicted_grounded_evidence", "oracle_reference"],
        "legacy_text_provenance": "not_independently_verified",
        "geometric_augmentation_contract": "synchronize_images_masks_boxes_and_spatial_text",
        "smoke_test_geometric_augmentation": False,
        "patient_group_limitations": split_checks,
        "phase2_required": ["LViT dependencies", "text encoder compatibility", "dataset paths and aliases",
                            "nearest mask resize", "actual test denominator", "training hyperparameters"],
    }
    write_json(root / "configs/data_protocol_v1.json", protocol)
    write_json(frozen / "dataset_inventory.json", {"protocol": protocol["protocol_id"],
               "counts": COUNTS, "samples": len(all_rows), "data_files": len(assets), "data_bytes": data_bytes,
               "data_source": str(args.source_data), "project_data_root": "datasets",
               "source_files_preserved": True, "copy_kind": "independent_regular_files",
               "generated_at": created_at})
    source_config = (root / "src/lvit/Config.py").read_text()
    if "monuseg_2" not in source_config:
        log("NOTE", detail="Copied LViT config differs from earlier audit; source snapshot is authoritative")
    environment = sorted((distribution.metadata.get("Name") or "", distribution.version)
                         for distribution in importlib.metadata.distributions())
    (root / "requirements/environment.current.txt").write_text(
        "\n".join(name + "==" + version for name, version in environment if name) + "\n", encoding="utf-8")
    write_json(output / "copy_verification.json", {
        "data_files": len(assets), "data_bytes": data_bytes, "lvit_files": len(source_files),
        "lvit_bytes": code_bytes, "all_hashes_matched": True, "source_files_unchanged": True,
        "independent_copies": True, "dataset_source": str(args.source_data), "lvit_source": str(args.source_lvit),
    })
    log("PREPARATION_FINISHED", protocol=protocol["protocol_id"], samples=len(all_rows))
    checked = subprocess.run(
        [str(Path("/root/miniconda3/bin/python")), str(root / "scripts/verify_stage1.py"), "--output", str(output / "verification.json")],
        cwd=root, check=False)
    if checked.returncode:
        raise SystemExit(checked.returncode)


if __name__ == "__main__":
    main()

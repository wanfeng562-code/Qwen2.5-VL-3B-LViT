"""Verify the fixed dataset protocol without loading BERT or training a model."""

import argparse
import csv
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
from audit_processed_datasets import infer_case, read_xlsx
from protocol_metrics import binary_sample_scores, macro_scores


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    protocol = json.loads((root / "configs/data_protocol_v1.json").read_text(encoding="utf-8"))
    frozen = root / protocol["manifest_root"]
    inventory = json.loads((frozen / "dataset_inventory.json").read_text(encoding="utf-8"))
    asset_rows = read_csv(frozen / "asset_hashes.csv")
    expected_counts = {"monuseg": {"train": 24, "val": 6, "test": 14},
                       "cov19": {"train": 160, "val": 40, "test": 200},
                       "mosmeddata": {"train": 160, "val": 40, "test": 273}}
    assert protocol["counts"] == expected_counts
    assert protocol["historical_split_seed"] is None
    assert protocol["user_confirmed"]["execution_order"] == ["monuseg", "cov19", "mosmeddata"]
    assert protocol["mask_input"]["foreground_rule"] == "grayscale_gt_0"
    assert protocol["mask_input"]["resize"] == "nearest"
    assert protocol["metrics"]["denominator"] == "actual_manifest_length"
    checked_assets = 0
    for asset in asset_rows:
        path = root / "datasets" / asset["path"]
        assert path.resolve().is_relative_to(root / "datasets")
        assert path.is_file() and not path.is_symlink(), str(path)
        assert path.stat().st_size == int(asset["bytes"]), str(path)
        assert sha(path) == asset["sha256"], str(path)
        checked_assets += 1
    assert checked_assets == inventory["data_files"] == 1845

    text_cache = {}
    sample_ids = set()
    samples, counts, observed_cases = 0, {}, {}
    for task, splits in expected_counts.items():
        counts[task] = {}
        observed_cases[task] = {}
        names_by_split = {}
        for split, expected in splits.items():
            rows = read_csv(frozen / task / (split + ".csv"))
            assert len(rows) == expected
            counts[task][split] = len(rows)
            observed_cases[task][split] = set()
            names_by_split[split] = set()
            for row in rows:
                assert row["dataset"] == task and row["split"] == split
                assert row["sample_id"] not in sample_ids
                sample_ids.add(row["sample_id"])
                for field, checksum in [("image_path", "image_sha256"), ("mask_path", "mask_sha256")]:
                    path = root / row[field]
                    assert path.resolve().is_relative_to(root / "datasets")
                    assert path.is_file() and sha(path) == row[checksum]
                book = root / row["text_workbook"]
                if str(book) not in text_cache:
                    parsed = read_xlsx(book)
                    assert parsed["headers"] == ["Image", "Description"]
                    lookup = {record["Image"]: record["Description"] for record in parsed["records"]}
                    assert len(lookup) == len(parsed["records"])
                    text_cache[str(book)] = lookup
                text = text_cache[str(book)][row["text_key"]]
                assert text.strip()
                assert hashlib.sha256(text.encode("utf-8")).hexdigest() == row["description_sha256"]
                case, kind = infer_case(task, Path(row["image_path"]).name)
                assert row["patient_or_series_id"] == (case or "")
                assert row["identifier_kind"] == (kind or "unknown")
                if case:
                    observed_cases[task][split].add(case)
                names_by_split[split].add(Path(row["image_path"]).name)
                samples += 1
        for i, first in enumerate(["train", "val", "test"]):
            for second in ["train", "val", "test"][i + 1:]:
                assert not names_by_split[first] & names_by_split[second]
                expected_overlap = next(item["known_case_or_series_overlap"]
                                        for item in protocol["patient_group_limitations"][task]["comparisons"]
                                        if item["first"] == first and item["second"] == second)
                assert sorted(observed_cases[task][first] & observed_cases[task][second]) == expected_overlap
    assert samples == len(sample_ids) == 917
    assert len(text_cache) == 9
    assert not observed_cases["monuseg"]["train"] & observed_cases["monuseg"]["val"]
    assert not observed_cases["monuseg"]["train"] & observed_cases["monuseg"]["test"]
    assert not observed_cases["monuseg"]["val"] & observed_cases["monuseg"]["test"]

    # Edge cases and real denominators are part of the protocol, not model tests.
    empty = np.zeros((2, 2))
    foreground = np.ones((2, 2))
    assert binary_sample_scores(empty, empty) == {"dice": 1.0, "iou": 1.0}
    assert binary_sample_scores(foreground, empty) == {"dice": 0.0, "iou": 0.0}
    assert binary_sample_scores(empty, foreground) == {"dice": 0.0, "iou": 0.0}
    assert binary_sample_scores(np.full((2, 2), 0.5), foreground) == {"dice": 1.0, "iou": 1.0}
    assert macro_scores([{"dice": 1.0, "iou": 1.0}, {"dice": 0.0, "iou": 0.0}]) == {"sample_count": 2, "dice": 0.5, "iou": 0.5}

    for relative in [
        "datasets/monuseg/Train_Folder/img/example.png",
        "weights/Qwen2.5-VL-3B-Instruct/model.safetensors",
        "cache/text_features/example.npy", "outputs/stage1/result.json",
        "src/lvit/LV_loss/fig1.png", "云服务器账密.txt", ".env",
    ]:
        checked = subprocess.run(["git", "-C", str(root), "check-ignore", "--no-index", relative], capture_output=True)
        assert checked.returncode == 0, relative
    for relative in ["README.md", "datasets/README.md", "weights/README.md", "cache/README.md", "outputs/README.md",
                     "src/lvit/Load_Dataset.py", "configs/data_protocol_v1.json", "manifests/legacy_subset_v1/monuseg/train.csv"]:
        checked = subprocess.run(["git", "-C", str(root), "check-ignore", "--no-index", relative], capture_output=True)
        assert checked.returncode == 1, relative
    for relative in ["src/lvit/LICENSE", "src/lvit/SOURCE.md", "src/lvit/source_snapshot.json",
                     "docs/阶段一数据与实验协议.md", "requirements/environment.current.txt"]:
        assert (root / relative).is_file(), relative

    result = {
        "status": "passed", "protocol": protocol["protocol_id"], "samples": samples,
        "verified_data_assets": checked_assets, "text_workbooks": len(text_cache),
        "counts": counts, "dataset_hashes_match": True,
        "known_overlap_disclosed_and_frozen": True, "monuseg_patient_ids_disjoint": True,
        "metric_edge_cases_passed": True, "git_ignore_rules_passed": True,
        "scope": "Phase-one data and protocol validation; does not execute the legacy BERT/DataLoader or train LViT.",
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()

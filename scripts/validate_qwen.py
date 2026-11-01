"""Validate local Qwen image inference and save timings and GPU memory metrics."""

import argparse
import csv
import importlib.metadata
import json
import os
import statistics
import subprocess
import threading
import time
import traceback
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from PIL import Image, ImageDraw
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

GIB = 1024**3
TIMEZONE = ZoneInfo("Asia/Shanghai")


def log(message):
    print(f"[{datetime.now(TIMEZONE).isoformat(timespec='seconds')}] {message}", flush=True)


def cuda_memory():
    return {
        "allocated_gib": round(torch.cuda.memory_allocated() / GIB, 4),
        "reserved_gib": round(torch.cuda.memory_reserved() / GIB, 4),
        "peak_allocated_gib": round(torch.cuda.max_memory_allocated() / GIB, 4),
        "peak_reserved_gib": round(torch.cuda.max_memory_reserved() / GIB, 4),
    }


class GPUMonitor:
    def __init__(self, path):
        self.path = path
        self.samples = []
        self.process = None
        self.thread = None
        self.errors = []

    def start(self):
        self.process = subprocess.Popen(
            [
                "nvidia-smi",
                "--query-gpu=timestamp,memory.used,utilization.gpu,power.draw",
                "--format=csv,noheader,nounits",
                "--loop-ms=200",
                "--id=0",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        def read():
            with self.path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.writer(stream)
                writer.writerow(["timestamp", "memory_used_mib", "utilization_percent", "power_watts"])
                stream.flush()
                for line in self.process.stdout:
                    row = next(csv.reader([line]))
                    if len(row) != 4:
                        continue
                    writer.writerow([value.strip() for value in row])
                    stream.flush()
                    try:
                        self.samples.append(float(row[1]))
                    except ValueError:
                        self.errors.append("Unparseable memory sample")

        self.thread = threading.Thread(target=read, daemon=True)
        self.thread.start()

    def close(self):
        if self.process is not None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)
            self.thread.join(timeout=5)
            error = self.process.stderr.read().strip()
            if error:
                self.errors.append(error)
        return {
            "sample_interval_ms": 200,
            "sample_count": len(self.samples),
            "sampled_peak_used_mib": max(self.samples) if self.samples else None,
            "sampled_peak_used_gib": round(max(self.samples) / 1024, 4) if self.samples else None,
            "errors": self.errors,
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--image", type=Path, help="Optional real image; otherwise create a deterministic shapes fixture.")
    parser.add_argument("--runs", type=int, default=3)
    args = parser.parse_args()
    if args.runs < 1:
        parser.error("--runs must be positive")

    root = Path(__file__).resolve().parents[1]
    model_dir = args.model_dir or root / "weights/Qwen2.5-VL-3B-Instruct"
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    result = {
        "status": "running",
        "started_at": datetime.now(TIMEZONE).isoformat(),
        "model_directory": str(model_dir),
        "output_directory": str(out),
        "offline_model_loading": True,
        "run_count": args.runs,
    }
    monitor = None
    start = time.perf_counter()
    try:
        versions = {
            name: importlib.metadata.version(name)
            for name in ["torch", "torchvision", "transformers", "accelerate", "qwen-vl-utils", "Pillow"]
        }
        versions["python"] = os.sys.version
        versions["pytorch_cuda"] = torch.version.cuda
        result["environment"] = versions
        (out / "environment.txt").write_text(json.dumps(versions, indent=2), encoding="utf-8")
        log("START: offline local Qwen validation")
        log("ENVIRONMENT: " + json.dumps(versions))
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable")
        torch.cuda.set_device(0)
        torch.cuda.init()
        free, total = torch.cuda.mem_get_info()
        if free < 12 * GIB:
            raise RuntimeError(f"Insufficient free GPU memory for this test: {free / GIB:.2f} GiB")
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("GPU does not support the configured BF16 test")
        result["gpu"] = {
            "name": torch.cuda.get_device_name(0),
            "total_memory_gib": round(total / GIB, 4),
            "initial_free_memory_gib": round(free / GIB, 4),
            "driver": subprocess.check_output(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader", "--id=0"], text=True
            ).strip(),
        }
        monitor = GPUMonitor(out / "gpu_memory.csv")
        monitor.start()

        image_path = out / "test_image.png"
        if args.image:
            image = Image.open(args.image).convert("RGB")
            fixture = {"kind": "user_image", "source": str(args.image.resolve())}
        else:
            image = Image.new("RGB", (640, 384), "white")
            draw = ImageDraw.Draw(image)
            draw.ellipse((70, 90, 270, 290), fill=(230, 30, 30))
            draw.rectangle((400, 110, 570, 280), fill=(30, 70, 230))
            fixture = {
                "kind": "synthetic_shapes",
                "expected_content": "A red circle on the left and a blue square on the right, on a white background.",
                "limitation": "Checks image input and generation functionality; not a medical accuracy evaluation.",
            }
        image.save(image_path)
        result["image"] = {"path": str(image_path), "width": image.width, "height": image.height, **fixture}
        prompt = "Describe the shapes, their colors, and their positions in this image in one or two sentences."
        if args.image:
            prompt = "Describe the visible contents of this image in one or two sentences."
        result["settings"] = {
            "dtype": "bfloat16",
            "attention": "sdpa",
            "batch_size": 1,
            "min_pixels": 256 * 28 * 28,
            "max_pixels": 1280 * 28 * 28,
            "max_new_tokens": 128,
            "do_sample": False,
            "prompt": prompt,
        }
        log(f"TEST_IMAGE: {image_path} ({image.width}x{image.height}; {fixture['kind']})")

        processor_start = time.perf_counter()
        processor = AutoProcessor.from_pretrained(
            str(model_dir),
            local_files_only=True,
            min_pixels=result["settings"]["min_pixels"],
            max_pixels=result["settings"]["max_pixels"],
            use_fast=False,
        )
        result["processor_load_seconds"] = round(time.perf_counter() - processor_start, 4)
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        load_start = time.perf_counter()
        log("MODEL_LOAD_BEGIN")
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            str(model_dir),
            local_files_only=True,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
            device_map={"": 0},
            low_cpu_mem_usage=True,
        ).eval().requires_grad_(False)
        torch.cuda.synchronize()
        result["model_load_seconds"] = round(time.perf_counter() - load_start, 4)
        result["memory_after_model_load"] = cuda_memory()
        result["model_parameter_count"] = sum(p.numel() for p in model.parameters())
        result["parameter_devices"] = sorted({str(p.device) for p in model.parameters()})
        if result["parameter_devices"] != ["cuda:0"]:
            raise RuntimeError("Some parameters are not on the GPU")
        log("MODEL_LOAD_OK: " + json.dumps({
            "seconds": result["model_load_seconds"],
            "memory": result["memory_after_model_load"],
            "devices": result["parameter_devices"],
        }))

        preprocessing_start = time.perf_counter()
        messages = [{
            "role": "user",
            "content": [{"type": "image", "image": str(image_path)}, {"type": "text", "text": prompt}],
        }]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        images, videos = process_vision_info(messages)
        inputs = processor(text=[text], images=images, videos=videos, padding=True, return_tensors="pt")
        result["preprocessing_seconds"] = round(time.perf_counter() - preprocessing_start, 4)
        result["input_tokens"] = int(inputs["input_ids"].shape[-1])
        result["image_grid_thw"] = inputs["image_grid_thw"].tolist()
        transfer_start = time.perf_counter()
        inputs = inputs.to("cuda:0")
        torch.cuda.synchronize()
        result["input_transfer_seconds"] = round(time.perf_counter() - transfer_start, 4)
        log("INPUTS_READY: " + json.dumps({
            "input_tokens": result["input_tokens"],
            "image_grid_thw": result["image_grid_thw"],
            "preprocessing_seconds": result["preprocessing_seconds"],
        }))

        runs = []
        for i in range(args.runs):
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            generation_start = time.perf_counter()
            with torch.inference_mode():
                generated = model.generate(
                    **inputs, max_new_tokens=128, do_sample=False, use_cache=True
                )
            torch.cuda.synchronize()
            seconds = time.perf_counter() - generation_start
            new_ids = generated[:, inputs["input_ids"].shape[-1]:]
            tokens = int(new_ids.shape[-1])
            decode_start = time.perf_counter()
            response = processor.batch_decode(
                new_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]
            decode_seconds = time.perf_counter() - decode_start
            if not response.strip():
                raise RuntimeError("The model returned an empty response")
            metric = {
                "run": i + 1,
                "kind": "first_call" if i == 0 else "warm_call",
                "generation_seconds": round(seconds, 4),
                "decode_seconds": round(decode_seconds, 4),
                "new_tokens_including_eos": tokens,
                "tokens_per_second": round(tokens / seconds, 3),
                "memory": cuda_memory(),
                "response": response,
            }
            runs.append(metric)
            log("GENERATION_OK: " + json.dumps(metric, ensure_ascii=False))
            del generated, new_ids
        result["runs"] = runs
        if len(runs) > 1:
            result["warm_mean_generation_seconds"] = round(statistics.mean(r["generation_seconds"] for r in runs[1:]), 4)
            result["warm_mean_tokens_per_second"] = round(statistics.mean(r["tokens_per_second"] for r in runs[1:]), 3)
        result["overall_peak_allocated_gib"] = max(
            [result["memory_after_model_load"]["peak_allocated_gib"]] + [r["memory"]["peak_allocated_gib"] for r in runs]
        )
        result["overall_peak_reserved_gib"] = max(
            [result["memory_after_model_load"]["peak_reserved_gib"]] + [r["memory"]["peak_reserved_gib"] for r in runs]
        )
        result["status"] = "passed"
    except Exception as error:
        result["status"] = "failed"
        result["error_type"] = type(error).__name__
        result["error"] = str(error)
        traceback.print_exc()
    finally:
        if monitor is not None:
            result["nvidia_smi_monitor"] = monitor.close()
        result["total_validation_seconds_excluding_imports"] = round(time.perf_counter() - start, 4)
        result["finished_at"] = datetime.now(TIMEZONE).isoformat()
        (out / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        log(f"FINISHED: {result['status']}; results={out / 'result.json'}")
    if result["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

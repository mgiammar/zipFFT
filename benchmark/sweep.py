"""Sweeps a list configuration benchmarking zipFFT against the naive PyTorch reference.

Shape lists use the same schema as configs.yaml's real_conv2d/complex_conv2d entries:
    - {signal_y: .., signal_x: .., fft_y: .., fft_x: .., batch: ..,
      cross_correlate: true|false, use_tiled_swizzled_io: true|false, ffts_per_block_y:
      ..}

Usage
-----
# Sweep a handful of built-in smoke-test shapes for the real path:
python sweep.py --path=real

# Sweep every shape already declared in the repo's configs.yaml:
python sweep.py --path=real --shapes-file=../configs.yaml --shapes-key=real_conv2d

# Sweep a custom shape list, with correctness checking against installed zipfft package:
python sweep.py --path=complex --shapes-file=my_shapes.yaml --check-correctness
"""

import argparse
import csv
import json
import re
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import yaml

import build as build_mod

BENCHMARK_DIR = Path(__file__).resolve().parent
REPO_ROOT = BENCHMARK_DIR.parent

DEFAULT_SHAPES = {
    "real": [
        {
            "signal_y": 96,
            "signal_x": 96,
            "fft_y": 128,
            "fft_x": 128,
            "batch": 8,
            "cross_correlate": True,
        },
        {
            "signal_y": 192,
            "signal_x": 192,
            "fft_y": 256,
            "fft_x": 256,
            "batch": 4,
            "cross_correlate": True,
        },
        {
            "signal_y": 384,
            "signal_x": 384,
            "fft_y": 512,
            "fft_x": 512,
            "batch": 1,
            "cross_correlate": True,
            "use_tiled_swizzled_io": True,
            "ffts_per_block_y": 4,
        },
    ],
    "complex": [
        {
            "signal_y": 96,
            "signal_x": 96,
            "fft_y": 128,
            "fft_x": 128,
            "batch": 4,
            "cross_correlate": True,
        },
        {
            "signal_y": 192,
            "signal_x": 192,
            "fft_y": 256,
            "fft_x": 256,
            "batch": 4,
            "cross_correlate": True,
        },
        {
            "signal_y": 384,
            "signal_x": 192,
            "fft_y": 512,
            "fft_x": 256,
            "batch": 1,
            "cross_correlate": True,
            "use_tiled_swizzled_io": True,
            "ffts_per_block_y": 4,
        },
    ],
}


def load_shapes(shapes_file: str | None, shapes_key: str, path: str) -> list[dict]:
    if not shapes_file:
        return DEFAULT_SHAPES[path]
    shapes_path = Path(shapes_file)
    with open(shapes_path) as f:
        data = yaml.safe_load(f)
    if shapes_key not in data:
        raise KeyError(
            f"'{shapes_key}' not found in {shapes_path} (top-level keys: {list(data)})"
        )
    return data[shapes_key]


def slugify_gpu_name(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "-", name).strip("-").lower()


def git_commit_hash() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except Exception:
        return "unknown"


def run_cuda_benchmark(path: str, shape: dict, args) -> dict:
    op = "corr" if shape.get("cross_correlate", True) else "conv"
    binary = build_mod.build_binary(
        target=path,
        fft_x=shape["fft_x"],
        fft_y=shape["fft_y"],
        signal_x=shape["signal_x"],
        signal_y=shape["signal_y"],
        batch=shape["batch"],
        op=op,
        arch=args.arch,
        swizzled=shape.get("use_tiled_swizzled_io", False),
        ffts_per_block_y=shape.get("ffts_per_block_y", 0),
        nvcc=args.nvcc,
        verbose=args.verbose,
    )

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        json_path = tmp.name

    cmd = [
        str(binary),
        f"--warmup={args.warmup}",
        f"--iters={args.iters}",
        f"--device={args.device}",
        f"--json={json_path}",
    ]
    if args.peak_flops_tflops is not None:
        cmd.append(f"--peak-flops-tflops={args.peak_flops_tflops}")
    if args.peak_bandwidth_gbs is not None:
        cmd.append(f"--peak-bandwidth-gbs={args.peak_bandwidth_gbs}")

    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"{binary} failed:\n{proc.stderr}")

    with open(json_path) as f:
        return json.load(f)


def run_pytorch_benchmark(path: str, shape: dict, args) -> dict:
    op = "corr" if shape.get("cross_correlate", True) else "conv"
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        json_path = tmp.name

    cmd = [
        sys.executable,
        str(BENCHMARK_DIR / "pytorch_ref.py"),
        f"--path={path}",
        f"--op={op}",
        f"--fft-x={shape['fft_x']}",
        f"--fft-y={shape['fft_y']}",
        f"--signal-x={shape['signal_x']}",
        f"--signal-y={shape['signal_y']}",
        f"--batch={shape['batch']}",
        f"--warmup={args.warmup}",
        f"--iters={args.iters}",
        f"--device={args.device}",
        f"--json={json_path}",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"pytorch_ref.py failed:\n{proc.stderr}")

    with open(json_path) as f:
        return json.load(f)


def _short_error(e: Exception) -> str:
    """PyTorch TORCH_CHECK errors embed a full C++ backtrace in str(e); keep only the message."""
    return str(e).split("Exception raised from")[0].strip()


def check_correctness(path: str, shape: dict, device: str) -> dict | None:
    """Compares zipfft's installed python bindings against a naive PyTorch reference for
    one shape. Returns None (with a warning) if the zipfft package or the requested
    extension isn't available, rather than failing the whole sweep -- correctness
    checking is opt-in and orthogonal to performance benchmarking.
    """
    try:
        import torch  # noqa: F401  (must precede `import zipfft`, see zipfft/__init__.py)
        import zipfft
    except ImportError as e:
        print(f"  [correctness] skipped: {e}")
        return None

    module_name = "padded_rconv2d" if path == "real" else "padded_cconv2d"
    if not zipfft.is_extension_available(module_name):
        print(
            f"  [correctness] skipped: {module_name} "
            "extension not compiled into this zipfft install"
        )
        return None

    fft_y, fft_x = shape["fft_y"], shape["fft_x"]
    sig_y, sig_x = shape["signal_y"], shape["signal_x"]
    batch = shape["batch"]
    cross_correlate = shape.get("cross_correlate", True)
    valid_y, valid_x = fft_y - sig_y + 1, fft_x - sig_x + 1

    module = getattr(zipfft, module_name)
    op_fn = module.corr if cross_correlate else module.conv

    if path == "real":
        image = torch.randn(fft_y, fft_x, dtype=torch.float32, device=device)
        template = torch.randn(batch, sig_y, sig_x, dtype=torch.float32, device=device)

        image_fft = torch.fft.rfft2(image)  # (fft_y, fft_x // 2 + 1)
        conv_data = image_fft.transpose(-2, -1).contiguous()  # (fft_x // 2 + 1, fft_y)

        template_fft = torch.fft.rfft2(template, s=(fft_y, fft_x))
        prod = (
            torch.conj(template_fft) * image_fft
            if cross_correlate
            else template_fft * image_fft
        )
        torch_out = torch.fft.irfft2(prod, s=(fft_y, fft_x))[
            :, :valid_y, :valid_x
        ].contiguous()

        workspace = torch.empty(
            batch, fft_y, fft_x // 2 + 1, dtype=torch.complex64, device=device
        )
        zipfft_out = torch.empty(
            batch, valid_y, valid_x, dtype=torch.float32, device=device
        )
        try:
            op_fn(template, workspace, conv_data, zipfft_out, fft_y, fft_x)
        except RuntimeError as e:
            print(
                f"  [correctness] skipped: shape not in the installed zipfft's "
                "compiled configs (add it to configs.yaml + rebuild to check it): "
                f"{_short_error(e)}"
            )
            return None
    else:
        image = torch.complex(
            torch.randn(fft_y, fft_x, device=device),
            torch.randn(fft_y, fft_x, device=device),
        )
        template = torch.complex(
            torch.randn(batch, sig_y, sig_x, device=device),
            torch.randn(batch, sig_y, sig_x, device=device),
        )

        image_fft = torch.fft.fft2(image)  # (fft_y, fft_x)
        conv_data = image_fft.transpose(-2, -1).contiguous()  # (fft_x, fft_y)

        template_fft = torch.fft.fft2(template, s=(fft_y, fft_x))
        prod = (
            torch.conj(template_fft) * image_fft
            if cross_correlate
            else template_fft * image_fft
        )
        torch_out = torch.fft.ifft2(prod)[:, :valid_y, :valid_x].contiguous()

        workspace = torch.empty(
            batch, fft_y, fft_x, dtype=torch.complex64, device=device
        )
        zipfft_out = torch.empty(
            batch, valid_y, valid_x, dtype=torch.complex64, device=device
        )
        try:
            op_fn(template, workspace, conv_data, zipfft_out, fft_y, fft_x)
        except RuntimeError as e:
            print(
                f"  [correctness] skipped: shape not in the installed zipfft's "
                "compiled configs (add it to configs.yaml + rebuild to check it): "
                f"{_short_error(e)}"
            )
            return None

    abs_diff = torch.abs(torch_out - zipfft_out)
    return {
        "max_abs_diff": abs_diff.max().item(),
        "mean_abs_diff": abs_diff.mean().item(),
    }


def flatten_row(
    shape: dict,
    path: str,
    cuda_result: dict,
    pytorch_result: dict,
    correctness: dict | None,
) -> dict:
    zipfft_min_ms = (
        cuda_result["timing_ms"]["min"] / cuda_result["timing_ms"]["repeats_per_sample"]
    )
    pytorch_min_ms = pytorch_result["timing_ms"]["min"]
    row = {
        "path": path,
        "operation": cuda_result["operation"],
        "fft_y": shape["fft_y"],
        "fft_x": shape["fft_x"],
        "signal_y": shape["signal_y"],
        "signal_x": shape["signal_x"],
        "batch": shape["batch"],
        "use_tiled_swizzled_io": shape.get("use_tiled_swizzled_io", False),
        "ffts_per_block_y": shape.get("ffts_per_block_y", 0),
        "zipfft_min_ms": zipfft_min_ms,
        "zipfft_mean_ms": cuda_result["timing_ms"]["mean"],
        "zipfft_std_ms": cuda_result["timing_ms"]["std_dev"],
        "zipfft_gflops": cuda_result["performance"]["throughput_gflops"],
        "zipfft_bandwidth_gbs": cuda_result["performance"]["achieved_bandwidth_gbs"],
        "roofline_efficiency_percent": cuda_result["performance"][
            "roofline_efficiency_percent"
        ],
        "bandwidth_efficiency_percent": cuda_result["performance"][
            "bandwidth_efficiency_percent"
        ],
        "pytorch_min_ms": pytorch_min_ms,
        "pytorch_mean_ms": pytorch_result["timing_ms"]["mean"],
        "speedup_vs_pytorch": pytorch_min_ms / zipfft_min_ms if zipfft_min_ms else None,
        "correctness_max_abs_diff": (
            correctness["max_abs_diff"] if correctness else None
        ),
        "correctness_mean_abs_diff": (
            correctness["mean_abs_diff"] if correctness else None
        ),
    }
    return row


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--path", choices=["real", "complex"], required=True)
    p.add_argument("--shapes-file", default=None)
    p.add_argument("--shapes-key", default=None, help="Defaults to '<path>_conv2d'.")
    p.add_argument("--arch", default="8.9")
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--nvcc", default=None)
    p.add_argument("--peak-flops-tflops", type=float, default=None)
    p.add_argument("--peak-bandwidth-gbs", type=float, default=None)
    p.add_argument("--check-correctness", action="store_true")
    p.add_argument("--results-dir", default=str(BENCHMARK_DIR / "results"))
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    shapes_key = args.shapes_key or f"{args.path}_conv2d"
    shapes = load_shapes(args.shapes_file, shapes_key, args.path)

    import torch

    gpu_name = torch.cuda.get_device_name(args.device)
    run_dir = (
        Path(args.results_dir)
        / f"{slugify_gpu_name(gpu_name)}_{datetime.now():%Y%m%d-%H%M%S}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)

    rows, cuda_raw, pytorch_raw = [], [], []
    for i, shape in enumerate(shapes):
        print(
            f"[{i + 1}/{len(shapes)}] {args.path} fft={shape['fft_y']}x{shape['fft_x']} "
            f"signal={shape['signal_y']}x{shape['signal_x']} batch={shape['batch']} "
            f"swizzled={shape.get('use_tiled_swizzled_io', False)}"
        )

        cuda_result = run_cuda_benchmark(args.path, shape, args)
        pytorch_result = run_pytorch_benchmark(args.path, shape, args)
        correctness = (
            check_correctness(args.path, shape, f"cuda:{args.device}")
            if args.check_correctness
            else None
        )

        cuda_raw.append(cuda_result)
        pytorch_raw.append(pytorch_result)
        row = flatten_row(shape, args.path, cuda_result, pytorch_result, correctness)
        rows.append(row)

        print(
            f"  zipfft={row['zipfft_min_ms']:.4f}ms  "
            f"pytorch={row['pytorch_min_ms']:.4f}ms  "
            f"speedup={row['speedup_vs_pytorch']:.2f}x  "
            f"roofline={row['roofline_efficiency_percent']:.1f}%"
            + (
                f"  max_abs_diff={correctness['max_abs_diff']:.2e}"
                if correctness
                else ""
            )
        )

    manifest = {
        "path": args.path,
        "num_shapes": len(shapes),
        "shapes": shapes,
        "git_commit": git_commit_hash(),
        "gpu_name": gpu_name,
        "warmup": args.warmup,
        "iters": args.iters,
        "check_correctness": args.check_correctness,
        "timestamp": datetime.now().isoformat(),
    }
    with open(run_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    with open(run_dir / "results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    with open(run_dir / "cuda_raw.jsonl", "w") as f:
        for r in cuda_raw:
            f.write(json.dumps(r) + "\n")
    with open(run_dir / "pytorch_raw.jsonl", "w") as f:
        for r in pytorch_raw:
            f.write(json.dumps(r) + "\n")

    print(f"\nWrote {len(rows)} results to {run_dir}/")


if __name__ == "__main__":
    main()

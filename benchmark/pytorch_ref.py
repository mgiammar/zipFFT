"""Naive PyTorch FFT-based 2D convolution/cross-correlation reference benchmark.

Usage
-----
python pytorch_ref.py --path=real --op=corr --fft-x=4096 --fft-y=4096 \\
    --signal-x=512 --signal-y=512 --batch=16 --json=result.json
"""

import argparse
import json
import time

import torch


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--path", choices=["real", "complex"], required=True)
    p.add_argument("--op", choices=["corr", "conv"], default="corr")
    p.add_argument("--fft-x", type=int, required=True)
    p.add_argument("--fft-y", type=int, required=True)
    p.add_argument("--signal-x", type=int, required=True)
    p.add_argument("--signal-y", type=int, required=True)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--json", default=None, help="Path to write JSON result to.")
    return p.parse_args()


def make_op(path: str, op: str, fft_shape: tuple[int, int]):
    """Returns a callable(image, template) -> output implementing the naive FFT path."""
    if path == "real":

        def run(image: torch.Tensor, template: torch.Tensor) -> torch.Tensor:
            image_fft = torch.fft.rfft2(image, s=fft_shape)
            template_fft = torch.fft.rfft2(template, s=fft_shape)
            prod = (
                image_fft * torch.conj(template_fft)
                if op == "corr"
                else image_fft * template_fft
            )
            return torch.fft.irfft2(prod, s=fft_shape)

    else:

        def run(image: torch.Tensor, template: torch.Tensor) -> torch.Tensor:
            image_fft = torch.fft.fft2(image, s=fft_shape)
            template_fft = torch.fft.fft2(template, s=fft_shape)
            prod = (
                image_fft * torch.conj(template_fft)
                if op == "corr"
                else image_fft * template_fft
            )
            return torch.fft.ifft2(prod, s=fft_shape)

    return run


def main():
    args = parse_args()
    device = f"cuda:{args.device}"
    torch.cuda.set_device(device)

    dtype = torch.float32 if args.path == "real" else torch.complex64
    fft_shape = (args.fft_y, args.fft_x)
    valid_y = args.fft_y - args.signal_y + 1
    valid_x = args.fft_x - args.signal_x + 1

    if args.path == "real":
        image = torch.randn(
            args.batch, args.fft_y, args.fft_x, dtype=dtype, device=device
        )
        template = torch.randn(
            args.batch, args.signal_y, args.signal_x, dtype=dtype, device=device
        )
    else:
        image = torch.complex(
            torch.randn(args.batch, args.fft_y, args.fft_x, device=device),
            torch.randn(args.batch, args.fft_y, args.fft_x, device=device),
        )
        template = torch.complex(
            torch.randn(args.batch, args.signal_y, args.signal_x, device=device),
            torch.randn(args.batch, args.signal_y, args.signal_x, device=device),
        )

    run = make_op(args.path, args.op, fft_shape)

    def iterate():
        out = run(image, template)
        return out[:, :valid_y, :valid_x]

    for _ in range(args.warmup):
        iterate()
    torch.cuda.synchronize()

    times_ms = []
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    for _ in range(args.iters):
        start_event.record()
        iterate()
        end_event.record()
        torch.cuda.synchronize()
        times_ms.append(start_event.elapsed_time(end_event))

    min_ms = min(times_ms)
    mean_ms = sum(times_ms) / len(times_ms)
    max_ms = max(times_ms)
    variance = sum((t - mean_ms) ** 2 for t in times_ms) / len(times_ms)
    std_ms = variance**0.5

    result = {
        "tool": "pytorch_ref",
        "schema_version": 1,
        "path": args.path,
        "operation": args.op,
        "shape": {
            "fft_x": args.fft_x,
            "fft_y": args.fft_y,
            "signal_x": args.signal_x,
            "signal_y": args.signal_y,
            "batch": args.batch,
        },
        "timing_ms": {
            "min": min_ms,
            "max": max_ms,
            "mean": mean_ms,
            "std_dev": std_ms,
            "num_samples": len(times_ms),
            "repeats_per_sample": 1,
        },
        "gpu": {
            "name": torch.cuda.get_device_name(args.device),
            "device_id": args.device,
        },
    }

    print(
        f"[pytorch_ref] path={args.path} op={args.op} fft={args.fft_y}x{args.fft_x} "
        f"signal={args.signal_y}x{args.signal_x} batch={args.batch}  "
        f"min={min_ms:.4f}ms mean={mean_ms:.4f}ms std={std_ms:.4f}ms"
    )

    if args.json:
        with open(args.json, "w") as f:
            json.dump(result, f, indent=2)
    else:
        print(json.dumps(result))


if __name__ == "__main__":
    main()

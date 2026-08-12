"""Compiles bench_real_conv2d.cu / bench_complex_conv2d.cu for a specific shape +
IO-path configuration into a uniquely-named binary under benchmark/build/.

Usage
-----
python build.py --target=real --fft-x=4096 --fft-y=4096 --signal-x=512 \\
    --signal-y=512 --batch=16 --arch=8.9

python build.py --target=complex --fft-x=512 --fft-y=512 --signal-x=384 \\
    --signal-y=384 --batch=4 --arch=8.9 --swizzled --ffts-per-block-y=4

Also importable: `from build import build_binary` returns the compiled binary's Path,
used by sweep.py to avoid shell-ing out to this script per shape.
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

BENCHMARK_DIR = Path(__file__).resolve().parent
REPO_ROOT = BENCHMARK_DIR.parent
BUILD_DIR = BENCHMARK_DIR / "build"

DRIVER_FILES = {
    "real": BENCHMARK_DIR / "bench_real_conv2d.cu",
    "complex": BENCHMARK_DIR / "bench_complex_conv2d.cu",
}


def get_mathdx_include_dir():
    """Locate cuFFTDx headers bundled in `nvidia-mathdx` package, if installed."""
    try:
        import nvidia.mathdx
    except ImportError:
        return None

    for search_path in nvidia.mathdx.__path__:
        include_dir = os.path.join(search_path, "include")
        if os.path.isdir(include_dir):
            return include_dir
    return None


def resolve_nvcc(explicit: str | None) -> str:
    if explicit:
        return explicit
    if os.environ.get("NVCC"):
        return os.environ["NVCC"]
    cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    if cuda_home:
        candidate = Path(cuda_home) / "bin" / "nvcc"
        if candidate.exists():
            return str(candidate)
    found = shutil.which("nvcc")
    if found:
        return found
    raise RuntimeError(
        "Could not locate nvcc. "
        "Set --nvcc=path, or the NVCC/CUDA_HOME environment variable."
    )


def unique_binary_name(
    target: str,
    op: str,
    fft_x: int,
    fft_y: int,
    signal_x: int,
    signal_y: int,
    batch: int,
    swizzled: bool,
    ffts_per_block_y: int,
) -> str:
    io_tag = f"swzY{ffts_per_block_y}" if swizzled else "plain"
    res = f"bench_{target}_{op}"
    res += f"_fft{fft_y}x{fft_x}"
    res += f"_sig{signal_y}x{signal_x}"
    res += f"_b{batch}"
    res += f"_{io_tag}"
    return res


def build_binary(
    target: str,
    fft_x: int,
    fft_y: int,
    signal_x: int,
    signal_y: int,
    batch: int,
    op: str = "corr",
    arch: str = "8.9",
    swizzled: bool = False,
    ffts_per_block_y: int = 0,
    ffts_per_block_x: int = 0,
    elements_per_thread_x: int = 0,
    elements_per_thread_y: int = 0,
    nvcc: str | None = None,
    extra_include_dirs: list[str] | None = None,
    output_dir: Path = BUILD_DIR,
    extra_nvcc_args: list[str] | None = None,
    verbose: bool = True,
) -> Path:
    """Compiles one driver for one shape/IO-path combination.

    Returns
    -------
    Path to the compiled binary.

    Raises
    ------
    ValueError with explanation if the shape/IO-path combination is invalid, or if the
    target has an nvcc error.
    """
    if target not in DRIVER_FILES:
        raise ValueError(f"target must be one of {list(DRIVER_FILES)}, got {target!r}")
    if swizzled and (ffts_per_block_y < 2 or ffts_per_block_y % 2 != 0):
        raise ValueError(
            "swizzled=True requires ffts_per_block_y to be an even number >= 2 "
            f"(got {ffts_per_block_y}) -- see real_conv_2d_io.hpp for why."
        )

    nvcc_path = resolve_nvcc(nvcc)
    arch_int = arch.replace(".", "") + "0"  # "8.9" -> "890"

    include_dirs = [str(REPO_ROOT / "src" / "include"), str(REPO_ROOT / "src" / "cuda")]
    mathdx_dir = get_mathdx_include_dir()
    if mathdx_dir:
        include_dirs.append(mathdx_dir)
    include_dirs.extend(d for d in (extra_include_dirs or []) if d)
    include_dirs.extend(
        d for d in os.environ.get("EXTRA_INCLUDE_DIRS", "").split(os.pathsep) if d
    )

    binary_name = unique_binary_name(
        target, op, fft_x, fft_y, signal_x, signal_y, batch, swizzled, ffts_per_block_y
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    binary_path = output_dir / binary_name

    cross_correlate = 1 if op == "corr" else 0
    cmd = [
        nvcc_path,
        "-o",
        str(binary_path),
        str(DRIVER_FILES[target]),
        "-std=c++17",
        "-O3",
        f"-arch=sm_{arch_int[:-1]}",
        f"-DENABLE_CUDA_ARCH_{arch_int}",
        # Avoids needing cuFFTDx's bundled CUTLASS headers on the include path -- matches
        # setup.py's production build flags.
        "-DCUFFTDX_DISABLE_CUTLASS_DEPENDENCY",
        f"-DFFT_SIZE_X={fft_x}",
        f"-DFFT_SIZE_Y={fft_y}",
        f"-DSIGNAL_X={signal_x}",
        f"-DSIGNAL_Y={signal_y}",
        f"-DBATCH_SIZE={batch}",
        f"-DCROSS_CORRELATE={cross_correlate}",
        f"-DUSE_TILED_SWIZZLED_IO={1 if swizzled else 0}",
        f"-DFFTS_PER_BLOCK_X={ffts_per_block_x}",
        f"-DFFTS_PER_BLOCK_Y={ffts_per_block_y}",
        f"-DELEMENTS_PER_THREAD_X={elements_per_thread_x}",
        f"-DELEMENTS_PER_THREAD_Y={elements_per_thread_y}",
    ]
    for d in include_dirs:
        cmd.extend(["-I", d])
    cmd.extend(extra_nvcc_args or [])

    if verbose:
        print(f"[build] {' '.join(cmd)}")
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"nvcc failed for {binary_name}:\n{proc.stderr}")
    if verbose and proc.stderr:
        print(proc.stderr, file=sys.stderr)  # warnings

    return binary_path


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--target", choices=["real", "complex"], required=True)
    p.add_argument("--fft-x", type=int, required=True)
    p.add_argument("--fft-y", type=int, required=True)
    p.add_argument("--signal-x", type=int, required=True)
    p.add_argument("--signal-y", type=int, required=True)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--op", choices=["corr", "conv"], default="corr")
    p.add_argument("--arch", default="8.9", help='CUDA SM arch, e.g. "8.9"')
    p.add_argument(
        "--swizzled", action="store_true", help="Enable tiled+swizzled Y-dimension IO."
    )
    p.add_argument("--ffts-per-block-y", type=int, default=0)
    p.add_argument("--ffts-per-block-x", type=int, default=0)
    p.add_argument("--elements-per-thread-x", type=int, default=0)
    p.add_argument("--elements-per-thread-y", type=int, default=0)
    p.add_argument("--nvcc", default=None, help="Path to nvcc; default: auto-detect.")
    p.add_argument(
        "--extra-include-dirs",
        default="",
        help=f"{os.pathsep}-separated extra -I dirs.",
    )
    p.add_argument("--output-dir", default=str(BUILD_DIR))
    return p.parse_args()


def main():
    args = parse_args()
    binary_path = build_binary(
        target=args.target,
        fft_x=args.fft_x,
        fft_y=args.fft_y,
        signal_x=args.signal_x,
        signal_y=args.signal_y,
        batch=args.batch,
        op=args.op,
        arch=args.arch,
        swizzled=args.swizzled,
        ffts_per_block_y=args.ffts_per_block_y,
        ffts_per_block_x=args.ffts_per_block_x,
        elements_per_thread_x=args.elements_per_thread_x,
        elements_per_thread_y=args.elements_per_thread_y,
        nvcc=args.nvcc,
        extra_include_dirs=[d for d in args.extra_include_dirs.split(os.pathsep) if d],
        output_dir=Path(args.output_dir),
    )
    print(binary_path)


if __name__ == "__main__":
    main()

"""Setup script for binding C++/CUDA code with Python using pybind11."""

from setuptools import setup, Extension
import pybind11
import argparse
import sys
import os
import shutil
import tomllib

# setuptools.build_meta's default backend execs this file without putting its own directory
# on sys.path (unlike a plain `python setup.py` invocation), so `import configs_schema` below
# would fail under `pip install` despite working when this script is run directly.
# See https://github.com/pypa/setuptools/issues/1642.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Auto-detect CUDA_HOME before importing torch so that cpp_extension.CUDA_HOME
# is initialized correctly. In conda build environments the host PyTorch may be
# a CPU-only build (torch.cuda._is_compiled() == False), which causes
# cpp_extension.CUDA_HOME to be forced to None regardless of env vars.
# Detecting nvcc here and patching after import works around this.
if not os.environ.get("CUDA_HOME") and not os.environ.get("CUDA_PATH"):
    _nvcc = shutil.which("nvcc")
    if _nvcc:
        os.environ["CUDA_HOME"] = os.path.dirname(os.path.dirname(_nvcc))

import torch
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import torch.utils.cpp_extension as _cpp_ext

# If the host PyTorch is a CPU-only build, cpp_extension.CUDA_HOME is None
# even when CUDA is available. Patch it so CUDAExtension() can construct the
# extension object during pip's metadata-generation phase.
if _cpp_ext.CUDA_HOME is None and os.environ.get("CUDA_HOME"):
    _cpp_ext.CUDA_HOME = os.environ["CUDA_HOME"]

# pyproject.toml is the single source of truth for the package version.
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "pyproject.toml"), "rb") as f:
    __version__ = tomllib.load(f)["project"]["version"]


# Parse command line arguments for CUDA architectures
def parse_cuda_architectures():
    """Parse CUDA architectures from command line arguments or environment variables."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--cuda-arch",
        "--cuda-architectures",
        dest="cuda_architectures",
        help='Comma-separated list of CUDA arch to compile for (e.g., "7.5,8.0,8.6")',
        default=None,  # Will use env var or fallback if None
    )
    parser.add_argument(
        "--enable-extensions",
        dest="enable_extensions",
        help='Comma-separated list of extensions to build (e.g., "padded_rconv2d")',
        default=None,  # Will use env var or fallback if None
    )

    # Parse known args to avoid conflicts with setuptools
    args, unknown = parser.parse_known_args()

    # Remove our custom args from sys.argv so setuptools doesn't see them
    for arg_name in ["--cuda-arch", "--cuda-architectures", "--enable-extensions"]:
        if arg_name in sys.argv:
            idx = sys.argv.index(arg_name)
            sys.argv.pop(idx)  # Remove the argument
            # Remove value for non-flag arguments
            if idx < len(sys.argv):
                sys.argv.pop(idx)

    return args


# Parse arguments
parsed_args = parse_cuda_architectures()

# Get CUDA architectures with precedence: CLI args > env vars > defaults
cuda_archs_str = (
    parsed_args.cuda_architectures
    or os.environ.get("CUDA_ARCHITECTURES")
    or "8.0,8.6,8.9,9.0,10.0,10.3,12.0"
)
cuda_architectures = [arch.strip() for arch in cuda_archs_str.split(",")]

# Get enabled extensions with precedence: CLI args > env vars > defaults
enabled_exts_str = (
    parsed_args.enable_extensions
    or os.environ.get("ENABLED_EXTENSIONS")
    or "padded_rconv2d,padded_cconv2d"
)
enabled_extensions = [ext.strip() for ext in enabled_exts_str.split(",")]


# fmt: off
DEBUG_PRINT = False
if DEBUG_PRINT:
    print("Using pybind11 include directory: ", pybind11.get_include())
    print("Using torch include directory:    ", pybind11.get_include(user=True))
    print("Using torch library directory:    ", pybind11.get_cmake_dir())
    print("Using library dirs:               ", torch.utils.cpp_extension.CUDA_HOME)
    print("                                  ", torch.utils.cpp_extension.TORCH_LIB_PATH)
    print("CUDA architectures to compile:    ", cuda_architectures)
# fmt: on


def get_mathdx_include_dir():
    """Locate cuFFTDx/MathDx headers bundled in `nvidia-mathdx` pkg, if installed."""
    try:
        import nvidia.mathdx
    except ImportError:
        return None

    for search_path in nvidia.mathdx.__path__:
        include_dir = os.path.join(search_path, "include")
        if os.path.isdir(include_dir):
            return include_dir
    return None


def get_extra_include_dirs():
    """Collect additional include paths.

    - explicit EXTRA_INCLUDE_DIRS (e.g. conda build.sh pointing at conda-forge's mathdx
      package)
    - an auto-detected `nvidia-mathdx` pip install.
    """
    raw = os.environ.get("EXTRA_INCLUDE_DIRS", "")
    dirs = [d for d in raw.split(os.pathsep) if d]

    mathdx_include_dir = get_mathdx_include_dir()
    if mathdx_include_dir and mathdx_include_dir not in dirs:
        dirs.append(mathdx_include_dir)

    return dirs


def get_compile_args():
    """Generate compile arguments including CUDA architectures."""
    nvcc_args = [
        "-O3",
        # PyTorch >= ~2.9 requires C++20 (its headers #error out under C++17).
        "-std=c++20",
        # NOTE: Necessary to un-define PyTorch default macros with fp16/bf16 to get
        # cuFFTDx library to compile correctly.
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "-U__CUDA_NO_HALF2_OPERATORS__",
        # Defining macro to disable CUTLASS dependencies in cuFFTDx
        "-DCUFFTDX_DISABLE_CUTLASS_DEPENDENCY",
    ]

    # Add preprocessor definitions for enabled CUDA architectures
    arch_defines = []
    for arch in cuda_architectures:
        # Convert decimal arch to what cuFFTDx expects
        # e.g. 8.9 -> 89 -> 890
        # e.g. 12.0 -> 120 -> 1200
        arch_int = arch.replace(".", "")
        arch_int = arch_int + "0"

        arch_defines.append(f"-DENABLE_CUDA_ARCH_{arch_int}")

    nvcc_args.extend(arch_defines)

    # Add architecture-specific flags
    for arch in cuda_architectures:
        nvcc_args.extend(
            [
                "-gencode",
                f"arch=compute_{arch.replace('.', '')},code=sm_{arch.replace('.', '')}",
            ]
        )

    return {
        "cxx": ["-O3", "-std=c++20"],
        "nvcc": nvcc_args,
    }


def get_torch_library_path():
    """Get the path to PyTorch libraries."""
    import torch

    torch_path = os.path.dirname(torch.__file__)
    return os.path.join(torch_path, "lib")


GENERATED_CUDA_DIR = "src/cuda/generated"


def _write_if_changed(path: str, content: str) -> bool:
    """Write content to path only if current file differs from desired content."""
    if os.path.exists(path):
        with open(path) as f:
            if f.read() == content:
                return False
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        f.write(content)
    return True


# (yaml_key, C++ array name, generated header path)
CONFIG_CODEGEN_TARGETS = [
    (
        "real_conv2d",
        "SUPPORTED_CONV_CONFIGS",
        "src/cuda/generated_real_conv_2d_configs.hpp",
    ),
    (
        "complex_conv2d",
        "SUPPORTED_C2C_CONV_CONFIGS",
        "src/cuda/generated_complex_conv_2d_configs.hpp",
    ),
]


def _entry_to_cpp_tuple(entry: dict) -> str:
    """Formats one shape entry as a C++ brace-init tuple literal, e.g.
    '{48, 48, 64, 64, 1, false, false, 0},'. Field order must match the std::tuple type
    in _render_config_header.
    """
    cross_correlate = "true" if entry["cross_correlate"] else "false"
    use_tiled_swizzled_io = (
        "true" if entry.get("use_tiled_swizzled_io", False) else "false"
    )
    ffts_per_block_y = entry.get("ffts_per_block_y", 0)
    return (
        f"        {{{entry['signal_y']}, {entry['signal_x']}, {entry['fft_y']}, "
        f"{entry['fft_x']}, {entry['batch']}, {cross_correlate}, "
        f"{use_tiled_swizzled_io}, {ffts_per_block_y}}},"
    )


def _render_config_header(array_name: str, entries: list[dict]) -> str:
    """Renders the full contents of one generated_*_configs.hpp file: a std::array of
    std::tuple, one tuple per entry."""
    lines = [
        "// Auto-generated from configs.yaml by setup.py -- do not edit directly.",
        "// Add/remove shapes in configs.yaml and rebuild instead.",
        "#pragma once",
        "",
        "#include <array>",
        "#include <tuple>",
        "",
        "// (signal_length_y, signal_length_x, fft_size_y, fft_size_x, batch_size, cross_correlate,",
        "//  use_tiled_swizzled_io, ffts_per_block_y)",
        "static constexpr std::array<",
        "    std::tuple<unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, bool,",
        "               bool, unsigned int>,",
        f"    {len(entries)}>",
        f"    {array_name} = {{{{",
        *(_entry_to_cpp_tuple(entry) for entry in entries),
        "    }};",
        "",
    ]
    return "\n".join(lines)


def generate_config_headers(yaml_path="configs.yaml"):
    """Renders configs.yaml into the C++ config-array headers included by the real/complex
    conv binding files. This lets new (signal, fft, batch) shapes be added by editing YAML
    instead of hand-writing C++ template instantiations -- see configs.yaml for the schema.

    Returns {yaml_key: expanded_entries} so callers (e.g. generate_conv_2d_shards) can
    reuse the same expanded list instead of re-parsing configs.yaml.
    """
    import yaml

    from configs_schema import expand_configs

    with open(yaml_path) as f:
        configs = yaml.safe_load(f)

    expanded_by_key = {}
    for yaml_key, array_name, out_path in CONFIG_CODEGEN_TARGETS:
        entries = expand_configs(configs[yaml_key])
        expanded_by_key[yaml_key] = entries
        _write_if_changed(out_path, _render_config_header(array_name, entries))

    return expanded_by_key


def _conv_dispatch_template_args(entry: dict) -> str:
    """(SignalLengthX, SignalLengthY, FFTSizeX, FFTSizeY, BatchSize, CrossCorrelate, UseTiledSwizzledIO, FFTsPerBlockY)"""
    cross_correlate = "true" if entry["cross_correlate"] else "false"
    use_tiled_swizzled_io = (
        "true" if entry.get("use_tiled_swizzled_io", False) else "false"
    )
    ffts_per_block_y = entry.get("ffts_per_block_y", 0)
    return (
        f"{entry['signal_x']}, {entry['signal_y']}, {entry['fft_x']}, {entry['fft_y']}, "
        f"{entry['batch']}, {cross_correlate}, {use_tiled_swizzled_io}, {ffts_per_block_y}"
    )


def _conv_shard_key(entry: dict) -> str:
    """Groups configs into shard files by (fft_y, fft_x, cross_correlate) family."""
    variant = "corr" if entry["cross_correlate"] else "conv"
    return f"{entry['fft_y']}x{entry['fft_x']}_{variant}"


# (yaml_key, shard filename prefix, dispatch-impl header, dispatch function, signature)
CONV_SHARD_TARGETS = {
    "real_conv2d": (
        "real_conv_2d_shard_",
        "real_conv_2d_dispatch_impl.cuh",
        "dispatch_padded_real_conv",
        "(float*, float2*, const float2*, float*, int, cudaStream_t)",
    ),
    "complex_conv2d": (
        "complex_conv_2d_shard_",
        "complex_conv_2d_dispatch_impl.cuh",
        "dispatch_padded_complex_conv",
        "(float2*, float2*, const float2*, float2*, int, cudaStream_t)",
    ),
}


def generate_conv_2d_shards(yaml_key: str, entries: list[dict]) -> list[str]:
    """Generates one .cu file per (fft_y, fft_x, cross_correlate) family for the given
    configs.yaml section, so each family compiles as its own TU in parallel."""
    shard_prefix, impl_header, dispatch_fn, signature = CONV_SHARD_TARGETS[yaml_key]
    shards: dict[str, list[dict]] = {}
    seen_tuples: dict[tuple, dict] = {}
    for entry in entries:
        key = (
            entry["signal_y"],
            entry["signal_x"],
            entry["fft_y"],
            entry["fft_x"],
            entry["batch"],
            entry["cross_correlate"],
        )
        if key in seen_tuples:
            raise ValueError(
                f"Duplicate (signal_y, signal_x, fft_y, fft_x, batch, cross_correlate) "
                f"config in configs.yaml's {yaml_key} section: {key} appears in both "
                f"{seen_tuples[key]} and {entry}. Each combination must be unique -- "
                "it would otherwise produce two explicit instantiations of the same "
                f"{dispatch_fn}<...> specialization (a link error)."
            )
        seen_tuples[key] = entry
        shards.setdefault(_conv_shard_key(entry), []).append(entry)

    generated_paths = []
    for key in sorted(shards):
        shard_entries = shards[key]
        out_path = f"{GENERATED_CUDA_DIR}/{shard_prefix}{key}.cu"
        lines = [
            "// Auto-generated from configs.yaml by setup.py -- do not edit directly.",
            f"// Explicit instantiations of {dispatch_fn} for the {key} "
            f"family ({len(shard_entries)} configs). Add/remove shapes in configs.yaml "
            "and rebuild instead.",
            f'#include "../{impl_header}"',
            "",
        ]
        for entry in shard_entries:
            lines.append(
                f"template void {dispatch_fn}<"
                f"{_conv_dispatch_template_args(entry)}>{signature};"
            )
        lines.append("")
        _write_if_changed(out_path, "\n".join(lines))
        generated_paths.append(out_path)

    if os.path.isdir(GENERATED_CUDA_DIR):
        expected = {os.path.basename(p) for p in generated_paths}
        for fname in os.listdir(GENERATED_CUDA_DIR):
            if fname.startswith(shard_prefix) and fname not in expected:
                os.remove(os.path.join(GENERATED_CUDA_DIR, fname))

    return generated_paths


expanded_configs_by_key = generate_config_headers()
real_conv_2d_shard_sources = generate_conv_2d_shards(
    "real_conv2d", expanded_configs_by_key["real_conv2d"]
)
complex_conv_2d_shard_sources = generate_conv_2d_shards(
    "complex_conv2d", expanded_configs_by_key["complex_conv2d"]
)

DEFAULT_COMPILE_ARGS = get_compile_args()

# Get PyTorch library directory
TORCH_LIB_DIR = get_torch_library_path()

# Conditionally create extensions
ext_modules = []

if "padded_rconv2d" in enabled_extensions:
    padded_real_conv_2d_extension = CUDAExtension(
        name="zipfft.padded_rconv2d",
        sources=["src/cuda/real_conv_2d_binding.cu"] + real_conv_2d_shard_sources,
        include_dirs=[pybind11.get_include()] + get_extra_include_dirs(),
        library_dirs=[TORCH_LIB_DIR],
        libraries=[
            "c10",
            "torch_cpu",
            "torch_python",
            "c10_cuda",
        ],
        runtime_library_dirs=[TORCH_LIB_DIR],
        extra_compile_args=DEFAULT_COMPILE_ARGS,
    )
    ext_modules.append(padded_real_conv_2d_extension)

if "padded_cconv2d" in enabled_extensions:
    padded_complex_conv_2d_extension = CUDAExtension(
        name="zipfft.padded_cconv2d",
        sources=["src/cuda/complex_conv_2d_binding.cu"] + complex_conv_2d_shard_sources,
        include_dirs=[pybind11.get_include()] + get_extra_include_dirs(),
        library_dirs=[TORCH_LIB_DIR],
        libraries=[
            "c10",
            "torch_cpu",
            "torch_python",
            "c10_cuda",
        ],
        runtime_library_dirs=[TORCH_LIB_DIR],
        extra_compile_args=DEFAULT_COMPILE_ARGS,
    )
    ext_modules.append(padded_complex_conv_2d_extension)

# Write build configuration to a file for testing
build_config = {
    "cuda_architectures": cuda_architectures,
    "enabled_extensions": enabled_extensions,
}

os.makedirs("src/zipfft", exist_ok=True)
with open("src/zipfft/build_config.py", "w") as f:
    f.write(f"# Auto-generated build configuration\n")
    f.write(f"CUDA_ARCHITECTURES = {cuda_architectures}\n")
    f.write(f"ENABLED_EXTENSIONS = {enabled_extensions}\n")
    f.write(f"VERSION = {__version__!r}\n")

# TODO: Make this setup script more robust (plus conda recipe)
# name/description/author/version/python_requires are declared statically in
# pyproject.toml's [project] table, which setuptools reads automatically.
setup(
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
)

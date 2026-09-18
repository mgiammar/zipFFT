"""configs.yaml has two entry styles at top-level section (real_conv2d, complex_conv2d):
  - test_configs: explicit one-off shape dicts, used as-is.
  - production_matrix: compact entries (lists of signal/fft shapes and batches) which
    expand into the same shape dicts via a cartesian product.
`expand_configs()` turns either style into the flat, validated list of shape dicts that
setup.py's codegen and benchmark/sweep.py both consume.
"""

import itertools

_REQUIRED_MATRIX_KEYS = ("signal_shapes", "fft_shapes", "batches")


def expand_configs(section) -> list[dict]:
    """Accepts one configs.yaml, either legacy flat or new matrix style."""
    if isinstance(section, list):
        entries = section
    else:
        entries = list(section.get("test_configs", []))
        for matrix_entry in section.get("production_matrix", []):
            entries.extend(_expand_matrix_entry(matrix_entry))

    for entry in entries:
        _validate_entry(entry)
    return entries


def _expand_matrix_entry(matrix_entry: dict) -> list[dict]:
    missing_keys = [k for k in _REQUIRED_MATRIX_KEYS if k not in matrix_entry]
    if missing_keys:
        raise ValueError(
            f"production_matrix entry is missing {missing_keys}: {matrix_entry}"
        )

    cross_correlate_options = matrix_entry["cross_correlate"]
    if not isinstance(cross_correlate_options, list):
        cross_correlate_options = [cross_correlate_options]

    shared_fields = {
        "use_tiled_swizzled_io": matrix_entry.get("use_tiled_swizzled_io", False),
        "ffts_per_block_y": matrix_entry.get("ffts_per_block_y", 0),
    }
    combinations = itertools.product(
        matrix_entry["signal_shapes"],
        matrix_entry["fft_shapes"],
        matrix_entry["batches"],
        cross_correlate_options,
    )
    return [
        {
            "signal_y": signal_y,
            "signal_x": signal_x,
            "fft_y": fft_y,
            "fft_x": fft_x,
            "batch": batch,
            "cross_correlate": cross_correlate,
            **shared_fields,
        }
        for (signal_y, signal_x), (fft_y, fft_x), batch, cross_correlate in combinations
    ]


def _validate_entry(entry: dict) -> None:
    swizzled = entry.get("use_tiled_swizzled_io", False)
    ffts_per_block_y = entry.get("ffts_per_block_y", 0)
    ffts_per_block_y_ok = ffts_per_block_y >= 2 and ffts_per_block_y % 2 == 0
    if swizzled and not ffts_per_block_y_ok:
        raise ValueError(
            f"{entry}: use_tiled_swizzled_io requires an even ffts_per_block_y >= 2 "
            "(see real_conv_2d_io.hpp for why)."
        )

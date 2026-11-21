"""只测管线峰值内存的子进程入口：不保留任何额外数组。

用法： python -I scripts/measure_pipeline_only.py <输入目录> <输出目录> <饱和阈值>
"""

from __future__ import annotations

import os
import resource
import sys
import time
from pathlib import Path

if os.environ.get("FLAT_TEST_SITE"):
    import site

    site.addsitedir(os.environ["FLAT_TEST_SITE"])

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from app.pipeline import build_master_flat  # noqa: E402


def main() -> int:
    input_dir, out_dir, sat = (
        Path(sys.argv[1]),
        Path(sys.argv[2]),
        float(sys.argv[3]),
    )
    paths = sorted(input_dir.glob("*.fits"))
    io_stats: dict[str, int] = {}
    t0 = time.perf_counter()
    product = build_master_flat(
        paths, [p.name for p in paths], sat,
        output_dir=out_dir, io_stats=io_stats,
    )
    elapsed = time.perf_counter() - t0
    peak_mib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    ratio = io_stats["max_pixel_bytes_read"] / io_stats["pixel_bytes_per_frame"]
    print(
        f"RESULT frames={len(paths)} elapsed={elapsed:.1f}s peak_mib={peak_mib:.1f} "
        f"read_multiples={ratio:.2f} product={product}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

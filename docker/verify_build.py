#!/usr/bin/env python3
"""
Build-time verification for the Jetson image.

Runs inside the container at build time, where **no GPU is visible**. So we
cannot execute a CUDA kernel; instead we assert that

  1. every native extension imports (they link against the right torch/CUDA), and
  2. pytorch3d actually carries CUDA rasterization symbols.

(2) is the important one. A CPU-only pytorch3d compiles, installs and imports
perfectly happily, then dies at *runtime* with

    RuntimeError: Not compiled with GPU support

inside object_renderer -> rasterize_points, taking the whole 2D branch (the most
expensive ~72% of the forward) down with it. That is exactly the failure this
image exists to prevent, so we fail the build here instead of shipping it.

Also runnable on a real Jetson for a full check including a live CUDA rasterize:
    python3 docker/verify_build.py
"""

import subprocess
import sys


def fail(msg):
    print(f"FAIL: {msg}")
    sys.exit(1)


def main():
    # ── 1. every native extension must import ────────────────────────────────
    import torch
    import MinkowskiEngine
    import clip  # noqa: F401
    import pytorch3d
    import hydra  # noqa: F401
    import lightning  # noqa: F401
    import COMMON_OPS  # noqa: F401
    from pytorch3d import _C

    print(f"torch       {torch.__version__}  (cuda build: {torch.version.cuda})")
    print(f"minkowski   {MinkowskiEngine.__version__}")
    print(f"pytorch3d   {pytorch3d.__version__}")
    print("COMMON_OPS  ok")

    # ── 2. pytorch3d must be a CUDA build ────────────────────────────────────
    so = _C.__file__
    try:
        syms = subprocess.run(["nm", "-D", "--defined-only", so],
                              capture_output=True, text=True, timeout=60).stdout
    except (FileNotFoundError, subprocess.TimeoutExpired):
        syms = ""

    cuda_syms = [ln for ln in syms.splitlines() if "RasterizePoints" in ln and "Cuda" in ln]

    if syms and not cuda_syms:
        fail("pytorch3d has NO CUDA rasterization symbols -> CPU-only build.\n"
             "      object_renderer would die at runtime with "
             "'Not compiled with GPU support'.\n"
             "      Ensure FORCE_CUDA=1 and TORCH_CUDA_ARCH_LIST are set when "
             "building pytorch3d.")

    if cuda_syms:
        print(f"pytorch3d CUDA rasterization symbols: {len(cuda_syms)} (good)")
    else:
        # nm unavailable: fall back to the linkage check rather than pass blindly
        try:
            ldd = subprocess.run(["ldd", so], capture_output=True, text=True,
                                 timeout=60).stdout
            if "cudart" not in ldd:
                fail("pytorch3d is not linked against libcudart -> CPU-only build.")
            print("pytorch3d linked against libcudart (good)")
        except (FileNotFoundError, subprocess.TimeoutExpired):
            print("WARN: could not inspect pytorch3d binary (no nm/ldd); "
                  "CUDA support unverified")

    # ── 3. if a GPU happens to be present, prove rasterization really runs ────
    if torch.cuda.is_available():
        from pytorch3d.renderer.points.rasterize_points import rasterize_points
        from pytorch3d.structures import Pointclouds
        pcd = Pointclouds(points=[torch.zeros(1, 3, device="cuda")])
        try:
            rasterize_points(pcd, image_size=8, radius=0.1, points_per_pixel=1)
            print("live CUDA rasterize: OK")
        except RuntimeError as e:
            fail(f"live CUDA rasterize failed: {e}")
    else:
        print("no GPU at build time -> skipped live CUDA rasterize (expected)")

    print("ALL IMPORTS OK")


if __name__ == "__main__":
    main()

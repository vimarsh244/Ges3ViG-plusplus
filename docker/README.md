# Jetson Orin container for Ges3ViG

Builds the whole native stack the model needs - PyTorch(+CUDA), **pytorch3d
built with CUDA**, MinkowskiEngine, CLIP, open3d, and the ges3vig custom CUDA
ops - on top of NVIDIA's JetPack 6 / L4T r36.4 PyTorch base
(`dustynv/pytorch:2.6-r36.4.0-cu128`, CUDA 12.8, Python 3.10).

Target: **Jetson Orin Nano** (Ampere, `TORCH_CUDA_ARCH_LIST=8.7`).

## Build

```bash
# on the Jetson (native, recommended):
./docker/build_and_test_jetson.sh

# native build + run the profiler:
./docker/build_and_test_jetson.sh --test

# on an x86 dev box (cross-build via QEMU - validates the Dockerfile, slow):
./docker/build_and_test_jetson.sh --cross

# just validate the Dockerfile, no compiling:
./docker/build_and_test_jetson.sh --lint-only
```

The build context is the **repo root**, not `docker/`. The script handles that;
if you invoke docker yourself:

```bash
docker buildx build -f docker/Dockerfile.jetson --platform linux/arm64 -t ges3vig-jetson .
```

## Profile

```bash
docker run --rm --runtime nvidia \
  -v "$PWD/profiling/results:/workspace/ges3vig/profiling/results" \
  ges3vig-jetson:latest \
  python3 profiling/profile_jetson.py --mode modules --batch-sizes 1 4 \
    --repeats 20 --csv profiling/results/jetson_modules.csv
```

See [`../profiling/README.md`](../profiling/README.md) for the modes and flags.

## Why the Dockerfile looks the way it does

Each of these is load-bearing; changing them tends to break the build in a
non-obvious way.

* **pip index scrubbing.** The dustynv base bakes `pypi.jetson-ai-lab.dev` (a
  dead domain) into several `pip.conf` files *and* into `PIP_INDEX_URL`. Env
  vars beat config files, so we delete every pip config **and** re-export the
  vars to the live `.io` mirror. Otherwise every `pip install` hangs/fails.
* **`FORCE_CUDA=1` for pytorch3d.** Without it you silently get a CPU-only
  build, and the 2D object renderer dies at runtime with
  `RuntimeError: Not compiled with GPU support`. This is not hypothetical - the
  x86 conda env on the dev box has exactly that broken build.
* **`--no-build-isolation` for pytorch3d.** Its `setup.py` imports torch at
  configure time; an isolated build env has no torch → `ModuleNotFoundError`.
* **`--no-deps` for CLIP / torchvision.** Both would otherwise pull a generic
  PyPI torch wheel and clobber the JetPack-tuned torch from the base image.
* **The MinkowskiEngine `sed`.** Patches `std::__to_address` in libstdc++ 11's
  `shared_ptr_base.h`; without it the ME nvcc compile fails.
* **`blinker` removal.** It is distutils-installed in the base image and pip
  cannot uninstall it, so `lightning`'s dependency resolution breaks.
* **`TORCH_CUDA_ARCH_LIST=8.7`.** No GPU is visible at build time, so the CUDA
  extensions (ME + `common_ops`) must be told the Orin arch explicitly.

## open3d

open3d has no aarch64 PyPI wheel. Drop a **cp310** aarch64 wheel into
`docker/wheels/` and the build uses it; the build script auto-stages the copy
already in this checkout (`../Ges3ViG/open3d-0.19.0-cp310-...whl`).

No wheel is fine: the build falls back to the Jetson index and, failing that,
continues without open3d - it's only needed for the real dataloader, not for the
random-tensor profiler. See [`wheels/README.md`](wheels/README.md).

## CI

[`.github/workflows/jetson-docker.yml`](../.github/workflows/jetson-docker.yml)

* **validate** (every push/PR touching docker/profiling/ges3vig): lints the
  Dockerfile (`buildx --check` + hadolint), checks every `COPY` source survives
  `.dockerignore`, compiles the profiler, shellchecks the build script. Fast, no
  compiling.
* **build-arm64** (manual `workflow_dispatch` with `full_build`): emulated on a hosted runner 
* **build-native-jetson** (manual, needs a self-hosted Jetson runner labelled
  `[self-hosted, linux, ARM64, jetson]`): native build, runs the profiler, and
  uploads the timings as an artifact.
  [NOT SUPPORTED FOR NOW]

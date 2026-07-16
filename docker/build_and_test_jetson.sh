#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Build (and optionally smoke-test) the Ges3ViG Jetson Orin image.
#
# The image is arm64. On an arm64 Jetson it builds natively. On an x86_64 dev
# box it cross-builds through QEMU emulation, which WORKS but is slow (the
# MinkowskiEngine + pytorch3d CUDA compiles take a long while under emulation)
# — that path is for validating the Dockerfile, not for producing timings.
#
# Usage:
#   ./docker/build_and_test_jetson.sh                 # build for the host arch
#   ./docker/build_and_test_jetson.sh --cross          # force arm64 via QEMU
#   ./docker/build_and_test_jetson.sh --test           # build + run profiler
#   ./docker/build_and_test_jetson.sh --lint-only      # validate Dockerfile only
#   IMAGE=my/tag:v1 ./docker/build_and_test_jetson.sh
#
#   # write the finished image to a tarball instead of into the local daemon.
#   # Handy when the docker store lives on a small disk but you have a big
#   # NAS/scratch elsewhere. Load it later with `docker load -i <tar>`, or push.
#   OUTPUT_TAR=/nas_hcfc/vimarsh/HRI/docker_out/ges3vig-jetson-arm64.tar \
#     ./docker/build_and_test_jetson.sh --cross
#
# NOTE: only the *result* can go elsewhere. BuildKit's own layer/cache store
# follows the docker daemon's data-root, and overlay2 cannot be backed by NFS,
# so the build itself always works on the local disk.
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
IMAGE="${IMAGE:-ges3vig-jetson:latest}"
DOCKERFILE="$SCRIPT_DIR/Dockerfile.jetson"

CROSS=0
RUN_TEST=0
LINT_ONLY=0
for arg in "$@"; do
  case "$arg" in
    --cross)     CROSS=1 ;;
    --test)      RUN_TEST=1 ;;
    --lint-only) LINT_ONLY=1 ;;
    -h|--help)   sed -n '2,20p' "$0"; exit 0 ;;
    *) echo "unknown flag: $arg" >&2; exit 2 ;;
  esac
done

echo "== repo   : $REPO_ROOT"
echo "== image  : $IMAGE"
echo "== host   : $(uname -m)"

# ── stage the open3d wheel into the build context if we can find one ─────────
# (see docker/wheels/README.md — no wheel is fine, the build falls back.)
shopt -s nullglob
existing=("$SCRIPT_DIR"/wheels/open3d-*cp310*aarch64.whl)
if [ ${#existing[@]} -eq 0 ]; then
  candidates=("$REPO_ROOT"/../Ges3ViG/open3d-*cp310*aarch64.whl)
  if [ ${#candidates[@]} -gt 0 ]; then
    echo "== staging open3d wheel: $(basename "${candidates[0]}")"
    cp "${candidates[0]}" "$SCRIPT_DIR/wheels/"
  else
    echo "== no open3d cp310 aarch64 wheel found; build will fall back (non-fatal)"
  fi
else
  echo "== open3d wheel already staged: $(basename "${existing[0]}")"
fi
shopt -u nullglob

# ── pick platform ────────────────────────────────────────────────────────────
# Must happen BEFORE the --check below: checking an arm64 base without
# --platform raises a bogus InvalidBaseImagePlatform warning.
PLATFORM_ARGS=()
if [ "$CROSS" -eq 1 ] || [ "$(uname -m)" != "aarch64" ]; then
  echo "== cross-building linux/arm64 via QEMU (slow; validates the Dockerfile)"
  docker run --privileged --rm tonistiigi/binfmt --install arm64 >/dev/null 2>&1 || true
  PLATFORM_ARGS=(--platform linux/arm64)
fi

# ── Dockerfile syntax/structure validation (no network, no compile) ──────────
# `--check` exits non-zero on *warnings* too, which would trip `set -e`, so it
# is advisory here: report it, but only abort for --lint-only.
echo "== validating Dockerfile syntax ..."
check_rc=0
docker buildx build --check "${PLATFORM_ARGS[@]}" -f "$DOCKERFILE" "$REPO_ROOT" || check_rc=$?
if [ "$check_rc" -eq 0 ]; then
  echo "== Dockerfile OK"
else
  echo "== Dockerfile check reported warnings (rc=$check_rc) - continuing"
fi

if [ "$LINT_ONLY" -eq 1 ]; then
  echo "== --lint-only: stopping after the check"
  exit "$check_rc"
fi

# a named builder gives us a persistent cache across runs
docker buildx inspect ges3vig-builder >/dev/null 2>&1 || \
  docker buildx create --name ges3vig-builder --driver docker-container >/dev/null

OUTPUT_ARGS=(--load)
if [ -n "${OUTPUT_TAR:-}" ]; then
  mkdir -p "$(dirname "$OUTPUT_TAR")"
  echo "== output   : $OUTPUT_TAR (tarball; not loaded into the local daemon)"
  OUTPUT_ARGS=(--output "type=docker,dest=$OUTPUT_TAR")
fi

echo "== building ..."
docker buildx build \
  --builder ges3vig-builder \
  "${PLATFORM_ARGS[@]}" \
  -f "$DOCKERFILE" \
  -t "$IMAGE" \
  "${OUTPUT_ARGS[@]}" \
  "$REPO_ROOT"

if [ -n "${OUTPUT_TAR:-}" ]; then
  echo "== built -> $OUTPUT_TAR  ($(du -h "$OUTPUT_TAR" | cut -f1))"
  echo "   load with: docker load -i $OUTPUT_TAR"
else
  echo "== built: $IMAGE"
fi

# ── smoke test ───────────────────────────────────────────────────────────────
if [ "$RUN_TEST" -eq 1 ]; then
  if [ "$(uname -m)" != "aarch64" ]; then
    echo "== skipping runtime test: not on a Jetson (emulated CUDA is unavailable)"
    exit 0
  fi
  # a tarball build never entered the daemon; load it before we can run it
  if [ -n "${OUTPUT_TAR:-}" ]; then
    echo "== loading $OUTPUT_TAR into the local daemon for the test run ..."
    docker load -i "$OUTPUT_TAR"
  fi
  echo "== running the profiler inside the container ..."
  docker run --rm --runtime nvidia --network host \
    -v "$REPO_ROOT/profiling/results:/workspace/ges3vig/profiling/results" \
    "$IMAGE" \
    python3 profiling/profile_jetson.py --mode modules \
      --batch-sizes 1 4 --repeats 20 \
      --csv profiling/results/jetson_modules.csv
fi

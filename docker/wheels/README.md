# Prebuilt wheels for the Jetson image

Drop aarch64 wheels here that cannot be resolved from an index at build time.

## open3d

open3d publishes no `aarch64` wheel on PyPI, so the Jetson build looks for:

    open3d-*-cp310-*-manylinux*_aarch64.whl

The base image (`dustynv/pytorch:2.6-r36.4.0-cu128`) is **Python 3.10**, so it
must be a **cp310** wheel - a cp312 wheel will be ignored.

A known-good copy already exists in this checkout:

    cp ../../Ges3ViG/open3d-0.19.0-cp310-cp310-manylinux_2_35_aarch64.whl ./

`docker/build_and_test_jetson.sh` does this automatically when it finds one.

If no wheel is present the build still succeeds: it falls back to the Jetson
package index and, failing that, continues **without** open3d. open3d is only
needed for the real dataloader - the random-tensor profiler does not import it.

Wheels are gitignored (they are ~48 MB); this README keeps the directory
present so the Dockerfile's `COPY docker/wheels/` always resolves.

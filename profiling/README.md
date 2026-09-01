# Profiling M3DRefCLIP on a Jetson Orin AGX

`profile_jetson.py` measures **per-layer / per-module forward latency** of the
published M3DRefCLIP model, driven by **random tensors** - no dataset, no checkpoint
required.
The model source and configs are vendored from `3dlg-hcvc/multi3drefer` at commit
`cd5c23e1a997648c4906883458d86d3632d4a470`; no upstream remote is stored. It warms up, times N passes, and reports
`mean / std / min / p50 / p90 / p95 / max` plus `fps`, for a single sample and
for a batch.

## The two default commands

```bash
# local dev box. NOTE: this must be the /home/vim conda, not the NAS one -
# see "Use the right env" below.
PY=/home/vim/miniconda3/envs/ges3vig-gpu/bin/python
```

**1. Full sweep - runs every case and stores one CSV.** This is the one to run.

```bash
$PY profiling/profile_jetson.py --mode sweep \
    --cases points dropout foveation batch \
    --point-sweep 10000 20000 30000 40000 50000 60000 70000 80000 90000 \
    --dropout-sweep 1.0 0.75 0.5 \
    --foveation-sweep 80 48 32 16 8 4 1 \
    --batch-sweep 1 2 4 \
    --repeats 15 --warmup 5 \
    --csv profiling/results/sweep.csv
```

Those are all the defaults, so the short form is equivalent:

```bash
$PY profiling/profile_jetson.py --mode sweep --csv profiling/results/sweep.csv
```

**2. Whole model, one config - per-stage + per-layer + TOTAL.**

```bash
$PY profiling/profile_jetson.py --mode full \
    --batch-sizes 1 --num-points 40000 \
    --repeats 15 --warmup 5 --top-layers 25 \
    --csv profiling/results/full.csv
```

On the Jetson, inside the container (see [`../docker/`](../docker/)) - same
commands, just prefixed:

```bash
docker run --rm --runtime nvidia \
  -v "$PWD/profiling/results:/workspace/ges3vig/profiling/results" \
  ges3vig-jetson:latest \
  python3 profiling/profile_jetson.py --mode sweep \
    --csv profiling/results/jetson_sweep.csv
```

## The three modes

| mode | what it measures | needs geometry? |
|---|---|---|
| `modules` | each module alone, fed random tensors | no |
| `full` | the whole pipeline: stages + layers + TOTAL | yes (synthesized) |
| `sweep` | **auto-runs every case** below, stores one CSV | yes (synthesized) |

### `--mode sweep` - run everything, store everything

```bash
$PY profiling/profile_jetson.py --mode sweep --csv profiling/results/sweep.csv
```

Runs these cases back-to-back (pick a subset with `--cases`):

| case | what it varies | default |
|---|---|---|
| `points` | point count | **10k→90k in 10k steps** (`--point-sweep`) |
| `dropout` | random point drop (keep-ratio) | `1.0 0.75 0.5` (`--dropout-sweep`) |
| `foveation` | #proposals reaching the 2D encoder + fusion | `80 48 32 16 8 4 1` (`--foveation-sweep`) |
| `batch` | batch size | `1 2 4` (`--batch-sweep`) |

Every measurement lands in one CSV with `case`, `param`, `num_points`,
`keep_ratio`, `batch_size`, `kind` (stage/total/module), the latency stats and
`fps`. Each case also prints a compact scan table.

A note on **foveation**: M3DRefCLIP has no foveation logic. This case does *not* foveate the model - it measures
the **cost curve**: what the 2D encoder + fusion would cost if only *k*
proposals were rendered. That tells you what foveation would buy before anyone
implements it.

### `--mode modules` (default) - always works

Micro-benchmarks each sub-module in isolation, feeding it **random tensors** of
the shapes the config implies. It needs no scene geometry, so it runs anywhere:

| module | what it is | random input |
|---|---|---|
| `clip_image` | 2D vision encoder (CLIP ViT-B/32) | `B×K×views` images of `3×224×224` |
| `text_encoder` | language encoder (CLIP text) | `B×chunk` sentences of 77 tokens |
| `match_module` | cross-modal fusion | `B×80` proposals + `77` word tokens |

Use `--num-proposals` to set how many proposals per scene the 2D and fusion
paths see (default 48; the model pads to `max_num_proposals=80`).

### `--mode full` - whole pipeline

Synthesizes a **random point cloud** (dense Gaussian blobs, so the ball-query /
BFS clustering actually forms proposals), voxelizes it *exactly* like the real
dataloader, and runs the entire model. Forward hooks time:

* every **stage** (`detector`, `object_renderer`, `clip_image`, `text_encoder`,
  `match_module`),
* the **TOTAL** forward,
* the **slowest leaf layers** (`--top-layers`, default 25) - individual
  Minkowski convs, linears, etc.

Whatever the stages don't account for is reported as *non-module ops*
(pooling and sparse→dense conversion).

To profile the *real* workload instead of a random one, pass a cached batch:

```bash
$PY profiling/profile_jetson.py --mode full --batch /tmp/one_scene.pt
```

(A cached batch is just a collated `data_dict` saved with `torch.save`; the
random scene is representative enough for latency, but proposal counts on a real
scene differ, and the detector's cost depends on them.)

## Useful flags

| flag | meaning |
|---|---|
| `--batch-sizes 1 4` | profile a single sample and a batch (`full` uses the first) |
| `--repeats` / `--warmup` | timed passes (default 30) / warmup passes (default 8) |
| `--ckpt PATH` | load real weights (latency is ~identical to random init) |
| `--csv PATH` | write every measurement to CSV |
| `--num-points` | synthetic points per scene in `full` mode (default 40000) |
| `--keep-ratio` | randomly drop points, keeping this fraction (1.0 = all) |
| `--cases` | which sweep cases to run (`points dropout foveation batch`) |
| `--cpu` | force CPU |
| `--fp16` | autocast fp16 (auto-disabled when MinkowskiEngine is present) |
| `--stub-renderer` | fake the rasterizer, keep `clip_image` real - **auto-enabled when pytorch3d lacks CUDA; see below** |
| `--skip-2d` | drop the 2D branch entirely (blunt; prefer `--stub-renderer`) |
| `--no-auto-stub` | don't auto-stub; let the renderer fail loudly |

## Reading the output

Every table has `mean std min p50 p90 p95 max fps`.

* **`fps`** = scenes/second = `1 / (per-scene time)` = `batch_size / mean_seconds`.
  For a batched row it is per-scene throughput, not the pass rate.
* **Prefer `p50` over `mean` on a busy box.** A single outlier pass (another job
  grabbing the GPU/CPU) drags the mean badly; `p50` is stable. If `std` is a
  large fraction of `mean`, the machine was contended - re-run when it's idle.

### Why `full` mode times in three separate passes

Forward hooks CUDA-synchronize around every module they wrap, which serializes
the GPU. Timing the total *and* every leaf layer in one pass therefore inflates
both - measured at ~15% on the TOTAL here (60.6ms hooked vs 51.6ms unhooked for
the same scene). So it runs:

1. **no hooks** → the authoritative TOTAL,
2. **stage hooks only** (~7 syncs/pass) → per-stage,
3. **leaf hooks** → per-layer, which stays inflated and is labelled as such -
   **use it for ranking, not absolute cost**.

`stages + other` is reported from pass 2 so it reconciles with that pass's own
total; it will not exactly equal the unhooked TOTAL, and that's expected.

## Gotcha: pytorch3d must be a CUDA build

The 2D branch is `object_renderer` → `clip_image`, and only the **renderer**
needs a CUDA pytorch3d. A **CPU-only pytorch3d** raises
`RuntimeError: Not compiled with GPU support` at `rasterize_points` and takes the
whole pipeline down with it - including `clip_image`, the most expensive module.

> **Use the right env.** There are two conda envs named `ges3vig-gpu` on this box
> and only one of them works:
>
> | env | pytorch3d | use it? |
> |---|---|---|
> | `/home/vim/miniconda3/envs/ges3vig-gpu` | CUDA - **works** | **yes** - this is what `conda activate ges3vig-gpu` gives you, and what the eval suite uses |
> | `/nas_hcfc/vimarsh/HRI/miniconda3/envs/ges3vig-gpu` | CPU-only - broken | no (stale duplicate) |
>
> Same name, same version (0.7.8), different builds. If the renderer explodes,
> check which python you actually invoked.

If you are stuck on a host with a CPU-only pytorch3d, the profiler
**auto-enables `--stub-renderer`**: rasterization is replaced by random images of
the same shape, and `clip_image` + fusion still run for real. The reported total
then **excludes rasterization cost**, so treat those totals as a lower bound. It is a fallback, not the way to measure.

```bash
# on a good env this runs the real renderer; on a broken one it stubs + warns
$PY profiling/profile_jetson.py --mode full
```

`--skip-2d` also exists but is blunt: it removes the 2D branch entirely, which
changes `match_module`'s input dims and skips the foveation case. Prefer the stub.

The **Jetson image builds pytorch3d from source with `FORCE_CUDA=1`**, so on the
Jetson the renderer runs for real and no stub is applied.

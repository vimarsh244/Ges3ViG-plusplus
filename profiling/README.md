# Profiling Ges3ViG on a Jetson Orin Nano

`profile_jetson.py` measures **per-layer / per-module forward latency** of the
original Ges3ViG model, driven by **random tensors** - no dataset, no checkpoint
required. It warms up, times N passes, and reports
`mean / std / min / p50 / p90 / p95 / max` plus `fps`, for a single sample and
for a batch.

## The two default commands

```bash
PY=/nas_hcfc/vimarsh/HRI/miniconda3/envs/ges3vig-gpu/bin/python   # local dev box
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
| `dropout` | random point drop (keep-ratio) | `1.0 0.75 0.5 0.25 0.1` (`--dropout-sweep`) |
| `foveation` | #proposals reaching the 2D encoder + fusion | `80 48 32 16 8 4 1` (`--foveation-sweep`) |
| `batch` | batch size | `1 2 4` (`--batch-sweep`) |

Every measurement lands in one CSV with `case`, `param`, `num_points`,
`keep_ratio`, `batch_size`, `kind` (stage/total/module), the latency stats and
`fps`. Each case also prints a compact scan table.

A note on **foveation**: the original model has no foveation logic (the
`Ges3ViG/` fork does). So this case does *not* foveate the model - it measures
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
| `joint_module` | gesture self-attention | `B×55×128` joint features |
| `match_module` | cross-modal fusion | `B×80` proposals + `132` word tokens |
| `human_processor` | joint regression / bias weights | 30k points × 32ch |

Use `--num-proposals` to set how many proposals per scene the 2D and fusion
paths see (default 48; the model pads to `max_num_proposals=80`).

### `--mode full` - whole pipeline

Synthesizes a **random point cloud** (dense Gaussian blobs, so the ball-query /
BFS clustering actually forms proposals), voxelizes it *exactly* like the real
dataloader, and runs the entire model. Forward hooks time:

* every **stage** (`detector`, `human_processor`, `object_renderer`,
  `clip_image`, `text_encoder`, `joint_module`, `match_module`),
* the **TOTAL** forward,
* the **slowest leaf layers** (`--top-layers`, default 25) - individual
  Minkowski convs, linears, etc.

Whatever the stages don't account for is reported as *non-module ops*
(pointing-bias loop, sparse→dense conversion, the final einsum).

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
| `--skip-2d` | disable the 2D branch - **see the pytorch3d note below** |

## Reading the output

Every table has `mean std min p50 p95 max fps`.

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

The `ges3vig-gpu` conda env on the x86 dev box has exactly this problem: its
pytorch3d 0.7.8 has no CUDA symbols and isn't linked against libcudart.

So the profiler **auto-enables `--stub-renderer`** there: rasterization is
replaced by random images of the same shape, and `clip_image` + fusion still run
for real. The reported total then **excludes rasterization cost** (it's called
out in the output). Nothing else changes - the model keeps its real architecture.

```bash
# just works; the stub is applied automatically with a warning
$PY profiling/profile_jetson.py --mode full
```

`--skip-2d` also exists but is blunt: it removes the 2D branch entirely, which
changes `match_module`'s input dims and skips the foveation case. Prefer the stub.

The **Jetson image builds pytorch3d from source with `FORCE_CUDA=1`**, so on the
Jetson the renderer runs for real and no stub is applied.

## Reference numbers (x86 RTX 4000 Ada, random weights, renderer stubbed)

Not Jetson numbers - a baseline to sanity-check the Jetson run against. The Orin
Nano will be substantially slower. All from the default sweep command above.

### Where the time goes (`--mode full`, 20k pts, bs=1)

| stage | p50 (ms) | share |
|---|---|---|
| **`clip_image` (2D vision encoder)** | **111.1** | **66%** |
| `detector` (PointGroup + Minkowski + clustering) | 51.9 | 31% |
| `text_encoder` | 2.4 | |
| `match_module` (fusion) | 1.0 | |
| `human_processor` | 0.6 | |
| `joint_module` | 0.3 | |
| non-module ops (bias/dense-convert/einsum) | 1.1 | |
| **TOTAL** | **~168.6** | **5.9 fps** |

### Point-count scan (`points`)

The detector is ~linear in points; the total tracks it.

| points | total ms | fps | detector ms |
|---|---|---|---|
| 10k | 137.2 | 7.29 | 41.1 |
| 30k | 175.9 | 5.69 | 62.3 |
| 50k | 210.2 | 4.76 | 95.7 |
| 70k | 259.1 | 3.86 | 147.3 |
| 90k | 302.8 | 3.30 | 187.6 |

### Foveation cost curve (`foveation`) - 2D encoder + fusion only

Dead linear in proposals. This is the headline: the 2D branch alone costs more
than the entire rest of the network at the default 80 proposals.

| proposals | 2D+fusion ms | fps |
|---|---|---|
| 80 | 258.1 | 3.88 |
| 48 | 156.0 | 6.41 |
| 32 | 98.9 | 10.11 |
| 16 | 50.7 | 19.74 |
| 8 | 25.2 | 39.72 |
| 4 | 13.5 | 73.91 |
| 1 | 4.9 | - |

### Batch scan (`batch`) - per-scene throughput

Batching helps a little; it does not change the picture.

| batch | total ms | ms/scene | fps |
|---|---|---|---|
| 1 | 177.7 | 177.7 | 5.63 |
| 2 | 332.9 | 166.5 | 6.01 |
| 4 | 619.3 | 154.8 | 6.46 |

### Dropout scan (`dropout`, 40k base)

| keep | total ms | fps | detector ms |
|---|---|---|---|
| 1.0 | 179.7 | 5.56 | 72.9 |
| 0.75 | 188.5 | 5.30 | 63.5 |
| 0.5 | 165.2 | 6.05 | 52.4 |

Note the **detector falls monotonically** (72.9 → 63.5 → 52.4) but the **total
does not** - because dropping points also changes how many proposals the
clustering emits, and `clip_image` cost is driven by proposal count, not point
count. Dropout is not a reliable lever on end-to-end latency; foveation is.

### What this says

1. **The 2D CLIP image encoder is the bottleneck** - 66% of the forward, linear
   in `proposals × 3 views` **per scene**. Attack this first.
2. **The detector owns the rest** (~31%), almost all MinkowskiEngine sparse convs
   in `score_net` / `backbone`, and it scales with point count.
3. **Everything else is noise** (<3% combined): text, fusion, gesture, human.
4. Foveating to 16 proposals would cut the 2D branch ~5× (258 → 51ms), which is
   worth far more than any point-cloud decimation.

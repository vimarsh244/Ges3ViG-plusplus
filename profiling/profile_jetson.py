#!/usr/bin/env python3
"""
Per-layer / per-module latency profiler for the ORIGINAL Ges3ViG model.

Goal
----
Measure how long a single forward pass takes for every module of the network
(vision 3D detector, human processor, 2D vision/CLIP image encoder, text encoder,
gesture/joint module, cross-modal fusion/match module ...) plus the end-to-end
total, so we can understand where the time goes on an NVIDIA Jetson Orin Nano.

It is deliberately self-contained: it drives the model with **random tensors**
(no dataset required), warms up, then times several passes and averages.

Two profiling modes
-------------------
  modules  (default, bulletproof)
      Micro-benchmark each dense-input sub-module in isolation with random
      tensors of config-derived shapes. Runs anywhere (no dataset, no CUDA
      custom-op geometry needed for the safe modules). Reports per-module
      latency for batch-size 1 and a larger batch.

  full     (whole-pipeline)
      Build a synthetic random `data_dict` (a plausible random point cloud that
      is voxelized exactly like the real dataloader), run the *entire* model,
      and time every layer with forward hooks + every top-level stage + the
      grand total. If the synthetic scene trips a CUDA custom op you can instead
      feed a real cached batch with `--batch scene.pt` (dump one with
      `--dump-batch`, see README).

Both modes:
  * run `--warmup` passes first (JIT / cudnn autotune / allocator warm),
  * time `--repeats` passes and report mean / std / min / p50 / p95 / max (ms),
  * synchronize CUDA around every timed region so numbers are real,
  * optionally write a CSV.

Examples
--------
    PY=/nas_hcfc/vimarsh/HRI/miniconda3/envs/ges3vig-gpu/bin/python

    # per-module micro-benchmark, bs=1 and bs=4, 30 timed passes
    $PY profiling/profile_jetson.py --mode modules --batch-sizes 1 4 --repeats 30

    # whole-model end-to-end on a synthetic random scene
    $PY profiling/profile_jetson.py --mode full --num-points 40000 --repeats 20

    # whole-model on a real cached batch (most representative)
    $PY profiling/profile_jetson.py --mode full --batch /tmp/one_scene.pt
"""

import argparse
import csv
import gc
import os
import platform
import statistics
import sys
import time
from contextlib import nullcontext

# ── make the project importable when run from anywhere ────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
from omegaconf import DictConfig, ListConfig, OmegaConf

# PyTorch 2.6 flipped weights_only=True; omegaconf types live in Lightning ckpts.
torch.serialization.add_safe_globals([DictConfig, ListConfig])


# small helpers
def is_jetson() -> bool:
    """Best-effort NVIDIA Jetson (aarch64 + tegra) detection."""
    if platform.machine() not in ("aarch64", "arm64"):
        return False
    for path in ("/proc/device-tree/model", "/etc/nv_tegra_release",
                 "/proc/device-tree/compatible"):
        try:
            with open(path) as f:
                txt = f.read().lower()
            if "jetson" in txt or "tegra" in txt or "nvidia" in txt:
                return True
        except (FileNotFoundError, UnicodeDecodeError, PermissionError):
            pass
    return torch.cuda.is_available()


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def pytorch3d_has_cuda() -> bool:
    """
    True if the installed pytorch3d was built with CUDA support.

    The 2D object renderer rasterizes on the GPU; a CPU-only pytorch3d wheel
    raises "Not compiled with GPU support" at rasterize_points. The Jetson image
    builds pytorch3d from source with FORCE_CUDA=1 so this holds there.
    """
    try:
        from pytorch3d import _C
        return hasattr(_C, "_rasterize_points_cuda") or _cuda_symbols_present(_C)
    except Exception:  # noqa: BLE001
        return False


def _cuda_symbols_present(_C) -> bool:
    """Cheap probe: try a 1-point CUDA rasterization and see if it errors."""
    if not torch.cuda.is_available():
        return False
    try:
        from pytorch3d.renderer.points.rasterize_points import rasterize_points
        from pytorch3d.structures import Pointclouds
        pcd = Pointclouds(points=[torch.zeros(1, 3, device="cuda")])
        rasterize_points(pcd, image_size=8, radius=0.1, points_per_pixel=1)
        return True
    except RuntimeError as e:
        if "Not compiled with GPU support" in str(e):
            return False
        return True
    except Exception:  # noqa: BLE001
        return True


def summarize(latencies_ms, batch_size=1):
    """
    mean/std/min/p50/p95/max for a list of latencies (ms), plus throughput.

    `fps` is scenes/second = 1 / (per-scene time), i.e. batch_size / mean_seconds.
    For a batched pass that is the per-scene throughput, not the pass rate.
    """
    if not latencies_ms:
        return {}
    vals = sorted(latencies_ms)

    def pct(p):
        if len(vals) == 1:
            return vals[0]
        idx = (len(vals) - 1) * p
        lo = int(idx)
        hi = min(lo + 1, len(vals) - 1)
        frac = idx - lo
        return vals[lo] * (1 - frac) + vals[hi] * frac

    mean_ms = statistics.fmean(vals)
    p50 = pct(0.50)
    return {
        "mean_ms": mean_ms,
        "std_ms": statistics.pstdev(vals) if len(vals) > 1 else 0.0,
        "min_ms": vals[0],
        "p50_ms": p50,
        "p90_ms": pct(0.90),
        "p95_ms": pct(0.95),
        "max_ms": vals[-1],
        "ms_per_scene": mean_ms / max(batch_size, 1),
        "fps": (1000.0 * max(batch_size, 1) / mean_ms) if mean_ms > 0 else float("inf"),
        # p50-based throughput: robust to a single contended outlier pass
        "fps_p50": (1000.0 * max(batch_size, 1) / p50) if p50 > 0 else float("inf"),
    }


def time_callable(fn, device, warmup, repeats):
    """Warm up, then time `fn` `repeats` times with CUDA sync. Returns list[ms]."""
    for _ in range(warmup):
        fn()
    sync(device)
    lat = []
    for _ in range(repeats):
        sync(device)
        t0 = time.perf_counter()
        fn()
        sync(device)
        lat.append((time.perf_counter() - t0) * 1000.0)
    return lat


def move(obj, device):
    if torch.is_tensor(obj):
        return obj.to(device, non_blocking=True)
    if isinstance(obj, dict):
        return {k: move(v, device) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(move(v, device) for v in obj)
    return obj


def load_cfg(data_override):
    """Compose the Hydra config the same way test.py does (no CLI runner)."""
    from hydra import compose, initialize_config_dir
    cfg_dir = os.path.join(PROJECT_ROOT, "config")
    with initialize_config_dir(config_dir=cfg_dir, version_base=None):
        cfg = compose(config_name="global_imputed_config",
                      overrides=[f"data={data_override}"])
    return cfg


def build_model(cfg, device, ckpt_path=None):
    """Instantiate the model; optionally load real weights (else random init)."""
    from hydra.utils import instantiate
    model = instantiate(cfg.model.model_name, cfg)
    if ckpt_path:
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state = ckpt.get("state_dict", ckpt)
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing or unexpected:
            print(f"[ckpt] loaded with {len(missing)} missing / "
                  f"{len(unexpected)} unexpected keys (non-strict).")
        del ckpt
        gc.collect()
    else:
        print("[model] using RANDOM weights (fine for latency; pass --ckpt for real ones)")
    model.to(device)
    model.eval()
    return model


def stub_object_renderer(model, cfg):
    """
    Replace the pytorch3d rasterization with random images of the same shape.

    Why: the 2D branch is renderer -> clip_image, and only the *renderer* needs a
    CUDA pytorch3d. On a host with a CPU-only pytorch3d the renderer raises
    "Not compiled with GPU support" and takes the whole pipeline down with it —
    including clip_image, which is the single most expensive module.

    Stubbing only the rasterizer keeps the model's real architecture and lets
    clip_image + fusion run for real. We are profiling on random tensors anyway,
    so random images are consistent with the rest of the method. It does mean the
    reported total EXCLUDES rasterization cost — which is called out in the
    output, and is not needed on the Jetson image (it has a CUDA pytorch3d).
    """
    renderer = model.object_renderer
    views = renderer.views
    img_size = renderer.image_size
    device = model.device

    def _stub(data_dict, output_dict):
        n_aabbs = output_dict["pred_aabb_min_max_bounds"].shape[0]
        return torch.rand(n_aabbs * views, img_size, img_size, 3,
                          dtype=torch.float32, device=device)

    renderer.forward = _stub
    return model


def write_csv(path, rows):
    if not rows:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fields = list(rows[0].keys())
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"\n[csv] wrote {len(rows)} rows -> {path}")


LABEL_W = 44
TABLE_W = LABEL_W + 7 * 9 + 9


def _fit(label):
    """Keep the informative tail of long dotted module paths."""
    if len(label) <= LABEL_W:
        return f"{label:<{LABEL_W}}"
    return "…" + label[-(LABEL_W - 1):]


def table_header(title):
    print(f"\n{title}")
    print("-" * TABLE_W)
    print(f"{'module / stage':<{LABEL_W}}{'mean':>9}{'std':>9}{'min':>9}"
          f"{'p50':>9}{'p90':>9}{'p95':>9}{'max':>9}{'fps':>9}")
    print("-" * TABLE_W)


def table_row(label, s):
    if not s:
        print(f"{_fit(label)}{'(skipped / n-a)':>58}")
        return
    print(f"{_fit(label)}{s['mean_ms']:>9.3f}{s['std_ms']:>9.3f}{s['min_ms']:>9.3f}"
          f"{s['p50_ms']:>9.3f}{s['p90_ms']:>9.3f}{s['p95_ms']:>9.3f}"
          f"{s['max_ms']:>9.3f}{s['fps']:>9.2f}")


def print_table(title, stat_rows, unit="ms"):
    """stat_rows: list of (label, stats_dict). Prints a latency table."""
    table_header(title)
    for label, s in stat_rows:
        table_row(label, s)
    print("-" * TABLE_W)


# ══════════════════════════════════════════════════════════════════════════════
# MODE 1 — per-module micro-benchmark with random tensors
# ══════════════════════════════════════════════════════════════════════════════
def make_synthetic_module_inputs(model, cfg, device, B, num_proposals):
    """
    Build random inputs (correct shapes/dtypes) for each dense sub-module.
    Returns a dict {module_key: (callable_returning_a_fresh_forward_thunk, note)}.
    Every builder returns a *thunk* so we can time repeated calls cheaply.
    """
    net = cfg.model.network
    K = int(cfg.model.network.max_num_proposals)          # padded proposal slots (80)
    active = min(num_proposals, K)                        # "real" proposals per scene
    chunk = int(cfg.data.chunk_size)
    feat = int(net.matching_module.feat_channel)          # 128
    heads = int(net.matching_module.head)                 # 4
    det_ch = int(net.detector.output_channel)             # 32
    img_ch = int(net.clip_img_encoder.output_channel)     # 128
    num_joints = int(getattr(model.human_processor, "num_joints", 55))
    views = len(net.object_renderer.eye)                  # 3
    img_size = int(net.object_renderer.rasterizer_setting.image_size)  # 224
    match_in = det_ch * int(net.use_3d_features) + int(net.use_2d_feature) * img_ch  # 160
    # word sequence length = CLIP text tokens (77) + gesture/joint tokens
    ctx = 77
    word_len = ctx + num_joints

    builders = {}

    # ── 2D vision encoder: CLIP image encoder (per-proposal, per-view images) ──
    if net.use_2d_feature:
        def _vision():
            n_imgs = B * active * views
            x = torch.rand(n_imgs, 3, img_size, img_size, device=device)
            return lambda: model.clip_image(x)
        builders["clip_image (2D vision encoder)"] = (
            _vision, f"{B}x{active}x{views} imgs of 3x{img_size}x{img_size}")

    # ── language: CLIP word/text encoder ──
    def _text():
        # clip_tokens: (B, chunk, 77) int -> encoder flattens to (B*chunk, 77)
        tok = torch.randint(0, 49407, (B, chunk, ctx), device=device, dtype=torch.long)
        dd = {"clip_tokens": tok}
        od = {}
        return lambda: model.text_encoder(dd, od)
    builders["text_encoder (language/CLIP)"] = (_text, f"{B}x{chunk} sentences of {ctx} tokens")

    # ── gesture stream: joint self-attention module ──
    def _joint():
        jf = torch.randn(B, num_joints, feat, device=device)
        return lambda: model.joint_module(jf)
    builders["joint_module (gesture attn)"] = (_joint, f"{B}x{num_joints}x{feat}")

    # ── cross-modal fusion: match module ──
    def _match():
        od = {
            "aabb_features": torch.randn(B, K, match_in, device=device),
            "pred_aabb_min_max_bounds": torch.randn(B, K, 2, 3, device=device),
            "proposal_masks_dense": torch.zeros(B, K, dtype=torch.bool, device=device),
            "word_features": torch.randn(B * chunk, word_len, feat, device=device),
        }
        od["proposal_masks_dense"][:, :active] = True
        dd = {
            "ann_id": torch.zeros(B, chunk, dtype=torch.long, device=device),
            "lang_attention_mask": torch.zeros(
                B * chunk, heads, K, word_len, dtype=torch.bool, device=device),
        }
        return lambda: model.match_module(dd, od)
    builders["match_module (cross-modal fusion)"] = (_match, f"{B}x{K} prop, {word_len} word tok")

    # ── human processor (feasible with random per-point tensors) ──
    def _human():
        # enough points that >300 land on the human class after random argmax
        npts = 30000
        pxyz = torch.rand(npts, 3, device=device) * 4.0
        pfeat = torch.randn(npts, det_ch, device=device)
        vbatch = torch.randint(0, B, (npts,), device=device, dtype=torch.uint8)
        # bias the semantic scores so the human class is well represented
        sem = torch.randn(npts, int(cfg.data.semantic_class), device=device)
        human_label = model.human_processor.human_label
        sem[:, human_label] += 1.5
        return lambda: model.human_processor(pxyz, pfeat, vbatch, sem, B)
    builders["human_processor"] = (_human, f"{B} scenes, 30k pts of {det_ch}ch")

    return builders


def run_modules_mode(model, cfg, device, args):
    print("\n" + "=" * 92)
    print("MODE: modules  —  per-module micro-benchmark with RANDOM tensors")
    print("=" * 92)

    autocast = (torch.amp.autocast(device_type="cuda", dtype=torch.float16)
                if args.fp16 and device.type == "cuda" else nullcontext())

    csv_rows = []
    with torch.inference_mode(), autocast:
        for B in args.batch_sizes:
            builders = make_synthetic_module_inputs(model, cfg, device, B, args.num_proposals)
            stat_rows = []
            for label, (make_thunk, note) in builders.items():
                try:
                    thunk = make_thunk()
                    lat = time_callable(thunk, device, args.warmup, args.repeats)
                    s = summarize(lat, B)
                except Exception as e:  # noqa: BLE001 - keep going, report the module as skipped
                    s = {}
                    print(f"[warn] {label} (B={B}) failed: {type(e).__name__}: {e}")
                stat_rows.append((f"{label}", s))
                if s:
                    csv_rows.append({"mode": "modules", "case": "modules", "param": B,
                                     "batch_size": B, "module": label,
                                     "input_note": note, **{k: round(v, 4) for k, v in s.items()}})
                # free per-module allocations before the next one
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                gc.collect()
            print_table(f"Per-module latency (ms)  |  batch_size = {B}  |  "
                        f"{args.repeats} timed passes (after {args.warmup} warmup)", stat_rows)

    if args.csv:
        write_csv(args.csv, csv_rows)


# ══════════════════════════════════════════════════════════════════════════════
# MODE 2 — whole-pipeline: synthetic random data_dict, hook every layer
# ══════════════════════════════════════════════════════════════════════════════
def synthesize_data_dict(cfg, device, num_points, batch_size, seed=0, keep_ratio=1.0):
    """
    Create a plausible RANDOM scene and voxelize it exactly like the dataloader,
    so the whole model (incl. the MinkowskiEngine detector + custom clustering
    ops) can run end-to-end. Points are generated as a few dense Gaussian blobs
    (mimicking object/human surfaces) so the clustering actually forms proposals.
    """
    import numpy as np
    import MinkowskiEngine as ME
    rng = np.random.default_rng(seed)

    use_rgb = bool(cfg.data.point_features.use_rgb)
    use_normal = bool(cfg.data.point_features.use_normal)
    voxel_size = float(cfg.data.voxel_size)
    num_joints = 55

    per_scene = max(2000, num_points)
    voxel_xyz_list, voxel_feats_list, voxel_point_map_list = [], [], []
    point_xyz_all, point_rgb_all, vert_batch_ids = [], [], []
    joint_name_idx_map = []
    all_point_count_offsets = [0]
    num_voxel_batch = 0

    # joint layout the pointing-bias code indexes into
    joint_names = ["left_shoulder", "right_shoulder", "left_index1", "right_index1"]
    name_map = {n: i for i, n in enumerate(joint_names)}
    # pad the rest of the joints with unique indices
    for i in range(len(joint_names), num_joints):
        name_map[f"joint_{i}"] = i

    for b in range(batch_size):
        # a handful of dense blobs => the ball-query/bfs clustering finds proposals
        n_blobs = 12
        centers = rng.uniform(0.0, 4.0, size=(n_blobs, 3)).astype(np.float32)
        pts_per = per_scene // n_blobs
        blobs = []
        for c in centers:
            blobs.append(c[None, :] + rng.normal(0, 0.06, size=(pts_per, 3)).astype(np.float32))
        pxyz = np.concatenate(blobs, axis=0).astype(np.float32)

        # random point dropout — emulates a sparser / decimated sensor cloud
        if keep_ratio < 1.0:
            n_keep = max(64, int(pxyz.shape[0] * keep_ratio))
            sel = rng.choice(pxyz.shape[0], size=n_keep, replace=False)
            pxyz = pxyz[sel]
        n = pxyz.shape[0]

        rgb = rng.uniform(0.0, 1.0, size=(n, 3)).astype(np.float32)
        feats = np.empty((n, 0), np.float32)
        if use_rgb:
            feats = np.concatenate((feats, rgb), axis=1)
        if use_normal:
            nrm = rng.normal(0, 1, size=(n, 3)).astype(np.float32)
            feats = np.concatenate((feats, nrm), axis=1)
        feats = np.concatenate((feats, pxyz), axis=1)  # + xyz, exactly like the dataset

        v_xyz, v_feats, _, v_p2v = ME.utils.sparse_quantize(
            coordinates=pxyz - pxyz.min(axis=0), features=feats,
            return_index=True, return_inverse=True, quantization_size=voxel_size,
        )
        voxel_xyz_list.append(v_xyz)
        voxel_feats_list.append(v_feats)
        voxel_point_map_list.append(v_p2v + num_voxel_batch)
        num_voxel_batch += v_xyz.shape[0]

        point_xyz_all.append(torch.from_numpy(pxyz))
        point_rgb_all.append(torch.from_numpy(rgb))
        vert_batch_ids.append(torch.full((n,), b, dtype=torch.uint8))
        joint_name_idx_map.append(name_map)
        all_point_count_offsets.append(all_point_count_offsets[-1] + n)

    voxel_xyz, voxel_features = ME.utils.sparse_collate(
        coords=voxel_xyz_list, feats=voxel_feats_list)
    voxel_point_map = torch.cat(voxel_point_map_list, dim=0)

    chunk = int(cfg.data.chunk_size)
    dd = {
        "scene_id": [f"synthetic_{i}" for i in range(batch_size)],
        "voxel_xyz": voxel_xyz,
        "voxel_features": voxel_features,
        "voxel_point_map": voxel_point_map,
        "point_xyz": torch.cat(point_xyz_all, dim=0),
        "vert_batch_ids": torch.cat(vert_batch_ids, dim=0),
        # full-resolution cloud used by the 2D object renderer (same cloud here)
        "all_point_xyz": torch.cat(point_xyz_all, dim=0),
        "all_point_rgb": torch.cat(point_rgb_all, dim=0),
        "all_point_count_offsets": torch.tensor(all_point_count_offsets, dtype=torch.int32),
        "ann_id": torch.zeros(batch_size, chunk, dtype=torch.long),
        "object_id": torch.zeros(batch_size, chunk, dtype=torch.long),
        "clip_tokens": torch.randint(0, 49407, (batch_size, chunk, 77), dtype=torch.long),
        "joint_name_idx_map": joint_name_idx_map,
        "scene_center_xyz": torch.zeros(batch_size, 3),
    }
    return move(dd, device)


class LayerTimer:
    """Register forward pre/post hooks on modules to accumulate per-call latency."""

    def __init__(self, device):
        self.device = device
        self.handles = []
        self.starts = {}
        self.records = {}   # name -> list[ms]

    def _pre(self, name):
        def hook(_module, _inp):
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            self.starts[name] = time.perf_counter()
        return hook

    def _post(self, name):
        def hook(_module, _inp, _out):
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            dt = (time.perf_counter() - self.starts.get(name, time.perf_counter())) * 1000.0
            self.records.setdefault(name, []).append(dt)
        return hook

    def attach(self, model, names):
        for name, module in model.named_modules():
            if name in names:
                self.handles.append(module.register_forward_pre_hook(self._pre(name)))
                self.handles.append(module.register_forward_hook(self._post(name)))

    def reset(self):
        self.records.clear()
        self.starts.clear()

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()


TOP_STAGES = ["detector", "human_processor", "object_renderer", "clip_image",
              "text_encoder", "joint_module", "match_module"]


def get_batch(cfg, device, args, batch_size, num_points, keep_ratio=1.0):
    """Return (data_dict, batch_size, description) for a full-pipeline pass."""
    if args.batch:
        dd = move(torch.load(args.batch, map_location="cpu", weights_only=False), device)
        return dd, len(dd["scene_id"]), f"cached batch {os.path.basename(args.batch)}"
    dd = synthesize_data_dict(cfg, device, num_points, batch_size,
                              seed=args.seed, keep_ratio=keep_ratio)
    n_pts = int(dd["point_xyz"].shape[0])
    return dd, batch_size, f"synthetic {n_pts} pts (keep={keep_ratio:g}) x bs{batch_size}"


def measure_full(model, device, data_dict, batch_size, args, want_layers=False):
    """
    Time a full forward.

    Timing is done in SEPARATE passes on purpose. Hooks CUDA-synchronize around
    every module they wrap, which serializes the GPU — so measuring the total
    and the per-layer breakdown in one pass inflates both. Instead:
      pass 1: no hooks         -> honest TOTAL
      pass 2: stage hooks only -> per-stage (small, ~7 syncs/pass)
      pass 3: leaf hooks       -> per-layer breakdown (inflated, relative only)
    """
    autocast = (torch.amp.autocast(device_type="cuda", dtype=torch.float16)
                if args.fp16 and device.type == "cuda" else nullcontext())

    def run_passes(timer=None):
        with torch.inference_mode(), autocast:
            for _ in range(args.warmup):
                model(data_dict)
            sync(device)
            if timer:
                timer.reset()
            lat = []
            for _ in range(args.repeats):
                sync(device)
                t0 = time.perf_counter()
                model(data_dict)
                sync(device)
                lat.append((time.perf_counter() - t0) * 1000.0)
        return lat

    # ── pass 1: unhooked TOTAL ───────────────────────────────────────────────
    total_stats = summarize(run_passes(), batch_size)

    # ── pass 2: per-stage ────────────────────────────────────────────────────
    # Keep this pass's own total too: `other` (the non-module ops) must be
    # derived from the SAME pass as the stages, otherwise run-to-run variance
    # can make the stage sum exceed the unhooked total and produce nonsense.
    stage_timer = LayerTimer(device)
    stage_timer.attach(model, [s for s in TOP_STAGES if hasattr(model, s)])
    try:
        stage_pass_lat = run_passes(stage_timer)
        stage_rows = [(n, summarize(v, batch_size))
                      for n, v in stage_timer.records.items()]
    finally:
        stage_timer.remove()
    stage_rows.sort(key=lambda r: r[1].get("mean_ms", 0.0), reverse=True)
    stage_pass_stats = summarize(stage_pass_lat, batch_size)

    # ── pass 3: per-leaf-layer (optional; costly + inflated) ─────────────────
    layer_rows = []
    if want_layers:
        leaf_names = [n for n, m in model.named_modules()
                      if n and len(list(m.children())) == 0]
        layer_timer = LayerTimer(device)
        layer_timer.attach(model, leaf_names)
        try:
            run_passes(layer_timer)
            layer_rows = [(n, summarize(v, batch_size))
                          for n, v in layer_timer.records.items()]
        finally:
            layer_timer.remove()
        layer_rows.sort(key=lambda r: r[1].get("mean_ms", 0.0), reverse=True)

    return total_stats, stage_rows, layer_rows, stage_pass_stats


def run_full_mode(model, cfg, device, args):
    print("\n" + "=" * TABLE_W)
    print("MODE: full  —  whole-model per-layer + per-stage + total")
    print("=" * TABLE_W)

    try:
        data_dict, batch_size, desc = get_batch(
            cfg, device, args, args.batch_sizes[0], args.num_points, args.keep_ratio)
        print(f"[input] {desc}")
        total_stats, stage_rows, layer_rows, stage_pass_stats = measure_full(
            model, device, data_dict, batch_size, args, want_layers=True)
    except Exception as e:  # noqa: BLE001
        print(f"\n[error] full-model forward failed:\n"
              f"        {type(e).__name__}: {e}\n"
              f"        Tip: on a host whose pytorch3d lacks CUDA, add --skip-2d; "
              f"or feed a real batch with --batch scene.pt.")
        return

    stage_sum = sum(s["mean_ms"] for _, s in stage_rows)
    # derived from the stage pass, so stages + other == that pass's own total
    other_ms = stage_pass_stats["mean_ms"] - stage_sum

    print_table(f"Per-stage latency (ms)  |  batch_size = {batch_size}  |  "
                f"{args.repeats} passes (after {args.warmup} warmup)", stage_rows)
    print(f"{'(non-module ops: bias/dense-convert/einsum)':<{LABEL_W}}{other_ms:>9.3f}")
    print(f"{'  stages + other (hooked pass)':<{LABEL_W}}"
          f"{stage_pass_stats['mean_ms']:>9.3f}")
    table_row("TOTAL forward (unhooked, authoritative)", total_stats)
    print("=" * TABLE_W)
    if other_ms < -0.5:
        print("[warn] negative 'other' -> stage timings are noisy; raise --repeats")

    if layer_rows:
        print_table(f"Top {args.top_layers} slowest leaf layers (ms) — hook-inflated, "
                    f"use for RANKING not absolute cost", layer_rows[: args.top_layers])

    if args.csv:
        rows = [{"mode": "full", "case": "single", "param": args.num_points,
                 "kind": "stage", "batch_size": batch_size, "name": n,
                 **{k: round(v, 4) for k, v in s.items()}} for n, s in stage_rows]
        rows.append({"mode": "full", "case": "single", "param": args.num_points,
                     "kind": "total", "batch_size": batch_size, "name": "TOTAL",
                     **{k: round(v, 4) for k, v in total_stats.items()}})
        rows += [{"mode": "full", "case": "single", "param": args.num_points,
                  "kind": "layer", "batch_size": batch_size, "name": n,
                  **{k: round(v, 4) for k, v in s.items()}} for n, s in layer_rows]
        write_csv(args.csv, rows)


# ─────────────────────────────────────────────────────────────────────────────
# MODE 3 — sweep: auto-run every case and store the results
# ─────────────────────────────────────────────────────────────────────────────
def _sweep_summary(title, rows, param_label, last_label="detector_ms"):
    """Compact one-line-per-point view of a sweep."""
    if title:
        print(f"\n{title}")
    print("-" * 78)
    print(f"{param_label:>14}{'total_ms':>12}{'ms/scene':>12}{'fps':>10}{last_label:>14}")
    print("-" * 78)
    for r in rows:
        print(f"{r['param']:>14}{r['total_ms']:>12.2f}{r['ms_per_scene']:>12.2f}"
              f"{r['fps']:>10.2f}{r.get('detail_ms', float('nan')):>14.2f}")
    print("-" * 78)


def run_sweep_mode(model, cfg, device, args):
    """
    Run every profiling case back-to-back and store one combined CSV:

      points     : point-count scan (default 10k..90k step 10k) - detector cost
      dropout    : random point drop (keep-ratio scan)          - sparser cloud
      foveation  : #proposals fed to the 2D encoder + fusion    - foveated 2D
      batch      : batch-size scan                              - throughput
    """
    cases = args.cases
    all_rows = []
    print("\n" + "=" * TABLE_W)
    print(f"MODE: sweep  —  cases: {', '.join(cases)}")
    print("=" * TABLE_W)

    def full_point(param, batch_size, num_points, keep_ratio, case):
        """One full-pipeline measurement -> flat summary row + CSV rows."""
        data_dict, bs, desc = get_batch(cfg, device, args, batch_size,
                                        num_points, keep_ratio)
        total, stages, _, _ = measure_full(model, device, data_dict, bs, args,
                                           want_layers=False)
        stage_map = {n: s for n, s in stages}
        for n, s in stages:
            all_rows.append({"mode": "sweep", "case": case, "param": param,
                             "kind": "stage", "batch_size": bs,
                             "num_points": int(data_dict["point_xyz"].shape[0]),
                             "keep_ratio": keep_ratio, "name": n,
                             **{k: round(v, 4) for k, v in s.items()}})
        all_rows.append({"mode": "sweep", "case": case, "param": param,
                         "kind": "total", "batch_size": bs,
                         "num_points": int(data_dict["point_xyz"].shape[0]),
                         "keep_ratio": keep_ratio, "name": "TOTAL",
                         **{k: round(v, 4) for k, v in total.items()}})
        del data_dict
        if device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()
        return {"param": param, "total_ms": total["mean_ms"],
                "ms_per_scene": total["ms_per_scene"], "fps": total["fps"],
                "detail_ms": stage_map.get("detector", {}).get("mean_ms", float("nan"))}

    # ── case: point-count scan ───────────────────────────────────────────────
    if "points" in cases:
        rows = []
        for n_pts in args.point_sweep:
            try:
                r = full_point(n_pts, args.batch_sizes[0], n_pts, 1.0, "points")
                rows.append(r)
                print(f"  points={n_pts:>6}  total={r['total_ms']:7.2f} ms  "
                      f"fps={r['fps']:6.2f}")
            except Exception as e:  # noqa: BLE001
                print(f"  points={n_pts:>6}  FAILED: {type(e).__name__}: {e}")
        if rows:
            _sweep_summary("Point-count scan (full pipeline)", rows, "num_points")

    # ── case: random point dropout ───────────────────────────────────────────
    if "dropout" in cases:
        rows = []
        for keep in args.dropout_sweep:
            try:
                r = full_point(keep, args.batch_sizes[0], args.num_points, keep, "dropout")
                rows.append(r)
                print(f"  keep={keep:<5g}  total={r['total_ms']:7.2f} ms  "
                      f"fps={r['fps']:6.2f}")
            except Exception as e:  # noqa: BLE001
                print(f"  keep={keep:<5g}  FAILED: {type(e).__name__}: {e}")
        if rows:
            _sweep_summary(f"Random point-dropout scan (base {args.num_points} pts)",
                           rows, "keep_ratio")

    # ── case: batch-size scan ────────────────────────────────────────────────
    if "batch" in cases:
        rows = []
        for bs in args.batch_sweep:
            try:
                r = full_point(bs, bs, args.num_points, 1.0, "batch")
                rows.append(r)
                print(f"  bs={bs:<4}     total={r['total_ms']:7.2f} ms  "
                      f"fps={r['fps']:6.2f}")
            except Exception as e:  # noqa: BLE001
                print(f"  bs={bs:<4}     FAILED: {type(e).__name__}: {e}")
        if rows:
            _sweep_summary("Batch-size scan (full pipeline)", rows, "batch_size")

    # ── case: foveation (how many proposals reach the 2D encoder) ────────────
    # The original model has no foveation, so this measures the *cost curve*:
    # what the 2D encoder + fusion would cost if only k proposals were rendered.
    if "foveation" in cases:
        if not cfg.model.network.use_2d_feature:
            print("\n[skip] foveation: 2D branch disabled (--skip-2d)")
        else:
            rows = []
            B = args.batch_sizes[0]
            for k in args.foveation_sweep:
                try:
                    builders = make_synthetic_module_inputs(model, cfg, device, B, k)
                    sub = {}
                    with torch.inference_mode():
                        for label in ("clip_image (2D vision encoder)",
                                      "match_module (cross-modal fusion)"):
                            if label not in builders:
                                continue
                            thunk = builders[label][0]()
                            s = summarize(time_callable(thunk, device, args.warmup,
                                                        args.repeats), B)
                            sub[label] = s
                            all_rows.append({"mode": "sweep", "case": "foveation",
                                             "param": k, "kind": "module",
                                             "batch_size": B, "name": label,
                                             **{key: round(v, 4) for key, v in s.items()}})
                    twod = sub.get("clip_image (2D vision encoder)", {}).get("mean_ms", 0.0)
                    fuse = sub.get("match_module (cross-modal fusion)", {}).get("mean_ms", 0.0)
                    tot = twod + fuse
                    rows.append({"param": k, "total_ms": tot,
                                 "ms_per_scene": tot / max(B, 1),
                                 "fps": 1000.0 * B / tot if tot else float("inf"),
                                 "detail_ms": twod})
                    print(f"  proposals={k:<4} 2D+fusion={tot:7.2f} ms")
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                    gc.collect()
                except Exception as e:  # noqa: BLE001
                    print(f"  proposals={k:<4} FAILED: {type(e).__name__}: {e}")
            if rows:
                _sweep_summary(
                    "Foveation cost curve — 2D vision encoder + fusion ONLY "
                    "(not the whole pipeline)", rows, "proposals",
                    last_label="clip_image_ms")

    out = args.csv or os.path.join(SCRIPT_DIR, "results", "sweep.csv")
    write_csv(out, all_rows)


# ══════════════════════════════════════════════════════════════════════════════
def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=["modules", "full", "sweep"], default="modules",
                   help="modules = per-module random-tensor microbench (default); "
                        "full = whole-model per-layer+stage+total; "
                        "sweep = auto-run every case (points/dropout/foveation/batch) "
                        "and store one combined CSV")
    p.add_argument("--data", default="noelbowimputerefer",
                   help="Hydra data config to compose (only sets model shapes; "
                        "no dataset is read). Default: noelbowimputerefer")
    p.add_argument("--ckpt", default=None,
                   help="optional checkpoint for real weights (latency is ~identical "
                        "with random weights)")
    p.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 4],
                   help="batch sizes to profile (modules mode). full mode uses the first.")
    p.add_argument("--repeats", type=int, default=30, help="timed passes to average")
    p.add_argument("--warmup", type=int, default=8, help="warmup passes before timing")
    p.add_argument("--num-proposals", type=int, default=48,
                   help="active proposals per scene for the 2D/fusion microbench")
    p.add_argument("--num-points", type=int, default=40000,
                   help="synthetic point count per scene (full mode)")
    p.add_argument("--keep-ratio", type=float, default=1.0,
                   help="randomly drop points, keeping this fraction (1.0 = keep all)")

    # ── sweep-mode knobs ─────────────────────────────────────────────────────
    p.add_argument("--cases", nargs="+", default=["points", "dropout", "foveation", "batch"],
                   choices=["points", "dropout", "foveation", "batch"],
                   help="which sweep cases to auto-run (sweep mode)")
    p.add_argument("--point-sweep", type=int, nargs="+",
                   default=list(range(10000, 90001, 10000)),
                   help="point counts to scan (default 10k..90k step 10k)")
    p.add_argument("--dropout-sweep", type=float, nargs="+",
                   default=[1.0, 0.75, 0.5],
                   help="keep-ratios to scan for random point dropout "
                        "(1.0 = no dropout)")
    p.add_argument("--foveation-sweep", type=int, nargs="+",
                   default=[80, 48, 32, 16, 8, 4, 1],
                   help="#proposals reaching the 2D encoder/fusion (foveation cost curve)")
    p.add_argument("--batch-sweep", type=int, nargs="+", default=[1, 2, 4],
                   help="batch sizes to scan in sweep mode (default 1 2 4)")
    p.add_argument("--top-layers", type=int, default=25,
                   help="how many slowest leaf layers to print (full mode)")
    p.add_argument("--batch", default=None,
                   help="path to a cached real data_dict (.pt) to use in full mode")
    p.add_argument("--cpu", action="store_true", help="force CPU (default: CUDA if available)")
    p.add_argument("--skip-2d", action="store_true",
                   help="disable the 2D branch entirely (object_renderer + clip_image). "
                        "Blunt: it also changes match_module's input dims and skips the "
                        "foveation case. Prefer --stub-renderer.")
    p.add_argument("--stub-renderer", action="store_true",
                   help="replace pytorch3d rasterization with random images of the same "
                        "shape, keeping clip_image + fusion real. Auto-enabled when the "
                        "local pytorch3d has no CUDA. Excludes rasterizer cost from the "
                        "total. Not needed on the Jetson image.")
    p.add_argument("--no-auto-stub", action="store_true",
                   help="do not auto-enable --stub-renderer even if pytorch3d lacks CUDA "
                        "(let the renderer fail loudly instead)")
    p.add_argument("--fp16", action="store_true",
                   help="autocast fp16 (disabled automatically if MinkowskiEngine is present)")
    p.add_argument("--csv", default=None, help="optional CSV output path")
    p.add_argument("--seed", type=int, default=123)
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    use_cuda = (not args.cpu) and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    cfg = load_cfg(args.data)

    # The 2D branch rasterizes with pytorch3d on the GPU. A CPU-only pytorch3d
    # cannot, so fall back to stubbing just the rasterizer (keeps clip_image real).
    need_stub = False
    if cfg.model.network.use_2d_feature and not args.skip_2d:
        if use_cuda and not pytorch3d_has_cuda():
            if args.no_auto_stub:
                print("[warn] pytorch3d has NO CUDA here; the renderer will fail "
                      "(--no-auto-stub given)")
            else:
                need_stub = True
                print("[warn] pytorch3d has NO CUDA in this env -> auto-enabling "
                      "--stub-renderer:\n"
                      "       rasterization is replaced by random images; "
                      "clip_image + fusion still run\n"
                      "       for real, but the total EXCLUDES rasterizer cost. "
                      "Not needed on Jetson.")
    if args.stub_renderer:
        need_stub = True

    if args.skip_2d:
        OmegaConf.set_struct(cfg, False)
        cfg.model.network.use_2d_feature = False
        print("[info] --skip-2d: 2D branch (object_renderer + clip_image) disabled")

    print("=" * TABLE_W)
    print("Ges3ViG layer/module latency profiler")
    print(f"  device      : {device}"
          + (f" ({torch.cuda.get_device_name(0)})" if use_cuda else ""))
    print(f"  jetson      : {is_jetson()}")
    print(f"  torch       : {torch.__version__}")
    print(f"  data cfg    : {args.data}  (lang_dataset={cfg.data.lang_dataset})")
    print(f"  mode        : {args.mode}")
    print(f"  warmup/reps : {args.warmup} / {args.repeats}")
    print("=" * 92)

    model = build_model(cfg, device, args.ckpt)

    if need_stub and hasattr(model, "object_renderer"):
        stub_object_renderer(model, cfg)
        print("[info] object_renderer rasterization STUBBED (random images); "
              "clip_image runs for real")

    # MinkowskiEngine + fp16 don't mix on many builds -> disable fp16 if present
    if args.fp16 and any(type(m).__module__.startswith("MinkowskiEngine")
                         for m in model.modules()):
        print("[warn] MinkowskiEngine present -> disabling --fp16 to avoid dtype errors")
        args.fp16 = False

    if args.mode == "modules":
        run_modules_mode(model, cfg, device, args)
    elif args.mode == "sweep":
        run_sweep_mode(model, cfg, device, args)
    else:
        run_full_mode(model, cfg, device, args)


if __name__ == "__main__":
    main()

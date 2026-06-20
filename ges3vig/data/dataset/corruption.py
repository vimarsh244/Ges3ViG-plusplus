"""
Point-cloud corruption for robustness evaluation.

This module emulates two real-world sensor failure modes on top of the
(otherwise near-complete, fused) ScanNet point clouds the model was trained on:

    Q7 - Gaussian noise over the whole cloud        (`noise`)
    Q8 - partial / incomplete capture (occlusion)   (`partial`)

It is shared by every ImputeRefer variant (ImputeRefer, NoElbowImputeRefer,
NoHumanImputeRefer, ...).  It is a NO-OP unless explicitly enabled via the
`data.eval_corruption` config block, so the default behaviour of every dataset
is completely unchanged.

Design rules
------------
* Corrupt only what the network *sees* - the per-point arrays
  (xyz / rgb / normal / instance_ids / sem_labels).  NEVER touch the ground
  truth (`aabb_corner_xyz`, `aabb_obj_ids`); the GT is the answer key and must
  stay clean or IoU becomes meaningless.
* Be deterministic per (scene_id, seed) so a run is reproducible and two
  checkpoints can be compared on the exact same corruption.
* Use a *local* RandomState so we never disturb the global RNG that drives
  human-mesh sampling and scene augmentation.
* Preserve the instance-label invariant the loaders rely on
  (`assert index == i` over consecutive instance ids): never let a corruption
  delete every point of an instance - always keep at least one.
"""

import numpy as np


# per-point arrays that must be masked together when points are dropped
_PER_POINT_KEYS = ("xyz", "rgb", "normal", "instance_ids", "sem_labels")


def _scene_rng(cfg, scene_id):
    """Deterministic, scene-specific RandomState that does not touch global RNG."""
    base_seed = int(cfg.get("seed", 42))
    scene_hash = abs(hash(str(scene_id))) % (2 ** 31)
    return np.random.RandomState((base_seed + scene_hash) % (2 ** 32))


def _is_enabled(cfg, split):
    if cfg is None:
        return False
    if not cfg.get("enabled", False):
        return False
    # never corrupt the training set unless explicitly asked
    if split == "train" and not cfg.get("apply_on_train", False):
        return False
    return True


def _apply_noise(scene_data, noise_cfg, rng):
    """Q7: additive Gaussian noise on geometry (and optionally colour)."""
    sigma = float(noise_cfg.get("sigma", 0.0))
    mu = float(noise_cfg.get("mu", 0.0))
    mode = str(noise_cfg.get("mode", "isotropic"))
    rgb_sigma = float(noise_cfg.get("rgb_sigma", 0.0))

    xyz = scene_data["xyz"]
    n = xyz.shape[0]

    if sigma > 0.0 or mu != 0.0:
        if mode == "normal" and "normal" in scene_data:
            # noise along the surface normal ~ depth/ray-direction error
            offset = (mu + rng.randn(n, 1) * sigma) * scene_data["normal"]
        else:
            # isotropic per-coordinate jitter
            offset = mu + rng.randn(n, 3) * sigma
        scene_data["xyz"] = (xyz + offset).astype(np.float32)

    if rgb_sigma > 0.0 and "rgb" in scene_data:
        rgb = scene_data["rgb"] + rng.randn(*scene_data["rgb"].shape) * rgb_sigma
        # rgb lives in [-1, 1] in this pipeline
        scene_data["rgb"] = np.clip(rgb, -1.0, 1.0).astype(np.float32)

    return scene_data


def _protect_instances(keep_mask, instance_ids, rng):
    """
    Guarantee every instance present in the scene keeps >= 1 point so the
    downstream `assert index == i` over consecutive instance ids never trips.
    A single surviving point is enough to keep the id alive while still
    destroying the object geometrically (which is the point of the test).
    """
    unique_ids = np.unique(instance_ids)
    unique_ids = unique_ids[unique_ids != -1]
    for inst_id in unique_ids:
        inst_idx = np.where(instance_ids == inst_id)[0]
        if not keep_mask[inst_idx].any():
            keep_mask[rng.choice(inst_idx)] = True
    return keep_mask


def _mask_dropout(scene_data, partial_cfg, rng):
    keep_ratio = float(partial_cfg.get("keep_ratio", 1.0))
    n = scene_data["xyz"].shape[0]
    return rng.random_sample(n) < keep_ratio


def _mask_frustum(scene_data, partial_cfg, rng):
    """Keep only points inside a single random camera view frustum (cone)."""
    xyz = scene_data["xyz"]
    centroid = xyz.mean(axis=0)
    extent = xyz.max(axis=0) - xyz.min(axis=0)

    # place the camera just outside the scene on a random horizontal bearing,
    # at a human-ish height, looking toward the scene centroid
    theta = rng.uniform(0.0, 2.0 * np.pi)
    radius = 0.5 * np.linalg.norm(extent[:2]) + 1e-6
    cam = centroid + np.array([radius * np.cos(theta), radius * np.sin(theta), 0.0])
    view_dir = centroid - cam
    view_dir /= (np.linalg.norm(view_dir) + 1e-9)

    rays = xyz - cam
    dist = np.linalg.norm(rays, axis=1) + 1e-9
    cos_ang = (rays @ view_dir) / dist

    half_fov = np.deg2rad(float(partial_cfg.get("fov_deg", 90.0))) / 2.0
    keep = cos_ang >= np.cos(half_fov)

    max_range = partial_cfg.get("max_range", None)
    if max_range is not None:
        keep &= dist <= float(max_range)
    return keep


def _mask_occlusion(scene_data, partial_cfg, rng):
    """Remove points inside one or more random axis-aligned occluder boxes."""
    xyz = scene_data["xyz"]
    lo = xyz.min(axis=0)
    extent = xyz.max(axis=0) - lo
    num_boxes = int(partial_cfg.get("num_boxes", 1))
    box_frac = float(partial_cfg.get("box_frac", 0.2))

    remove = np.zeros(xyz.shape[0], dtype=bool)
    box_size = extent * box_frac
    for _ in range(num_boxes):
        center = lo + rng.random_sample(3) * extent
        bmin = center - box_size / 2.0
        bmax = center + box_size / 2.0
        inside = np.all((xyz >= bmin) & (xyz <= bmax), axis=1)
        remove |= inside
    return ~remove


_PARTIAL_MODES = {
    "dropout": _mask_dropout,
    "frustum": _mask_frustum,
    "occlusion": _mask_occlusion,
}


def _apply_partial(scene_data, partial_cfg, rng):
    """Q8: keep only a subset of points (occlusion / incomplete capture)."""
    mode = str(partial_cfg.get("mode", "dropout"))
    if mode not in _PARTIAL_MODES:
        raise ValueError(
            f"Unknown eval_corruption.partial.mode '{mode}'. "
            f"Choose one of {sorted(_PARTIAL_MODES)}."
        )
    keep_mask = _PARTIAL_MODES[mode](scene_data, partial_cfg, rng)
    keep_mask = _protect_instances(keep_mask, scene_data["instance_ids"], rng)

    for key in _PER_POINT_KEYS:
        if key in scene_data:
            scene_data[key] = scene_data[key][keep_mask]
    return scene_data


def apply_pointcloud_corruption(scene_data, cfg, split, scene_id, use_multiview=False):
    """
    Apply configured corruption to ``scene_data`` IN PLACE and return it.

    Parameters
    ----------
    scene_data : dict
        Mutable per-scene dict already holding the (human-injected) point cloud,
        i.e. with keys xyz / rgb / normal / instance_ids / sem_labels and the
        untouched GT keys aabb_corner_xyz / aabb_obj_ids.
    cfg : DictConfig | dict | None
        The ``data.eval_corruption`` block.  When ``None`` or ``enabled=False``
        this is a no-op and ``scene_data`` is returned unchanged.
    split : str
        "train" / "val" / "test".  Training is never corrupted unless
        ``apply_on_train`` is set.
    scene_id : str
        Used to seed a deterministic, scene-specific RNG.
    use_multiview : bool
        Whether the model consumes precomputed per-point multiview features.
        Point-dropping corruption is incompatible with these (the feature table
        is indexed by original point order), so we fail loudly instead of
        silently misaligning features.
    """
    if not _is_enabled(cfg, split):
        return scene_data

    rng = _scene_rng(cfg, scene_id)

    noise_cfg = cfg.get("noise", None)
    if noise_cfg is not None and noise_cfg.get("enabled", False):
        scene_data = _apply_noise(scene_data, noise_cfg, rng)

    partial_cfg = cfg.get("partial", None)
    if partial_cfg is not None and partial_cfg.get("enabled", False):
        if use_multiview:
            raise RuntimeError(
                "eval_corruption.partial drops points, which desyncs the "
                "precomputed per-point multiview features. Disable "
                "point_features.use_multiview when using partial-capture corruption."
            )
        scene_data = _apply_partial(scene_data, partial_cfg, rng)

    return scene_data

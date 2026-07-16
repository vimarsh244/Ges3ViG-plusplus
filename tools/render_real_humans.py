"""Render the ImputeRefer++real human point clouds to figure-ready PNGs.

Reads the cleaned real-human assets (humans_real layout: <person>/meshes/<angle>/pose_<code>.ply)
and writes:
  individual/   one PNG per (person, angle, pose)
  contact_sheets/  per-person grids (angle x pose) for eyeballing the whole set
  samples/      4-panel figure images

Panels are titled by pointing hand + elbow state + arm angle, since every capture
is the same build. Views are isotropic (no axis distortion) and share one bounding
box within a figure, so relative heights across panels are true to the data.
"""

import argparse
import os

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from scipy.spatial import ConvexHull

DEFAULT_HUMAN_DIR = "/nas_hcfc/HRI_general/common/dataset/humans_real"
DEFAULT_OUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "output",
    "real_human_images",
)

# Pose code -> (pointing hand, elbow state). Codes come from the capture naming
# convention: s/b = straight/bent, lh = left hand, h = right hand.
POSE_LABELS = {
    "sh": ("Right Hand", "Straight"),
    "bh": ("Right Hand", "Bent Elbow"),
    "lh": ("Left Hand", "Straight"),
    "blh": ("Left Hand", "Bent Elbow"),
}
POSE_ORDER = ["sh", "bh", "lh", "blh"]

# The asset prep aligns each capture's pointing direction to +X, so viewing from
# near +/-Y always puts the pointing arm across the frame rather than
# foreshortened down the view axis. Body yaw, however, is NOT normalised (it
# depends on how each subject stood relative to their own arm), so a single
# fixed azimuth leaves some captures edge-on. auto_view_azimuth() searches a
# window around the arm-in-plane view for the widest body silhouette, then picks
# which of the two opposing sides faces the camera.
VIEW_AZIM = 90.0
VIEW_ELEV = 4.0
VIEW_ZOOM = 1.25
VIEW_WINDOW = 35.0


def pose_code_from_name(pose_name):
    return pose_name.replace("pose_", "").split("_")[0]


def panel_title(pose_name, angle):
    code = pose_code_from_name(pose_name)
    hand, elbow = POSE_LABELS.get(code, ("Unknown", "Unknown"))
    return f"{hand}, {elbow}\n({angle}°)"


def load_points(path):
    """Load a capture as (points, colors). Assets are written as meshes with no
    faces, so fall back to vertices when the point-cloud reader yields nothing."""
    pcd = o3d.io.read_point_cloud(path)
    pts = np.asarray(pcd.points, dtype=np.float64)
    cols = np.asarray(pcd.colors, dtype=np.float64)
    if pts.shape[0] == 0:
        mesh = o3d.io.read_triangle_mesh(path)
        pts = np.asarray(mesh.vertices, dtype=np.float64)
        cols = np.asarray(mesh.vertex_colors, dtype=np.float64)
    if cols.shape != pts.shape:
        cols = np.full(pts.shape, 0.5, dtype=np.float64)
    return pts, np.clip(cols, 0.0, 1.0)


def discover_assets(human_dir, people=None):
    """Yield (person, angle, pose_name, ply_path) for every capture on disk."""
    assets = []
    for person in sorted(os.listdir(human_dir)):
        mesh_root = os.path.join(human_dir, person, "meshes")
        if not os.path.isdir(mesh_root):
            continue
        if people and person.lower() not in people:
            continue
        for angle_name in sorted(os.listdir(mesh_root), key=lambda a: int(a) if a.isdigit() else 0):
            angle_dir = os.path.join(mesh_root, angle_name)
            if not os.path.isdir(angle_dir) or not angle_name.isdigit():
                continue
            for f in sorted(os.listdir(angle_dir)):
                if f.endswith(".ply"):
                    assets.append(
                        (person, int(angle_name), os.path.splitext(f)[0], os.path.join(angle_dir, f))
                    )
    return assets


def _silhouette_area(pts, azim):
    """Area of the point cloud's outline as seen from `azim` (elev ~ 0)."""
    a = np.deg2rad(azim)
    screen_x = -np.sin(a) * pts[:, 0] + np.cos(a) * pts[:, 1]
    try:
        return ConvexHull(np.stack([screen_x, pts[:, 2]], axis=1)).volume
    except Exception:
        return 0.0


def _face_azimuth(pts, cols):
    """Rough direction the head faces, from where the warm (skin) colour sits.

    The back of a head is hair -- dark and blue-ish; the face is warm. Weighting
    head points by (R - B) and taking their offset from the head centre gives a
    usable front direction. Returns None when the cue is too weak to trust.
    """
    z = pts[:, 2]
    head = z >= z.min() + 0.88 * (z.max() - z.min())
    if head.sum() < 30:
        return None
    head_pts, head_cols = pts[head], cols[head]
    skin = np.clip(head_cols[:, 0] - head_cols[:, 2], 0.0, None)
    if skin.sum() < 1e-6:
        return None
    center = head_pts[:, :2].mean(axis=0)
    v = (skin[:, None] * (head_pts[:, :2] - center[None, :])).sum(axis=0) / skin.sum()
    if np.linalg.norm(v) < 1e-9:
        return None
    return float(np.rad2deg(np.arctan2(v[1], v[0])))


def _angle_gap(a, b):
    return abs((a - b + 180.0) % 360.0 - 180.0)


def auto_view_azimuth(pts, cols, window=VIEW_WINDOW):
    """Azimuth that keeps the pointing arm in-plane and opens up the body."""
    offsets = np.arange(-window, window + 1e-6, 2.5)
    areas = [_silhouette_area(pts, VIEW_AZIM + t) for t in offsets]
    best = float(offsets[int(np.argmax(areas))])

    # Opposite sides give an identical silhouette; break the tie with the face.
    near, far = VIEW_AZIM + best, VIEW_AZIM + 180.0 + best
    face = _face_azimuth(pts, cols)
    if face is None:
        return near
    return near if _angle_gap(near, face) <= _angle_gap(far, face) else far


def shared_bounds(point_sets, pad=1.04):
    """One isotropic box covering every cloud, so panels stay comparable."""
    lo = np.min([p.min(axis=0) for p in point_sets], axis=0)
    hi = np.max([p.max(axis=0) for p in point_sets], axis=0)
    center = (lo + hi) / 2.0
    extent = hi - lo
    xy_span = max(extent[0], extent[1]) * pad
    z_span = extent[2] * pad
    return center, xy_span, z_span


def draw_panel(ax, pts, cols, center, xy_span, z_span, point_size,
               azim=VIEW_AZIM, elev=VIEW_ELEV, zoom=VIEW_ZOOM):
    ax.scatter(
        pts[:, 0], pts[:, 1], pts[:, 2],
        c=cols, s=point_size, marker=".", linewidths=0, depthshade=False,
    )
    ax.set_xlim(center[0] - xy_span / 2, center[0] + xy_span / 2)
    ax.set_ylim(center[1] - xy_span / 2, center[1] + xy_span / 2)
    ax.set_zlim(center[2] - z_span / 2, center[2] + z_span / 2)
    # Box aspect proportional to world spans keeps the render isotropic; zoom
    # claws back the wide margin mpl reserves around a 3D axes.
    ax.set_box_aspect((xy_span, xy_span, z_span), zoom=zoom)
    ax.set_proj_type("ortho")
    ax.view_init(elev=elev, azim=azim)
    ax.grid(False)
    ax.set_axis_off()


def render_individual(assets, out_dir, point_size, dpi, azim=None):
    os.makedirs(out_dir, exist_ok=True)
    written = []
    for person, angle, pose_name, path in assets:
        pts, cols = load_points(path)
        view = azim if azim is not None else auto_view_azimuth(pts, cols)
        center, xy_span, z_span = shared_bounds([pts])
        fig = plt.figure(figsize=(3.0, 3.6))
        ax = fig.add_subplot(111, projection="3d")
        draw_panel(ax, pts, cols, center, xy_span, z_span, point_size, azim=view)
        ax.set_title(panel_title(pose_name, angle), fontsize=13, fontweight="bold", y=1.0)
        fig.subplots_adjust(left=0.01, right=0.99, top=0.9, bottom=0.01)
        out = os.path.join(out_dir, f"{person}_{angle}_{pose_name.replace('pose_', '')}.png")
        fig.savefig(out, dpi=dpi, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        written.append(out)
    return written


def render_contact_sheet(assets, person, out_path, point_size, dpi, azim=None):
    """Grid of angle (rows) x pose (cols) for one person."""
    rows = sorted({a for p, a, _, _ in assets if p == person})
    # Retakes (pose_sh_2) share a cell with the primary take; keep the first.
    entries = {}
    for p, a, n, path in assets:
        if p == person:
            entries.setdefault((a, pose_code_from_name(n)), path)
    clouds = {k: load_points(v) for k, v in entries.items()}
    if not clouds:
        return None
    center, xy_span, z_span = shared_bounds([c[0] for c in clouds.values()])

    fig = plt.figure(figsize=(3.0 * len(POSE_ORDER), 3.6 * len(rows)))
    for r, angle in enumerate(rows):
        for c, code in enumerate(POSE_ORDER):
            key = (angle, code)
            if key not in clouds:
                continue
            ax = fig.add_subplot(len(rows), len(POSE_ORDER), r * len(POSE_ORDER) + c + 1, projection="3d")
            pts, cols = clouds[key]
            view = azim if azim is not None else auto_view_azimuth(pts, cols)
            draw_panel(ax, pts, cols, center, xy_span, z_span, point_size, azim=view)
            hand, elbow = POSE_LABELS[code]
            ax.set_title(f"{hand}, {elbow}\n({angle}°)", fontsize=12, fontweight="bold", y=1.0)
    fig.suptitle(f"ImputeRefer++real — {person}", fontsize=17, fontweight="bold", y=1.0)
    fig.subplots_adjust(wspace=0.0, hspace=0.2, left=0.01, right=0.99, top=0.95, bottom=0.01)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path


def render_sample(selection, human_dir, out_path, point_size, dpi, azim=None):
    """4-panel figure. `selection` is a list of (person, angle, pose_code)."""
    clouds, titles = [], []
    for person, angle, code in selection:
        path = os.path.join(human_dir, person, "meshes", str(angle), f"pose_{code}.ply")
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
        clouds.append(load_points(path))
        titles.append(panel_title(f"pose_{code}", angle))

    # One box for every panel keeps relative heights honest across the figure.
    center, xy_span, z_span = shared_bounds([c[0] for c in clouds])
    fig = plt.figure(figsize=(3.2 * len(clouds), 4.0))
    for i, ((pts, cols), title) in enumerate(zip(clouds, titles)):
        ax = fig.add_subplot(1, len(clouds), i + 1, projection="3d")
        view = azim if azim is not None else auto_view_azimuth(pts, cols)
        draw_panel(ax, pts, cols, center, xy_span, z_span, point_size, azim=view)
        ax.set_title(title, fontsize=15, fontweight="bold", y=1.0)
    fig.subplots_adjust(wspace=0.0, left=0.01, right=0.99, top=0.88, bottom=0.01)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path


# Figure-ready 4-panel selections, each (person, angle, pose_code).
SAMPLE_FIGURES = {
    # One capture per pointing hand / elbow state, each at a different arm angle.
    "sample_pose_variation": [
        ("parth", 30, "sh"),
        ("nilesh", 60, "bh"),
        ("saransh", 90, "lh"),
        ("grace", 120, "blh"),
    ],
    # Same subject and pose, sweeping the arm angle.
    "sample_angle_sweep": [
        ("parth", 30, "sh"),
        ("parth", 60, "sh"),
        ("parth", 90, "sh"),
        ("parth", 120, "sh"),
    ],
    # Same subject and angle, all four hand / elbow combinations.
    "sample_hand_elbow": [
        ("nilesh", 90, "sh"),
        ("nilesh", 90, "bh"),
        ("nilesh", 90, "lh"),
        ("nilesh", 90, "blh"),
    ],
    # Four different subjects, to show identity variation across the capture set.
    "sample_subjects": [
        ("parth", 90, "sh"),
        ("nilesh", 90, "sh"),
        ("saransh", 90, "sh"),
        ("grace", 90, "sh"),
    ],
}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--human-dir", default=DEFAULT_HUMAN_DIR)
    ap.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    ap.add_argument("--people", nargs="*", default=None, help="restrict to these identities")
    ap.add_argument("--point-size", type=float, default=5.0)
    ap.add_argument("--dpi", type=int, default=150)
    ap.add_argument("--azim", type=float, default=None,
                    help="force a fixed view azimuth instead of picking one per capture")
    ap.add_argument("--skip-individual", action="store_true")
    ap.add_argument("--skip-contact-sheets", action="store_true")
    ap.add_argument("--skip-samples", action="store_true")
    args = ap.parse_args()

    people = [p.lower() for p in args.people] if args.people else None
    assets = discover_assets(args.human_dir, people)
    if not assets:
        raise SystemExit(f"no assets found under {args.human_dir}")
    print(f"found {len(assets)} captures")

    os.makedirs(args.out_dir, exist_ok=True)

    if not args.skip_individual:
        written = render_individual(
            assets, os.path.join(args.out_dir, "individual"), args.point_size, args.dpi, args.azim
        )
        print(f"individual: {len(written)} PNGs")

    if not args.skip_contact_sheets:
        sheet_dir = os.path.join(args.out_dir, "contact_sheets")
        os.makedirs(sheet_dir, exist_ok=True)
        for person in sorted({p for p, _, _, _ in assets}):
            out = render_contact_sheet(
                assets, person, os.path.join(sheet_dir, f"{person}.png"),
                args.point_size, args.dpi, args.azim,
            )
            if out:
                print(f"contact sheet: {out}")

    if not args.skip_samples:
        sample_dir = os.path.join(args.out_dir, "samples")
        os.makedirs(sample_dir, exist_ok=True)
        for name, selection in SAMPLE_FIGURES.items():
            out = render_sample(
                selection, args.human_dir, os.path.join(sample_dir, f"{name}.png"),
                args.point_size, args.dpi, args.azim,
            )
            print(f"sample: {out}")


if __name__ == "__main__":
    main()

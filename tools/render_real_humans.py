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
import json
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
VIEW_ELEV = 0.0
VIEW_WINDOW = 35.0

# Estimated pointing arm overlay (shoulder -> elbow -> wrist -> index).
ARM_COLOR = "#e8112d"
ARM_WIDTH = 4.0
ARM_JOINTS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "real_human_arm_joints.json")


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


def _panel_grid(nrows, ncols, extent, panel_h=4.4, title_band=0.62):
    """Figure whose panels match the data box aspect, with a clear title band.

    Sizing each panel to extent[x]:extent[z] means the subject fills it instead
    of floating in padding, and the band above every row keeps titles clear of
    raised arms.
    """
    aspect = float(extent[0]) / float(extent[2])
    panel_w = panel_h * aspect
    fig_w, fig_h = ncols * panel_w, nrows * (panel_h + title_band)
    fig = plt.figure(figsize=(fig_w, fig_h))
    axes = [fig.add_subplot(nrows, ncols, i + 1) for i in range(nrows * ncols)]
    fig.subplots_adjust(
        left=0.005, right=0.995, bottom=0.005,
        top=1.0 - title_band / fig_h,
        wspace=0.0, hspace=title_band / panel_h,
    )
    return fig, axes


def load_arm(person, angle, pose_stem):
    """Estimated shoulder->elbow->wrist->index polyline for a capture, or None.

    Produced by tools/estimate_real_human_arms.py, which recovers the arm from
    the point cloud with a pose detector. The joints shipped in humans_real are
    placeholder constants (see the README), so the overlay is never drawn from
    them. Keyed by the full stem ("sh_2") to keep retakes distinct.
    """
    entry = _load_arm_table().get(f"{person}/{angle}/{pose_stem}")
    return np.asarray(entry["polyline"], dtype=np.float64) if entry else None


_ARM_TABLE = None


def _load_arm_table():
    global _ARM_TABLE
    if _ARM_TABLE is None:
        if os.path.isfile(ARM_JOINTS_PATH):
            with open(ARM_JOINTS_PATH) as f:
                _ARM_TABLE = json.load(f)
        else:
            _ARM_TABLE = {}
    return _ARM_TABLE


def save_figure(fig, out_base, formats, dpi):
    """Write `fig` once per requested format. PDF/SVG come out as vector."""
    written = []
    for ext in formats:
        out = f"{out_base}.{ext}"
        fig.savefig(out, dpi=dpi, bbox_inches="tight", facecolor="white")
        written.append(out)
    plt.close(fig)
    return written


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


def rotate_to_canonical_view(pts, azim):
    """Spin the cloud about Z so that viewing it at VIEW_AZIM shows it as it
    would look from `azim`.

    Rendering every panel from one known azimuth is what makes tight framing
    possible: the projected width is then exactly the cloud's x extent, so the
    box can hug the data instead of padding for an unknown rotation. The subjects
    are anchored at the shoulder on the Z axis, so this rotation leaves them
    aligned with each other.
    """
    delta = np.deg2rad(VIEW_AZIM - azim)
    c, s = np.cos(delta), np.sin(delta)
    rot = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    return pts @ rot.T


def shared_bounds(point_sets, pad=1.03):
    """One isotropic box covering every (already canonically rotated) cloud.

    Every panel shares it, so heights stay comparable across a figure. Returns
    the box centre and its x/y/z spans; at VIEW_AZIM the panel shows x by z.
    """
    lo = np.min([p.min(axis=0) for p in point_sets], axis=0)
    hi = np.max([p.max(axis=0) for p in point_sets], axis=0)
    center = (lo + hi) / 2.0
    extent = (hi - lo) * pad
    # Depth never reaches the screen at this view, but keep it non-degenerate so
    # the box aspect stays a valid 3-vector.
    extent[1] = max(extent[1], 1e-3)
    return center, extent


def project(pts):
    """Orthographic projection of a canonically rotated cloud to screen (x, y).

    Clouds are pre-rotated so the camera sits on +Y looking down -Y, so the
    projection collapses to (-x, z) and depth is y. Doing this by hand in a 2D
    axes -- rather than leaning on mpl's 3D axes -- is what gives exact framing:
    the panel shows precisely the requested box, with no hidden margin.
    """
    pts = np.asarray(pts, dtype=np.float64)
    return -pts[:, 0], pts[:, 2], pts[:, 1]


def draw_panel(ax, pts, cols, center, extent, point_size, arm=None):
    sx, sy, depth = project(pts)
    order = np.argsort(depth)  # painter's algorithm: nearer points drawn last
    ax.scatter(
        sx[order], sy[order], c=np.clip(cols[order], 0, 1),
        s=point_size, marker=".", linewidths=0,
    )
    if arm is not None and len(arm) >= 2:
        ax_, ay_, _ = project(arm)
        ax.plot(ax_, ay_, color=ARM_COLOR, linewidth=ARM_WIDTH,
                solid_capstyle="round", solid_joinstyle="round", zorder=10)
        ax.scatter(ax_, ay_, c=ARM_COLOR, s=ARM_WIDTH ** 2 * 2.2,
                   marker="o", linewidths=0, zorder=11)

    cx = -center[0]
    ax.set_xlim(cx - extent[0] / 2, cx + extent[0] / 2)
    ax.set_ylim(center[2] - extent[2] / 2, center[2] + extent[2] / 2)
    ax.set_aspect("equal", adjustable="box")
    ax.set_axis_off()


def render_individual(assets, out_dir, point_size, dpi, azim=None, formats=("png",)):
    os.makedirs(out_dir, exist_ok=True)
    written = []
    for person, angle, pose_name, path in assets:
        pts, cols = load_points(path)
        view = azim if azim is not None else auto_view_azimuth(pts, cols)
        arm = load_arm(person, angle, pose_name.replace("pose_", ""))
        pts = rotate_to_canonical_view(pts, view)
        arm = rotate_to_canonical_view(arm, view) if arm is not None else None
        center, extent = shared_bounds([pts])
        fig, axes = _panel_grid(1, 1, extent, panel_h=3.8, title_band=0.55)
        draw_panel(axes[0], pts, cols, center, extent, point_size, arm=arm)
        axes[0].set_title(panel_title(pose_name, angle), fontsize=13,
                          fontweight="bold", y=1.0)
        base = os.path.join(out_dir, f"{person}_{angle}_{pose_name.replace('pose_', '')}")
        written += save_figure(fig, base, formats, dpi)
    return written


def render_contact_sheet(assets, person, out_base, point_size, dpi, azim=None,
                         formats=("png",)):
    """Grid of angle (rows) x pose (cols) for one person."""
    rows = sorted({a for p, a, _, _ in assets if p == person})
    # Retakes (pose_sh_2) share a cell with the primary take; keep the first.
    entries = {}
    for p, a, n, path in assets:
        if p == person:
            entries.setdefault((a, pose_code_from_name(n)), (path, n))
    clouds = {}
    for (angle, code), (path, pose_name) in entries.items():
        pts, cols = load_points(path)
        view = azim if azim is not None else auto_view_azimuth(pts, cols)
        arm = load_arm(person, angle, pose_name.replace("pose_", ""))
        clouds[(angle, code)] = (
            rotate_to_canonical_view(pts, view),
            cols,
            rotate_to_canonical_view(arm, view) if arm is not None else None,
        )
    if not clouds:
        return None
    center, extent = shared_bounds([c[0] for c in clouds.values()])

    fig, axes = _panel_grid(len(rows), len(POSE_ORDER), extent)
    for r, angle in enumerate(rows):
        for c, code in enumerate(POSE_ORDER):
            ax = axes[r * len(POSE_ORDER) + c]
            if (angle, code) not in clouds:
                ax.set_axis_off()
                continue
            pts, cols, arm = clouds[(angle, code)]
            draw_panel(ax, pts, cols, center, extent, point_size, arm=arm)
            hand, elbow = POSE_LABELS[code]
            ax.set_title(f"{hand}, {elbow}\n({angle}°)", fontsize=12, fontweight="bold", y=1.0)
    return save_figure(fig, out_base, formats, dpi)


def render_sample(selection, human_dir, out_base, point_size, dpi, azim=None, formats=("png",)):
    """4-panel figure. `selection` is a list of (person, angle, pose_code)."""
    panels, titles = [], []
    for person, angle, code in selection:
        path = os.path.join(human_dir, person, "meshes", str(angle), f"pose_{code}.ply")
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
        pts, cols = load_points(path)
        view = azim if azim is not None else auto_view_azimuth(pts, cols)
        arm = load_arm(person, angle, code)
        panels.append((
            rotate_to_canonical_view(pts, view),
            cols,
            rotate_to_canonical_view(arm, view) if arm is not None else None,
        ))
        titles.append(panel_title(f"pose_{code}", angle))

    # One box for every panel keeps relative heights honest across the figure.
    center, extent = shared_bounds([p[0] for p in panels])
    fig, axes = _panel_grid(1, len(panels), extent)
    for ax, (pts, cols, arm), title in zip(axes, panels, titles):
        draw_panel(ax, pts, cols, center, extent, point_size, arm=arm)
        ax.set_title(title, fontsize=15, fontweight="bold", y=1.0)
    return save_figure(fig, out_base, formats, dpi)


# Figure-ready 4-panel selections, each (person, angle, pose_code).
SAMPLE_FIGURES = {
    # Four different subjects, one per hand/elbow combination (left pair then
    # right pair), each at a different arm angle.
    # nilesh is deliberately avoided here: their bent-elbow captures lose the
    # lower legs (bh/120 keeps only 18% of points in the bottom third).
    "sample_pose_variation": [
        ("vim", 30, "lh"),
        ("grace", 60, "blh"),
        ("parth", 90, "sh"),
        ("saransh", 120, "bh"),
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
    ap.add_argument("--point-size", type=float, default=12.0)
    ap.add_argument("--dpi", type=int, default=150)
    ap.add_argument("--azim", type=float, default=None,
                    help="force a fixed view azimuth instead of picking one per capture")
    ap.add_argument("--formats", nargs="+", default=["png", "pdf"],
                    help="output formats, e.g. png pdf svg (pdf/svg are vector)")
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

    formats = [f.lower().lstrip(".") for f in args.formats]

    if not args.skip_individual:
        written = render_individual(
            assets, os.path.join(args.out_dir, "individual"),
            args.point_size, args.dpi, args.azim, formats,
        )
        print(f"individual: {len(written)} files")

    if not args.skip_contact_sheets:
        sheet_dir = os.path.join(args.out_dir, "contact_sheets")
        os.makedirs(sheet_dir, exist_ok=True)
        for person in sorted({p for p, _, _, _ in assets}):
            out = render_contact_sheet(
                assets, person, os.path.join(sheet_dir, person),
                args.point_size, args.dpi, args.azim, formats,
            )
            if out:
                print(f"contact sheet: {', '.join(out)}")

    if not args.skip_samples:
        sample_dir = os.path.join(args.out_dir, "samples")
        os.makedirs(sample_dir, exist_ok=True)
        for name, selection in SAMPLE_FIGURES.items():
            out = render_sample(
                selection, args.human_dir, os.path.join(sample_dir, name),
                args.point_size, args.dpi, args.azim, formats,
            )
            print(f"sample: {', '.join(out)}")


if __name__ == "__main__":
    main()

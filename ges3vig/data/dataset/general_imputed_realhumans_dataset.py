import copy
import hashlib
import os
import pickle
import random

import MinkowskiEngine as ME
import numpy as np
import open3d as o3d
import torch
from torch.utils.data import Dataset

from ges3vig.data.dataset.general_imputed_multi_dataset import (
    GeneralMultiImputedDataset,
)

DEFAULT_REAL_NAMES = ["grace", "parth", "nilesh", "saransh", "vim"]
POSE_SIDE_BY_CODE = {
    "lh": "left",
    "blh": "left",
    "sh": "right",
    "bh": "right",
}
SYNTHETIC_TYPES = [
    "man_fat_black",
    "man_fat_white",
    "man_medium_black",
    "man_medium_white",
    "man_tall_black",
    "man_tall_white",
    "woman_fat_black",
    "woman_fat_white",
    "woman_medium_black",
    "woman_medium_white",
    "woman_tall_black",
    "woman_tall_white",
]

# Canonical SMPL-X 55-joint order. This is exactly the order the synthetic
# humans store their joints in, and therefore the order the JointRegressor
# (num_joints=55) was trained to output. Real-human assets only provide the 10
# arm joints; we must place them at their canonical indices so that
#   - the human-joint loss lines up with the 55-wide regressor output, and
#   - joint_name_idx_map[...] indexes the right joint in output_dict["joints"]
#     for the pointing-bias / distance code (e.g. left_shoulder -> 16).
SMPLX_JOINT_ORDER = [
    "pelvis", "left_hip", "right_hip", "spine1", "left_knee", "right_knee",
    "spine2", "left_ankle", "right_ankle", "spine3", "left_foot", "right_foot",
    "neck", "left_collar", "right_collar", "head", "left_shoulder",
    "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist",
    "jaw", "left_eye_smplhf", "right_eye_smplhf", "left_index1", "left_index2",
    "left_index3", "left_middle1", "left_middle2", "left_middle3", "left_pinky1",
    "left_pinky2", "left_pinky3", "left_ring1", "left_ring2", "left_ring3",
    "left_thumb1", "left_thumb2", "left_thumb3", "right_index1", "right_index2",
    "right_index3", "right_middle1", "right_middle2", "right_middle3",
    "right_pinky1", "right_pinky2", "right_pinky3", "right_ring1", "right_ring2",
    "right_ring3", "right_thumb1", "right_thumb2", "right_thumb3",
]
SMPLX_JOINT_INDEX = {name: idx for idx, name in enumerate(SMPLX_JOINT_ORDER)}

# 8 AABB corners, in the same order preprocess_all_imputed_data.py builds them.
AABB_CORNER_COMBINATIONS = np.array(
    np.meshgrid([0, 1], [0, 1], [0, 1]), dtype=np.float64
).T.reshape(-1, 3)
NUM_SMPLX_JOINTS = len(SMPLX_JOINT_ORDER)  # 55


def _build_canonical_joint_layout(joints_by_name):
    """Lay real-human arm joints into the canonical 55-joint SMPL-X frame.

    Returns (joint_matrix[1,55,3], joint_name_idx_map{name->canonical idx},
    joint_mask[1,55]) where the mask is 1 only on joints we actually have, so
    the joint-regression loss ignores the 45 unobserved joints (zeros).
    """
    joint_matrix = np.zeros((1, NUM_SMPLX_JOINTS, 3), dtype=np.float64)
    joint_mask = np.zeros((1, NUM_SMPLX_JOINTS), dtype=np.float32)
    for name, coord in joints_by_name.items():
        idx = SMPLX_JOINT_INDEX.get(name)
        if idx is None:  # e.g. bare 'left_index'/'right_index' not in SMPL-X 55
            continue
        joint_matrix[0, idx] = np.asarray(coord, dtype=np.float64)
        joint_mask[0, idx] = 1.0
    return joint_matrix, dict(SMPLX_JOINT_INDEX), joint_mask


class GeneralImputedRealHumansDataset(GeneralMultiImputedDataset):
    """
    Imputed dataset base that uses cleaned real-human assets (humans-like folder layout)
    while keeping the same scene/language format as GeneralMultiImputedDataset.
    """

    def __init__(self, data_cfg=None, split="train"):
        Dataset.__init__(self)
        self.data_cfg = data_cfg
        self.split = split

        configured_names = list(
            getattr(data_cfg, "real_human_names", DEFAULT_REAL_NAMES)
        )
        self.real_human_names = [x.lower() for x in configured_names]
        self.real_human_scale = float(getattr(data_cfg, "real_human_scale", 1.0))

        # Synthetic assets are still needed even though no synthetic body is
        # inserted: every stored scene transform is expressed in the local frame
        # of that type's l_h_100 pose, which is the alignment anchor.
        self.synthetic_human_dir = str(
            getattr(
                data_cfg,
                "synthetic_human_dir",
                os.path.join(data_cfg.data_root_path, "humans"),
            )
        )
        self.height_scale_limits = (
            float(getattr(data_cfg, "real_human_min_scale", 0.72)),
            float(getattr(data_cfg, "real_human_max_scale", 1.30)),
        )
        self._anchor_cache = {}

        self.human_data, self.human_angle_index = self._load_real_human_data(
            data_cfg.human_dir
        )
        if not self.human_data:
            raise RuntimeError(
                f"No real-human assets found under: {data_cfg.human_dir}"
            )

        self.pose_index = self._build_pose_index()

        self.synthetic_to_real = {}
        for idx, synth_name in enumerate(SYNTHETIC_TYPES):
            if self.real_human_names:
                self.synthetic_to_real[synth_name] = self.real_human_names[
                    idx % len(self.real_human_names)
                ]

        self._load_language_data()
        self._load_scene_data()
        self._pack_data_to_chunks()

    @staticmethod
    def _pose_side_from_name(pose_name):
        pose_name_l = str(pose_name).lower()
        stem = pose_name_l.split("pose_")[-1]
        code = stem.split("_")[0]
        return POSE_SIDE_BY_CODE.get(code, "unknown")

    def _load_real_human_data(self, human_dir):
        human_data = {}
        angle_index = {}

        if not os.path.isdir(human_dir):
            raise FileNotFoundError(f"human_dir does not exist: {human_dir}")

        for person in sorted(os.listdir(human_dir)):
            person_path = os.path.join(human_dir, person)
            if not os.path.isdir(person_path):
                continue
            if self.real_human_names and person.lower() not in self.real_human_names:
                continue

            mesh_root = os.path.join(person_path, "meshes")
            joints_root = os.path.join(person_path, "joints")
            if not os.path.isdir(mesh_root) or not os.path.isdir(joints_root):
                continue

            person_entries = {}
            person_angles = []

            for angle_name in sorted(os.listdir(mesh_root)):
                mesh_angle_dir = os.path.join(mesh_root, angle_name)
                joints_angle_dir = os.path.join(joints_root, angle_name)
                if not os.path.isdir(mesh_angle_dir) or not os.path.isdir(
                    joints_angle_dir
                ):
                    continue
                try:
                    angle = int(angle_name)
                except ValueError:
                    continue

                pose_entries = []
                for mesh_file in sorted(os.listdir(mesh_angle_dir)):
                    if not mesh_file.endswith(".ply"):
                        continue
                    pose_name = os.path.splitext(mesh_file)[0]
                    joint_file = os.path.join(joints_angle_dir, f"{pose_name}.pkl")
                    if not os.path.isfile(joint_file):
                        continue

                    mesh_path = os.path.join(mesh_angle_dir, mesh_file)
                    pcd = o3d.io.read_point_cloud(mesh_path)
                    points = np.asarray(pcd.points, dtype=np.float64)
                    colors = np.asarray(pcd.colors, dtype=np.float64)
                    normals = np.asarray(pcd.normals, dtype=np.float64)

                    if points.size == 0:
                        mesh = o3d.io.read_triangle_mesh(mesh_path)
                        points = np.asarray(mesh.vertices, dtype=np.float64)
                        colors = np.asarray(mesh.vertex_colors, dtype=np.float64)
                        normals = np.asarray(mesh.vertex_normals, dtype=np.float64)

                    if colors.shape != points.shape:
                        colors = np.full(points.shape, 0.5, dtype=np.float64)
                    if normals.shape != points.shape:
                        normals = np.zeros(points.shape, dtype=np.float64)

                    with open(joint_file, "rb") as f:
                        joints = pickle.load(f)

                    side = self._pose_side_from_name(pose_name)

                    pose_entries.append(
                        {
                            "pose_name": pose_name,
                            "side": side,
                            "points": points,
                            "colors": colors,
                            "normals": normals,
                            "joints": joints,
                        }
                    )

                if pose_entries:
                    person_entries[angle] = pose_entries
                    person_angles.append(angle)

            if person_entries:
                human_data[person] = person_entries
                angle_index[person] = sorted(person_angles)

        return human_data, angle_index

    def _sample_rngs(self, index, scene_id, object_id):
        if self.split == "train":
            return random, np.random

        key = f"{self.split}:{index}:{scene_id}:{object_id}"
        digest = hashlib.blake2b(key.encode("utf-8"), digest_size=8).digest()
        seed = int.from_bytes(digest, byteorder="little", signed=False)
        return random.Random(seed), np.random.default_rng(seed)

    def _synthetic_anchor(self, human_name, side):
        """Shoulder position and pointing yaw of the pose a scene transform assumes.

        ``imputer_full.py`` builds every stored transform from index 0 of the
        type's angle-sorted asset list, which is ``l_h_100`` (mirrored in X for
        the right arm).
        """
        key = (human_name, side)
        if key in self._anchor_cache:
            return self._anchor_cache[key]

        path = os.path.join(
            self.synthetic_human_dir, human_name, "joints", "100", "l_h_100.pkl"
        )
        with open(path, "rb") as f:
            joints = {
                k: np.asarray(v, dtype=np.float64) for k, v in pickle.load(f).items()
            }
        if side == "right":
            mirrored = {}
            for k, v in joints.items():
                v = v.copy()
                v[0] = -v[0]
                if "left" in k:
                    mirrored[k.replace("left", "right")] = v
                elif "right" in k:
                    mirrored[k.replace("right", "left")] = v
                else:
                    mirrored[k] = v
            joints = mirrored

        shoulder = joints[f"{side}_shoulder"]
        direction = joints[f"{side}_index1"] - shoulder
        anchor = (shoulder, float(np.arctan2(direction[1], direction[0])))
        self._anchor_cache[key] = anchor
        return anchor

    def _alignment_transform(self, human_name, side, real_shoulder):
        """4x4 mapping a prepared real human into the synthetic type's local frame.

        Prepared assets (tools/realhumans/prepare_real_humans.py) stand with feet
        on z=0, their pointing shoulder on the z axis and their pointing ray
        yawed to -Y. The stored scene transform instead expects the synthetic
        body's shoulder position and yaw, so without this the wrong point lands
        on the solved floor spot: ~0.19 m lateral and ~0.18 m vertical error,
        which measures as ~12.9 deg of pointing error instead of ~3.6 deg.
        """
        synth_shoulder, synth_yaw = self._synthetic_anchor(human_name, side)

        real_shoulder_height = float(np.asarray(real_shoulder, dtype=np.float64)[2])
        scale = 1.0
        if real_shoulder_height > 1e-3:
            scale = float(
                np.clip(
                    synth_shoulder[2] / real_shoulder_height, *self.height_scale_limits
                )
            )

        # Prepared assets are canonicalised to a -pi/2 pointing yaw.
        delta = synth_yaw + np.pi / 2.0
        c, s = np.cos(delta), np.sin(delta)

        transform = np.eye(4, dtype=np.float64)
        transform[:3, :3] = (
            np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]) * scale
        )
        transform[:3, 3] = synth_shoulder - np.array(
            [0.0, 0.0, scale * real_shoulder_height]
        )
        return transform

    def _resolve_person_name(self, synthetic_name, py_rng=random):
        synthetic_name = str(synthetic_name)
        mapped = self.synthetic_to_real.get(synthetic_name)
        if mapped in self.human_data:
            return mapped
        names = list(self.human_data.keys())
        return names[py_rng.randint(0, len(names) - 1)]

    def _nearest_angle(self, person_name, target_angle):
        available = self.human_angle_index[person_name]
        if not available:
            raise RuntimeError(f"No angles available for person {person_name}")
        return min(available, key=lambda a: abs(a - target_angle))

    def _build_pose_index(self):
        """``{side: {angle_bin: [(person, angle, pose), ...]}}`` pooled over identities."""
        index = {"left": {}, "right": {}}
        for person, angles in self.human_angle_index.items():
            for angle in angles:
                for pose in self.human_data[person][angle]:
                    index[pose["side"]].setdefault(angle, []).append(
                        (person, angle, pose)
                    )
        return {
            side: dict(sorted(by_angle.items())) for side, by_angle in index.items()
        }

    def _select_pose_entry(self, person_name, target_angle, lr, py_rng=random):
        """Nearest-elevation capture on the *correct* arm, pooled over identities.

        Restricting to one identity (the previous behaviour) was doubly harmful:
        each person only covers ~8 of the 13 elevation bins, and when that
        person had no capture on the needed side at that angle the old fallback
        silently returned a wrong-arm pose. Measured over the scene angle
        distribution that was 27.3% of requests, and it is catastrophic here
        because the assets canonicalise the *pointing* shoulder to the origin -
        the other arm is down at the person's side.

        Identity is a free augmentation axis instead: _alignment_transform
        rescales whoever is chosen to the synthetic body's shoulder height.
        ``person_name`` is kept in the signature for API compatibility and is
        only used to bias the choice when several identities tie.
        """
        by_angle = self.pose_index.get(lr) or {}
        if not by_angle:  # no capture at all on this side
            by_angle = self.pose_index["left"] or self.pose_index["right"]
        nearest = min(by_angle, key=lambda a: abs(a - target_angle))
        candidates = by_angle[nearest]

        preferred = [c for c in candidates if c[0] == person_name]
        pool = preferred if preferred else candidates
        person, angle, pose = pool[py_rng.randint(0, len(pool) - 1)]
        return angle, pose

    def _build_dummy_human(self):
        # No human present: still emit a full canonical 55-joint frame (all
        # zeros, fully masked out) so batches with and without humans collate
        # together and the regressor output (55) always lines up.
        return _build_canonical_joint_layout({})

    def __getitem__(self, index):
        data_dict = {}

        scene_id = self.scene_ids[self.chunk_scene_indices[index]]
        object_id = self.language_data[scene_id][self.chunk_lang_indices[index]][
            "object_id"
        ]
        py_rng, np_rng = self._sample_rngs(index, scene_id, object_id)

        scene_data = copy.deepcopy(self.all_scene_data[scene_id])
        human_data = scene_data["human_info"]
        lr = ["left", "right"][py_rng.randint(0, 1)]

        all_humans_data = human_data[object_id]

        if len(all_humans_data["left"]) == 0 and len(all_humans_data["right"]) == 0:
            lr = None
        elif len(all_humans_data["left"]) > 0 and len(all_humans_data["right"]) == 0:
            lr = "left"
        elif len(all_humans_data["right"]) > 0 and len(all_humans_data["left"]) == 0:
            lr = "right"

        human_joints_final = {}
        arm_angle_value = None

        if lr is not None:
            multi_human_lr = all_humans_data[lr]
            human_id = py_rng.randint(0, len(multi_human_lr) - 1)
            human_dict = multi_human_lr[human_id]
            human_transforms = human_dict["axis_aligned_transforms"]
            arm_angles = human_dict["angle_bin"]
            transform_id = py_rng.randint(0, len(human_transforms) - 1)

            scene_human_name = human_dict["human_name"]
            person_name = self._resolve_person_name(scene_human_name, py_rng)

            target_angle = int(round(float(arm_angles[transform_id].item())))
            resolved_angle, pose_entry = self._select_pose_entry(
                person_name, target_angle, lr, py_rng
            )

            human_transform = human_transforms[transform_id]
            human_joints = pose_entry["joints"]
            instance_id = human_dict["instance_id"]
            sem_label = human_dict["sem_label"]
            arm_angle_value = resolved_angle

            shoulder_key = f"{lr}_shoulder"
            if shoulder_key not in human_joints:
                shoulder_key = (
                    "left_shoulder"
                    if "left_shoulder" in human_joints
                    else "right_shoulder"
                )

            # Map the prepared real human into the synthetic type's local frame,
            # which is the frame the stored scene transform was solved in.
            align_transform = self._alignment_transform(
                human_dict["human_name"], lr, human_joints[shoulder_key]
            )

            # The perturbation is applied after alignment, so it must pivot on
            # the *aligned* shoulder (which lands on the synthetic shoulder).
            aligned_shoulder = (
                align_transform
                @ np.append(np.asarray(human_joints[shoulder_key], np.float64), 1.0)
            )[:3]
            human_center_translate = torch.eye(4, dtype=torch.float64)
            human_center_translate[:3, 3] = -torch.tensor(
                aligned_shoulder, dtype=torch.float64
            )

            human_perturbation_range = (
                0.5 * self.data_cfg.human_perturbation_percentage * 2 * torch.pi / 100
            )
            human_perturbation = (
                torch.tensor(
                    py_rng.random() - 0.5,
                    dtype=torch.float64,
                )
                * human_perturbation_range
            )
            perturb_transform = torch.eye(4, dtype=torch.float64)
            perturb_transform[0, 0] = torch.cos(human_perturbation)
            perturb_transform[1, 1] = torch.cos(human_perturbation)
            perturb_transform[0, 1] = torch.sin(human_perturbation)
            perturb_transform[1, 0] = -torch.sin(human_perturbation)
            perturb_transform = (
                human_center_translate.inverse()
                @ perturb_transform
                @ human_center_translate
            ).numpy()
            human_transform = human_transform @ perturb_transform @ align_transform

            raw_points = np.asarray(pose_entry["points"], dtype=np.float64)
            raw_colors = np.asarray(pose_entry["colors"], dtype=np.float64)
            raw_normals = np.asarray(pose_entry["normals"], dtype=np.float64)

            scale_anchor = np.asarray(
                human_joints.get(shoulder_key, np.zeros((3,), dtype=np.float64)),
                dtype=np.float64,
            )
            # NOTE: _alignment_transform already rescales the body to the
            # synthetic shoulder height. Leave real_human_scale at 1.0 (the
            # config default) unless you deliberately want a second, composing
            # scale factor on top of it.
            if self.real_human_scale != 1.0:
                raw_points = (
                    scale_anchor[None, :]
                    + (raw_points - scale_anchor[None, :]) * self.real_human_scale
                )
                human_joints = {
                    name: scale_anchor
                    + (np.asarray(coord, dtype=np.float64) - scale_anchor)
                    * self.real_human_scale
                    for name, coord in human_joints.items()
                }

            if raw_points.shape[0] > 0:
                points_h = np.concatenate(
                    [raw_points, np.ones((raw_points.shape[0], 1), dtype=np.float64)],
                    axis=1,
                )
                human_points = (human_transform @ points_h.T).T[:, :3]
            else:
                human_points = np.empty((0, 3), dtype=np.float64)

            rot = human_transform[:3, :3]
            if raw_normals.shape[0] == raw_points.shape[0] and raw_points.shape[0] > 0:
                human_normals = raw_normals @ rot.T
            else:
                human_normals = np.zeros_like(human_points, dtype=np.float64)

            if raw_colors.shape[0] == raw_points.shape[0] and raw_points.shape[0] > 0:
                human_colors = raw_colors
            else:
                human_colors = np.full((len(human_points), 3), 0.5, dtype=np.float64)

            for name, coord in human_joints.items():
                joint_coord_homo = np.array([0.0, 0.0, 0.0, 1.0], np.float64)
                joint_coord_homo[:3] = np.asarray(coord)
                joint_coord_transformed_homo = human_transform @ joint_coord_homo
                human_joints_final[name] = joint_coord_transformed_homo[:3]

            if len(human_points) > 0:
                human_choices = np_rng.choice(
                    len(human_points), len(human_points), replace=False
                )
                human_points = human_points[human_choices]
                human_rgb = (2 * np.asarray(human_colors) - 1)[human_choices]
                human_normals = np.asarray(human_normals)[human_choices]
            else:
                human_rgb = np.empty((0, 3), dtype=np.float32)
                human_normals = np.empty((0, 3), dtype=np.float32)

            human_instance_ids = np.array(
                [instance_id] * len(human_points), dtype=np.int16
            )
            human_sem_labels = np.array([sem_label] * len(human_points), dtype=np.int16)
            # The stored aabb_corner_xyz describes the *synthetic* body this real
            # capture replaces, so it does not bound the points actually inserted
            # (a single-view capture is a partial shell, and the alignment
            # rescales it). Derive the detection target from the real points.
            if len(human_points) > 0:
                _min_corner = human_points.min(axis=0)
                _max_corner = human_points.max(axis=0)
                human_aabb_bbox = (
                    _min_corner
                    + (_max_corner - _min_corner) * AABB_CORNER_COMBINATIONS
                ).astype(human_dict["aabb_corner_xyz"].dtype)
            else:
                human_aabb_bbox = human_dict["aabb_corner_xyz"][transform_id]

            scene_data["xyz"] = np.concatenate(
                [scene_data["xyz"], human_points], axis=0
            ).astype(np.float32)
            scene_data["rgb"] = np.concatenate(
                [scene_data["rgb"], human_rgb], axis=0
            ).astype(np.float32)
            scene_data["normal"] = np.concatenate(
                [scene_data["normal"], human_normals], axis=0
            ).astype(np.float32)
            scene_data["instance_ids"] = np.concatenate(
                [scene_data["instance_ids"], human_instance_ids], axis=0
            )
            scene_data["sem_labels"] = np.concatenate(
                [scene_data["sem_labels"], human_sem_labels], axis=0
            )
            scene_data["aabb_corner_xyz"] = np.concatenate(
                [scene_data["aabb_corner_xyz"], human_aabb_bbox[None]]
            )
            scene_data["aabb_obj_ids"] = np.concatenate(
                [scene_data["aabb_obj_ids"], np.array([instance_id])]
            )

            data_dict["human_lr_label"] = (
                torch.tensor([0]) if lr == "left" else torch.tensor([1])
            )
            data_dict["has_human"] = torch.tensor([True])
        else:
            data_dict["human_lr_label"] = torch.tensor([0])
            data_dict["has_human"] = torch.tensor([False])

        scene_center_xyz = scene_data["xyz"].mean(axis=0)

        original_num_points = scene_data["xyz"].shape[0]
        choices = np.ones(shape=original_num_points, dtype=bool)
        if self.split == "train" and original_num_points > self.data_cfg.max_num_point:
            choices = np_rng.choice(
                original_num_points, self.data_cfg.max_num_point, replace=False
            )

        xyz_augment_matrix = self._get_xyz_augment_matrix()
        data_dict["point_xyz"] = (scene_data["xyz"] - scene_center_xyz)[
            choices
        ] @ xyz_augment_matrix

        if human_joints_final:
            human_joints_final_augmented = {}
            for name, joint in human_joints_final.items():
                human_joints_final_augmented[name] = (
                    joint - scene_center_xyz
                ) @ xyz_augment_matrix

            joint_matrix, joint_map, joint_mask = _build_canonical_joint_layout(
                human_joints_final_augmented
            )
            data_dict["human_joint_matrix"] = joint_matrix
            data_dict["joint_name_idx_map"] = joint_map
            data_dict["human_joint_mask"] = joint_mask
        else:
            joint_matrix, joint_map, joint_mask = self._build_dummy_human()
            data_dict["human_joint_matrix"] = joint_matrix
            data_dict["joint_name_idx_map"] = joint_map
            data_dict["human_joint_mask"] = joint_mask

        point_normal = (
            scene_data["normal"][choices]
            @ np.linalg.inv(xyz_augment_matrix).transpose()
        )
        point_rgb = scene_data["rgb"][choices]
        data_dict["instance_ids"] = scene_data["instance_ids"][choices]
        data_dict["sem_labels"] = scene_data["sem_labels"][choices].astype(np.int64)
        data_dict["scene_center_xyz"] = scene_center_xyz
        data_dict["arm_angle"] = arm_angle_value

        instance_num_point = []
        unique_instance_ids = np.unique(data_dict["instance_ids"])
        unique_instance_ids = unique_instance_ids[unique_instance_ids != -1]
        data_dict["num_instances"] = unique_instance_ids.shape[0]
        instance_centers = np.empty(
            shape=(data_dict["point_xyz"].shape[0], 3), dtype=np.float32
        )

        for idx, inst_id in enumerate(unique_instance_ids):
            assert idx == inst_id
            inst_i_idx = np.where(data_dict["instance_ids"] == inst_id)[0]
            mean_xyz_i = data_dict["point_xyz"][inst_i_idx].mean(0)
            instance_centers[inst_i_idx] = mean_xyz_i
            instance_num_point.append(inst_i_idx.size)

        data_dict["instance_num_point"] = np.array(instance_num_point, dtype=np.int32)
        data_dict["instance_centers"] = instance_centers

        data_dict["all_point_xyz"] = (
            scene_data["xyz"] - scene_center_xyz
        ) @ xyz_augment_matrix
        data_dict["all_point_rgb"] = (scene_data["rgb"] + 1) / 2

        augmented_gt_aabb_corners_tmp = (
            scene_data["aabb_corner_xyz"] - scene_center_xyz
        ) @ xyz_augment_matrix
        data_dict["gt_aabb_min_max_bounds"] = np.stack(
            (
                augmented_gt_aabb_corners_tmp.min(1),
                augmented_gt_aabb_corners_tmp.max(1),
            ),
            axis=1,
        )
        data_dict["gt_aabb_obj_ids"] = scene_data["aabb_obj_ids"]

        point_features = np.empty(
            shape=(data_dict["point_xyz"].shape[0], 0), dtype=np.float32
        )
        if self.data_cfg.point_features.use_rgb:
            point_features = np.concatenate((point_features, point_rgb), axis=1)
        if self.data_cfg.point_features.use_normal:
            point_features = np.concatenate((point_features, point_normal), axis=1)
        if self.data_cfg.point_features.use_multiview:
            if not hasattr(self, "multiview_data"):
                self._open_hdf5()
            point_features = np.concatenate(
                (point_features, self.multiview_data[scene_id][()][choices]), axis=1
            )

        point_features = np.concatenate(
            (point_features, data_dict["point_xyz"]), axis=1
        )

        (
            data_dict["voxel_xyz"],
            data_dict["voxel_features"],
            _,
            data_dict["voxel_point_map"],
        ) = ME.utils.sparse_quantize(
            coordinates=data_dict["point_xyz"] - data_dict["point_xyz"].min(axis=0),
            features=point_features,
            return_index=True,
            return_inverse=True,
            quantization_size=self.data_cfg.voxel_size,
        )
        data_dict["scene_id"] = scene_id
        return data_dict

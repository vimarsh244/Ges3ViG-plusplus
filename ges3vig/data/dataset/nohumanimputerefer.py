"""
NoHumanImputeRefer dataset.

Ablation variant of ImputeRefer that uses the same imputed scenes and
LLM-generated language data but without injecting any human mesh into
the scene.  This isolates the contribution of gesture/human cues by
running inference on scenes that contain no human.

Usage (run from project root):
    python tools/test_imputed.py \
        data=nohumanimputerefer \
        ckpt_path=<path/to/ckpt> \
        experiment_name=nohuman_ablation \
        data.inference.split=val

To additionally use the distance evaluator:
    ... data.evaluator._target_=ges3vig.evaluation.scanrefer_distance_evaluator.ScanReferDistanceEvaluator
"""

import copy
import json
import os

import clip
import MinkowskiEngine as ME
import numpy as np
import torch

from ges3vig.data.dataset.general_imputed_multi_dataset import GeneralMultiImputedDataset
from ges3vig.data.dataset.corruption import apply_pointcloud_corruption

# Minimal joint set required by PointingRegressor and human_pointer_biasing.
# All positions will be set to the scene centre (zeros in local coords),
# which is safe because has_human=False gates every code path that accesses them.
_DUMMY_JOINT_NAMES = [
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_index1",
    "right_index1",
]
_DUMMY_JOINT_IDX_MAP = {name: i for i, name in enumerate(_DUMMY_JOINT_NAMES)}
_NUM_DUMMY_JOINTS = len(_DUMMY_JOINT_NAMES)


class NoHumanImputeRefer(GeneralMultiImputedDataset):
    """ImputeRefer dataset with no human injected - gesture-cue ablation."""

    def __init__(self, data_cfg=None, split="train"):
        # Bypass parent __init__ to avoid loading the large human mesh library,
        # which is not needed for this no-human ablation.
        from torch.utils.data import Dataset as _Dataset
        _Dataset.__init__(self)
        self.data_cfg = data_cfg
        self.split = split
        self.human_data = {}  # empty - not used by this dataset

        self._load_language_data()
        self._load_scene_data()
        self._pack_data_to_chunks()

    # ------------------------------------------------------------------
    # Language loading (identical to ImputeRefer)
    # ------------------------------------------------------------------
    def _load_language_data(self):
        filtered_label_map = {}
        for i, valid_id in enumerate(self.data_cfg.scene_metadata.valid_semantic_mapping):
            filtered_label_map[valid_id] = i

        file_path = getattr(self.data_cfg.lang_metadata, f"{self.split}_language_data")
        print(f"Loading Lang Data From: {file_path}")
        with open(file_path, "r") as f:
            raw_data = json.load(f)

        # Optional eval_type filter
        eval_type_filter = getattr(self.data_cfg, "eval_type_filter", None)
        if eval_type_filter:
            print(f"Filtering by eval_type: {eval_type_filter}")
            raw_data = [item for item in raw_data if item.get("eval_type") == eval_type_filter]
            print(f"Filtered data: {len(raw_data)} samples")

        self.language_data = {}
        scene_ids = {}
        for item in raw_data:
            scene_ids[item["scene_id"]] = True
            if item["scene_id"] not in self.language_data:
                self.language_data[item["scene_id"]] = []

            object_name = item["object_name"].replace("_", " ")
            self.language_data[item["scene_id"]].append(
                {
                    "object_id": int(item["object_id"]),
                    "object_name": object_name,
                    "ann_id": int(item["ann_id"]),
                    "eval_type": item["eval_type"],
                    "clip_tokens": clip.tokenize(item["description"].strip(), truncate=True)[0],
                }
            )
        self.scene_ids = list(scene_ids.keys())

    # ------------------------------------------------------------------
    # Item loading - skip human mesh injection entirely
    # ------------------------------------------------------------------
    def __getitem__(self, index):
        data_dict = {}

        scene_id = self.scene_ids[self.chunk_scene_indices[index]]
        scene_data = copy.deepcopy(self.all_scene_data[scene_id])

        # ---- Human fields: dummy zeros, has_human=False ----
        data_dict["has_human"] = torch.tensor([False])
        data_dict["human_lr_label"] = torch.tensor([0])
        data_dict["arm_angle"] = 0.0

        # Dummy joint matrix with all joints at origin (scene-centre local coords).
        # Code paths that actually use joint positions are gated by has_human=True,
        # so these zeros are never meaningfully accessed.
        dummy_joint_matrix = np.zeros((_NUM_DUMMY_JOINTS, 3), dtype=np.float64)
        # Shape (1, N_joints, 3) to match the stacking done in the original loader.
        data_dict["human_joint_matrix"] = dummy_joint_matrix[np.newaxis]
        data_dict["joint_name_idx_map"] = dict(_DUMMY_JOINT_IDX_MAP)

        # Optional robustness corruption (noise / partial capture). No-op unless
        # data.eval_corruption.enabled is set; GT boxes are never touched.
        scene_data = apply_pointcloud_corruption(
            scene_data, self.data_cfg.get("eval_corruption", None), self.split, scene_id,
            use_multiview=self.data_cfg.point_features.use_multiview,
        )

        # ---- Standard scene processing (same as GeneralMultiImputedDataset) ----
        scene_center_xyz = scene_data["xyz"].mean(axis=0)

        original_num_points = scene_data["xyz"].shape[0]
        choices = np.ones(shape=original_num_points, dtype=bool)

        if self.split == "train" and original_num_points > self.data_cfg.max_num_point:
            choices = np.random.choice(original_num_points, self.data_cfg.max_num_point, replace=False)

        xyz_augment_matrix = self._get_xyz_augment_matrix()
        data_dict["point_xyz"] = (scene_data["xyz"] - scene_center_xyz)[choices] @ xyz_augment_matrix

        point_normal = scene_data["normal"][choices] @ np.linalg.inv(xyz_augment_matrix).transpose()
        point_rgb = scene_data["rgb"][choices]
        data_dict["instance_ids"] = scene_data["instance_ids"][choices]
        data_dict["sem_labels"] = scene_data["sem_labels"][choices].astype(np.int64)
        data_dict["scene_center_xyz"] = scene_center_xyz

        instance_num_point = []
        unique_instance_ids = np.unique(data_dict["instance_ids"])
        unique_instance_ids = unique_instance_ids[unique_instance_ids != -1]
        data_dict["num_instances"] = unique_instance_ids.shape[0]
        instance_centers = np.empty(shape=(data_dict["point_xyz"].shape[0], 3), dtype=np.float32)

        for idx, i in enumerate(unique_instance_ids):
            assert idx == i
            inst_i_idx = np.where(data_dict["instance_ids"] == i)[0]
            mean_xyz_i = data_dict["point_xyz"][inst_i_idx].mean(0)
            instance_centers[inst_i_idx] = mean_xyz_i
            instance_num_point.append(inst_i_idx.size)

        data_dict["instance_num_point"] = np.array(instance_num_point, dtype=np.int32)
        data_dict["instance_centers"] = instance_centers

        data_dict["all_point_xyz"] = (scene_data["xyz"] - scene_center_xyz) @ xyz_augment_matrix
        data_dict["all_point_rgb"] = (scene_data["rgb"] + 1) / 2

        augmented_gt_aabb_corners_tmp = (scene_data["aabb_corner_xyz"] - scene_center_xyz) @ xyz_augment_matrix
        data_dict["gt_aabb_min_max_bounds"] = np.stack(
            (augmented_gt_aabb_corners_tmp.min(1), augmented_gt_aabb_corners_tmp.max(1)), axis=1
        )
        data_dict["gt_aabb_obj_ids"] = scene_data["aabb_obj_ids"]

        point_features = np.empty(shape=(data_dict["point_xyz"].shape[0], 0), dtype=np.float32)
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

        point_features = np.concatenate((point_features, data_dict["point_xyz"]), axis=1)

        data_dict["voxel_xyz"], data_dict["voxel_features"], _, data_dict["voxel_point_map"] = (
            ME.utils.sparse_quantize(
                coordinates=data_dict["point_xyz"] - data_dict["point_xyz"].min(axis=0),
                features=point_features,
                return_index=True,
                return_inverse=True,
                quantization_size=self.data_cfg.voxel_size,
            )
        )
        data_dict["scene_id"] = scene_id

        # ---- Language data ----
        language_data_indices = [self.chunk_lang_indices[index]]
        language_data_in_scene = self.language_data[scene_id]
        num_language_data_in_scene = len(language_data_in_scene)

        data_dict["ann_id"] = np.empty(shape=self.data_cfg.chunk_size, dtype=np.int32)
        data_dict["object_id"] = np.empty(shape=self.data_cfg.chunk_size, dtype=np.int16)
        data_dict["gt_target_obj_id_mask"] = np.zeros(
            shape=(self.data_cfg.chunk_size, data_dict["gt_aabb_obj_ids"].shape[0]), dtype=bool
        )
        data_dict["clip_tokens"] = torch.empty(size=(self.data_cfg.chunk_size, 77), dtype=torch.int32)
        data_dict["eval_type"] = []
        data_dict["arm_angle"] = []

        for i, lang_idx in enumerate(language_data_indices):
            real_idx = lang_idx % num_language_data_in_scene
            lang_data = language_data_in_scene[real_idx]
            data_dict["ann_id"][i] = lang_data["ann_id"]
            data_dict["object_id"][i] = lang_data["object_id"]
            data_dict["gt_target_obj_id_mask"][i] = np.in1d(
                data_dict["gt_aabb_obj_ids"], lang_data["object_id"]
            )
            data_dict["clip_tokens"][i] = lang_data["clip_tokens"]
            data_dict["eval_type"].append(lang_data["eval_type"])
            data_dict["arm_angle"].append(0.0)

        return data_dict

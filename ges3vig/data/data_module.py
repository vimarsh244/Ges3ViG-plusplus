import torch
import MinkowskiEngine as ME
import lightning.pytorch as pl
from importlib import import_module
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate
from time import perf_counter


class DataModule(pl.LightningDataModule):
    def __init__(self, data_cfg):
        super().__init__()
        self.data_cfg = data_cfg
        dataset_name = data_cfg.lang_dataset
        self.dataset = getattr(import_module(f"ges3vig.data.dataset.{dataset_name.lower()}"), dataset_name)

        # Disable pin_memory on Jetson / unified-memory platforms (wastes memory)
        self._pin_memory = not self._is_unified_memory()

    def setup(self, stage=None):
        if stage == "fit":
            self.train_set = self.dataset(self.data_cfg, "train")
            self.val_set = self.dataset(self.data_cfg, "val")
        if stage == "test":
            self.test_set = self.dataset(self.data_cfg, self.data_cfg.inference.split)
        if stage == "predict":
            self.test_set = self.dataset(self.data_cfg, "test")

    @staticmethod
    def _is_unified_memory() -> bool:
        """Detect Jetson / unified-memory platforms where pin_memory is wasteful."""
        import platform
        if platform.machine() in ("aarch64", "arm64"):
            try:
                with open("/proc/device-tree/model", "r") as f:
                    model = f.read().lower()
                if "jetson" in model or "nvidia" in model:
                    return True
            except (FileNotFoundError, PermissionError):
                pass
            # Fallback: if GPU total ≈ system RAM total, it's unified memory
            try:
                if torch.cuda.is_available():
                    gpu_mem = torch.cuda.get_device_properties(0).total_mem
                    with open("/proc/meminfo") as f:
                        for line in f:
                            if line.startswith("MemTotal:"):
                                sys_mem = int(line.split()[1]) * 1024  # kB → bytes
                                # If GPU mem is ≥ 50% of system RAM, likely unified
                                if gpu_mem >= sys_mem * 0.5:
                                    return True
                                break
            except Exception:
                pass
        return False

    def train_dataloader(self):
        #self.train_set.shuffle_chunks()  # shuffle language data chunks after each epoch
        return DataLoader(self.train_set, batch_size=self.data_cfg.dataloader.batch_size, shuffle=True, pin_memory=self._pin_memory,
                          collate_fn=_collate_fn, num_workers=self.data_cfg.dataloader.num_workers, drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.data_cfg.dataloader.batch_size, pin_memory=self._pin_memory,
                          collate_fn=_collate_fn, num_workers=self.data_cfg.dataloader.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.data_cfg.dataloader.batch_size, pin_memory=self._pin_memory,
                          collate_fn=_collate_fn, num_workers=self.data_cfg.dataloader.num_workers)

    def predict_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.data_cfg.dataloader.batch_size, pin_memory=self._pin_memory,
                          collate_fn=_collate_fn, num_workers=self.data_cfg.dataloader.num_workers)


def _collate_fn(batch):
    collate_start = perf_counter()
    # Filter out None values (samples that couldn't be loaded)
    batch = [b for b in batch if b is not None]
    
    # If all samples were None, return None
    if len(batch) == 0:
        return None
    
    data_dict = {}

    # default collation
    default_collate_item_names = ("scene_id", "object_id", "ann_id", "clip_tokens", "scene_center_xyz")
    default_collate_data = []

    point_count_total = 0
    all_point_count_total = 0
    aabb_count_total = 0

    data_dict["point_count_offsets"] = torch.zeros(size=(len(batch) + 1,), dtype=torch.int32)
    data_dict["aabb_count_offsets"] = torch.zeros(size=(len(batch) + 1,), dtype=torch.int32)
    data_dict["all_point_count_offsets"] = torch.zeros(size=(len(batch) + 1,), dtype=torch.int32)
    data_dict["eval_type"] = []
    data_dict["arm_angle"] = []

    vert_batch_ids = []
    #data_dict["human_keypoints"] = torch.empty((0,2,3))
    #data_dict["has_human"] = torch.empty(size=(0,))
    for i, b in enumerate(batch):
        # organize default collation
        default_collate_data.append({k: b[k] for k in default_collate_item_names})

        # pre-calculate the numbers of total points and aabbs for sparse collation
        point_count_total += b["point_xyz"].shape[0]
        all_point_count_total += b["all_point_xyz"].shape[0]
        data_dict["all_point_count_offsets"][i + 1] = all_point_count_total
        aabb_count_total += b["gt_aabb_min_max_bounds"].shape[0]
        data_dict["point_count_offsets"][i + 1] = point_count_total
        data_dict["aabb_count_offsets"][i + 1] = aabb_count_total
        data_dict["eval_type"].append(b["eval_type"])
        data_dict["arm_angle"].append(b.get("arm_angle", [None]))
        #data_dict["human_keypoints"] = torch.cat([data_dict["human_keypoints"], b['human_keypoints'][None]])
        #data_dict["has_human"] = torch.cat([data_dict["has_human"], b['has_human'][None]])
        vert_batch_ids.append(
            torch.full((b["point_xyz"].shape[0],), fill_value=i, dtype=torch.uint8)
        )
        #print("batches: ", i)
    data_dict["vert_batch_ids"] = torch.cat(vert_batch_ids, dim=0)
    data_dict.update(default_collate(default_collate_data))
    lang_chunk_size = data_dict["ann_id"].shape[1]
    #print(f"data_dict['human_keypoints']: {data_dict['human_keypoints'].shape}")
    # sparse collation
    data_dict["point_xyz"] = torch.empty(size=(point_count_total, 3), dtype=torch.float32)

    data_dict["all_point_xyz"] = torch.empty(size=(all_point_count_total, 3), dtype=torch.float32)
    data_dict["all_point_rgb"] = torch.empty_like(data_dict["all_point_xyz"])

    data_dict["instance_ids"] = torch.empty(size=(point_count_total,), dtype=torch.int16)
    data_dict["sem_labels"] = torch.empty_like(data_dict["instance_ids"], dtype=torch.long)
    data_dict["instance_centers"] = torch.empty_like(data_dict["point_xyz"])

    data_dict["gt_aabb_min_max_bounds"] = torch.empty(size=(aabb_count_total, 2, 3), dtype=torch.float32)
    data_dict["gt_aabb_obj_ids"] = torch.empty(size=(aabb_count_total, ), dtype=torch.int16)

    data_dict["gt_target_obj_id_mask"] = torch.empty(
        size=(aabb_count_total, lang_chunk_size), dtype=torch.bool
    )
    
    

    num_voxel_batch = 0
    voxel_xyz_list = []
    voxel_features_list = []
    voxel_point_map_list = []
    instance_num_point = []

    num_instances = 0

    human_lr_label = []
    has_human = []
    human_joint_matrix = []
    joint_name_idx_map = []
    for i, b in enumerate(batch):
        batch_points_start_idx = data_dict["point_count_offsets"][i]
        batch_points_end_idx = data_dict["point_count_offsets"][i+1]

        data_dict["all_point_xyz"][data_dict["all_point_count_offsets"][i]:data_dict["all_point_count_offsets"][i+1]] = \
            torch.from_numpy(b["all_point_xyz"])
        data_dict["all_point_rgb"][
        data_dict["all_point_count_offsets"][i]:data_dict["all_point_count_offsets"][i+1]] = torch.from_numpy(
            b["all_point_rgb"])
        data_dict["point_xyz"][batch_points_start_idx:batch_points_end_idx] = torch.from_numpy(b["point_xyz"])

        instance_ids_tmp = torch.from_numpy(b["instance_ids"])
        instance_ids_tmp[instance_ids_tmp != -1] += num_instances
        num_instances += b["num_instances"]
        data_dict["instance_ids"][batch_points_start_idx:batch_points_end_idx] = instance_ids_tmp

        data_dict["sem_labels"][batch_points_start_idx:batch_points_end_idx] = torch.from_numpy(b["sem_labels"])
        data_dict["instance_centers"][batch_points_start_idx:batch_points_end_idx] = torch.from_numpy(
            b["instance_centers"]
        )
        instance_num_point.append(torch.from_numpy(b["instance_num_point"]))

        voxel_point_map_list.append(b["voxel_point_map"] + num_voxel_batch)
        num_voxel_batch += b["voxel_xyz"].shape[0]

        voxel_xyz_list.append(b["voxel_xyz"])
        voxel_features_list.append(b["voxel_features"])

        batch_aabbs_start_idx = data_dict["aabb_count_offsets"][i]
        batch_aabbs_end_idx = data_dict["aabb_count_offsets"][i+1]
        data_dict["gt_aabb_min_max_bounds"][batch_aabbs_start_idx:batch_aabbs_end_idx] = \
            torch.from_numpy(b["gt_aabb_min_max_bounds"])
        data_dict["gt_aabb_obj_ids"][batch_aabbs_start_idx:batch_aabbs_end_idx] = \
            torch.from_numpy(b["gt_aabb_obj_ids"])
        data_dict["gt_target_obj_id_mask"][batch_aabbs_start_idx:batch_aabbs_end_idx] = \
            torch.from_numpy(b["gt_target_obj_id_mask"]).permute(dims=(1, 0))

        human_lr_label.append(b["human_lr_label"])
        has_human.append(b["has_human"])
        human_joint_matrix.append(torch.from_numpy(b['human_joint_matrix']))
        joint_name_idx_map.append(b['joint_name_idx_map'])
    data_dict['joint_name_idx_map'] = joint_name_idx_map
    data_dict["human_joint_matrix"] = torch.cat(human_joint_matrix)
    data_dict["human_lr_label"] = torch.cat(human_lr_label)
    data_dict["has_human"] = torch.cat(has_human)
    data_dict["instance_num_point"] = torch.cat(instance_num_point, dim=0)

    sparse_collate_start = perf_counter()
    data_dict["voxel_xyz"], data_dict["voxel_features"] = ME.utils.sparse_collate(
        coords=voxel_xyz_list, feats=voxel_features_list
    )
    sparse_collate_ms = (perf_counter() - sparse_collate_start) * 1000.0
    data_dict["voxel_point_map"] = torch.cat(voxel_point_map_list, dim=0)

    sample_timing_list = [b.get("timing_data", {}) for b in batch]
    sample_timing_avg = {}
    if sample_timing_list and any(sample_timing_list):
        all_keys = sorted({k for timing in sample_timing_list for k in timing.keys()})
        for key in all_keys:
            vals = [float(timing[key]) for timing in sample_timing_list if key in timing]
            if vals:
                sample_timing_avg[key] = sum(vals) / len(vals)

    data_dict["timing_data"] = {
        "collate_total_ms": (perf_counter() - collate_start) * 1000.0,
        "collate_sparse_ms": sparse_collate_ms,
        "sample_timing_avg_ms": sample_timing_avg,
    }
    return data_dict

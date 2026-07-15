import lightning.pytorch as pl
import torch
import torch.nn as nn

class HumanLosses(pl.LightningModule):
    def __init__(self, include_dir_loss = False):
        super().__init__()
        #self.include_dir_loss = include_dir_loss
        self.lr_criterion = nn.CrossEntropyLoss()
    def forward(self, data_dict, output_dict, loss_dict):
        diffs = output_dict["joints"] - data_dict["human_joint_matrix"]
        per_joint = diffs.norm(dim = 2)  # (B, num_joints)
        # Datasets with only partial GT joints (e.g. real humans -> 10 arm
        # joints in a 55-joint frame) supply a 0/1 mask so the regressor isn't
        # penalised on the unobserved (zero-filled) joints. Absent => all joints
        # supervised, identical to the original behaviour (synthetic humans).
        joint_mask = data_dict.get("human_joint_mask", None)
        if joint_mask is not None:
            per_joint = per_joint * joint_mask.to(per_joint.dtype)
        loss_dict["human_joint_loss"] = per_joint.sum(dim = 1).mean(dim = 0)
        loss_dict["lr_loss"] = self.lr_criterion(output_dict['human_lr'], data_dict["human_lr_label"])

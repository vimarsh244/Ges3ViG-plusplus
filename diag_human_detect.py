"""Diagnostic: does PointGroup actually detect the (real) human?

Patches HumanProcessor.forward to record, per scene:
  - GT human points in the cloud (sem_labels == human_label)
  - points the detector's semantic head PREDICTS as human
  - whether has_human fired (>300 predicted human pts)
Run exactly like test.py but it only does a forward sweep and prints a summary.
"""
import os
import numpy as np
import torch
import hydra
import lightning.pytorch as pl
from omegaconf import DictConfig, ListConfig
from ges3vig.data.data_module import DataModule
import ges3vig.model.vision_module.human_processor_early_matcher_plus_query_former as hp

torch.serialization.add_safe_globals([DictConfig, ListConfig])

STATS = []
_orig_forward = hp.HumanProcessor.forward


def patched_forward(self, point_xyz, point_features, vert_batch_ids,
                    semantic_scores, batch_size, scene_id=None):
    semantic_preds = semantic_scores.argmax(1).to(torch.int16)
    for i in range(batch_size):
        m = vert_batch_ids == i
        pred_h = int(((semantic_preds == self.human_label) & m).sum().item())
        STATS.append({"human_label": int(self.human_label),
                      "pred_human_pts": pred_h,
                      "total_pts": int(m.sum().item()),
                      "has_human": pred_h > 300})
    return _orig_forward(self, point_xyz, point_features, vert_batch_ids,
                         semantic_scores, batch_size, scene_id)


hp.HumanProcessor.forward = patched_forward


@hydra.main(version_base=None, config_path="config", config_name="global_config")
def main(cfg):
    pl.seed_everything(0, workers=True)
    data_module = DataModule(cfg.data)
    cfg.data.evaluator.verbose = False
    model = hydra.utils.instantiate(cfg.model.model_name, cfg)
    trainer = pl.Trainer(accelerator="gpu", devices=1, max_epochs=1, logger=False,
                         limit_test_batches=cfg.trainer.get("limit_test_batches", 1.0))
    trainer.test(model=model, datamodule=data_module, ckpt_path=cfg.ckpt_path)

    n = len(STATS)
    ph = np.array([s["pred_human_pts"] for s in STATS])
    hh = np.array([s["has_human"] for s in STATS])
    print("\n\n================ HUMAN-DETECTION DIAGNOSTIC ================")
    print(f"human_label index (remapped): {STATS[0]['human_label']}")
    print(f"scenes evaluated:             {n}")
    print(f"has_human fired (>300 pred):  {hh.sum()} / {n}  ({100*hh.mean():.1f}%)")
    print(f"predicted-human points/scene: mean={ph.mean():.1f}  median={np.median(ph):.0f}  max={ph.max()}")
    print(f"scenes with 0 predicted human pts: {(ph==0).sum()} / {n}")
    print(f"scenes with >=1 predicted human pt: {(ph>0).sum()} / {n}")
    print("===========================================================")


if __name__ == "__main__":
    main()

"""Finetune Ges3ViG from an existing checkpoint on a NEW dataset (e.g. real humans).

Unlike `train.py`, this loads the FULL model weights from `finetune_ckpt` as
*initialization* and starts a FRESH trainer (epoch 0, fresh optimizer / LR
schedule). Resuming the released checkpoint via Lightning's `ckpt_path` would be
a no-op because it was saved at epoch 39 with max_epochs 40.

Usage (detector-only finetune, recommended for the real-human detection gap):

  python finetune.py \
    +finetune_ckpt=/nas_hcfc/HRI_general/common/checkpoints/Ges3ViG_ImputeRefer.ckpt \
    data=imputerefer_realhumans experiment_name=ft_realhumans_detector \
    trainer.max_epochs=15 model.network.detector.train_pg=True \
    model.network.detector.train_pg_num_epochs=999 \
    model.optimizer.lr=1e-4
"""
import os
import torch
import hydra
import lightning.pytorch as pl
from omegaconf import DictConfig, ListConfig
from ges3vig.data.data_module import DataModule
from lightning.pytorch.callbacks import LearningRateMonitor
from ges3vig.callback.gpu_cache_clean_callback import GPUCacheCleanCallback
from ges3vig.callback.lr_decay_callback_imputed import LrDecayImputedCallback

torch.serialization.add_safe_globals([DictConfig, ListConfig])


def init_callbacks(cfg):
    checkpoint_monitor = hydra.utils.instantiate(cfg.checkpoint_monitor)
    return [checkpoint_monitor, GPUCacheCleanCallback(),
            LrDecayImputedCallback(), LearningRateMonitor(logging_interval="epoch")]


@hydra.main(version_base=None, config_path="config", config_name="global_imputed_config")
def main(cfg):
    pl.seed_everything(cfg.train_seed, workers=True)
    os.makedirs(os.path.join(cfg.experiment_output_path, "training"), exist_ok=True)

    data_module = DataModule(cfg.data)
    model = hydra.utils.instantiate(cfg.model.model_name, cfg)

    # --- load FULL weights as initialization (not a resume) ---
    ft_ckpt = cfg.get("finetune_ckpt", None)
    assert ft_ckpt is not None and os.path.exists(ft_ckpt), \
        f"Provide +finetune_ckpt=<path>; got {ft_ckpt}"
    state = torch.load(ft_ckpt, map_location="cpu", weights_only=False)["state_dict"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"[finetune] loaded {len(state)} params from {ft_ckpt}")
    print(f"[finetune] missing={len(missing)} unexpected={len(unexpected)}")

    if cfg.model.network.freeze_human_decoder:
        for p in model.detector.human_decoder.parameters():
            p.requires_grad = False
    if cfg.model.network.freeze_full_detector:
        for p in model.detector.parameters():
            p.requires_grad = False

    logger = hydra.utils.instantiate(cfg.logger)
    trainer = pl.Trainer(callbacks=init_callbacks(cfg), logger=logger, **cfg.trainer)

    # NOTE: ckpt_path intentionally NOT passed -> fresh epoch/optimizer state.
    trainer.fit(model=model, datamodule=data_module)


if __name__ == "__main__":
    main()

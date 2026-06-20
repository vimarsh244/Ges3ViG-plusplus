import os
import torch
import hydra
import lightning.pytorch as pl
from omegaconf import DictConfig, ListConfig
from ges3vig.data.data_module import DataModule

# PyTorch 2.6 changed weights_only default to True; allowlist omegaconf types
# that are embedded in checkpoints saved with save_hyperparameters().
torch.serialization.add_safe_globals([DictConfig, ListConfig])


@hydra.main(version_base=None, config_path="config", config_name="global_config")
def main(cfg):
    # Seed control: by default use the fixed `test_seed` for reproducible IoU.
    # To run with a non-deterministic (random) seed, pass `test_seed=null`,
    # `test_seed=-1`, or `test_seed=random` on the command line.
    seed = cfg.test_seed
    if seed in (None, -1, "random", "rand"):
        seed = pl.seed_everything(workers=True)  # picks a random seed and returns it
        print(f"[test] Using RANDOM seed: {seed}")
    else:
        pl.seed_everything(int(seed), workers=True)
        print(f"[test] Using FIXED seed: {seed}")

    # create directories for inference outputs
    os.makedirs(os.path.join(cfg.pred_path, cfg.data.inference.split), exist_ok=True)

    # initialize data
    data_module = DataModule(cfg.data)

    # initialize model
    cfg.data.evaluator.verbose = True  # print out evaluation results after inference
    model = hydra.utils.instantiate(cfg.model.model_name, cfg)

    # initialize trainer.
    # `trainer.limit_test_batches` (int = #batches, float = fraction) lets you run a
    # quick subset for validation, e.g. `trainer.limit_test_batches=20`. Default 1.0 = full set.
    limit_test_batches = cfg.trainer.get("limit_test_batches", 1.0)
    trainer = pl.Trainer(accelerator=cfg.trainer.accelerator, devices=1, max_epochs=1, logger=False,
                         limit_test_batches=limit_test_batches)

    # check the checkpoint
    assert cfg.ckpt_path is not None, "Error: Checkpoint path is not provided."
    assert os.path.exists(cfg.ckpt_path), f"Error: Checkpoint path {cfg.ckpt_path} does not exist."

    # start inference
    trainer.test(model=model, datamodule=data_module, ckpt_path=cfg.ckpt_path)


if __name__ == '__main__':
    main()

import os

# import warnings
import gc  # Python's garbage collector
import hydra
import torch
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint


os.environ["HYDRA_FULL_ERROR"] = "1"

SAVE = True


@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def main(cfg):
    # if cfg.seed is an integer
    if cfg.seed != 0:
        seed_everything(cfg.seed, workers=True)
    torch.cuda.set_device(cfg.gpu_id)

    logger = hydra.utils.instantiate(cfg.logger)
    model = hydra.utils.instantiate(cfg.model)

    # Define a checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints/",
        filename="{epoch:02d}-{val_loss:.2f}",
        save_top_k=1,
        verbose=False,
        monitor="val/loss",
        mode="min",
    )
    trainer = hydra.utils.instantiate(
        cfg.trainer, logger=logger, callbacks=[checkpoint_callback]
    )

    trainer.fit(model=model)
    trainer.test(model=model)

    # Save the final model
    if SAVE:
        if cfg.model.relative is True:
            trainer.save_checkpoint(
                f"weights/{cfg.model.dataset}_{cfg.model.encoder}_{cfg.model.n_anchors}_{cfg.seed}_{cfg.model.idx_flag}_{cfg.model.sim_type}.ckpt"
            )
        else:
            trainer.save_checkpoint(
                f"weights/{cfg.model.dataset}_{cfg.model.encoder}_{cfg.seed}.ckpt"
            )

    # Cleanup CUDA memory
    del model  # Delete model to free up GPU memory
    torch.cuda.empty_cache()  # Clear cache
    gc.collect()  # Collect garbage to free memory


if __name__ == "__main__":
    main()

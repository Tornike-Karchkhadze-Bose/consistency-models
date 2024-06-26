import yaml
import argparse
from diffusion import Diffusion
import pytorch_lightning as pl
from ema import EMA, EMAModelCheckpoint
from torch.utils.data import DataLoader
from data import get_dataset
from pytorch_lightning.strategies.ddp import DDPStrategy
import click
import os
import datetime
from pytorch_lightning.loggers import WandbLogger
import numpy as np

def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace

@click.command()
@click.option('--cfg', default='config.yml', help='Configuration File')
def main(cfg):
    config = yaml.load(open(cfg, 'r'), Loader=yaml.FullLoader)
    with open(cfg, "r") as f:
        cfg = yaml.safe_load(f)
        cfg = dict2namespace(cfg)

    log_path = cfg.log_directory
    os.makedirs(log_path, exist_ok=True)

    # adding a random number of seconds so that exp folder names coincide less often
    random_seconds_shift = datetime.timedelta(seconds=np.random.randint(60))
    now = (datetime.datetime.now() - random_seconds_shift).strftime('%Y-%m-%dT%H-%M-%S')
    nowname = "%s_%s_%s" % (
        now,
        cfg.id.name,
        cfg.id.version,
        # int(time.time())
    )

    print("\nName of the run is:", nowname, "\n")

    run_path = os.path.join(
        log_path,
        cfg.project_name,
        nowname,
    )

    os.makedirs(run_path, exist_ok=True)

    wandb_logger = WandbLogger(
        save_dir=run_path,
        # version=nowname,
        project= cfg.project_name,
        config=config,
        name=nowname
    )
    wandb_logger._project = ""  # prevent naming experiment nama 2 time in logginf vals

    checkpoint_path = os.path.join(
        log_path,
        cfg.project_name,
        nowname,
        "checkpoints",
    )
    os.makedirs(checkpoint_path, exist_ok=True)

    ckpt_callback = EMAModelCheckpoint(dirpath= checkpoint_path, save_top_k=cfg.training.save_top_k, monitor="val_loss", save_last=True, filename='{epoch}-{val_loss:.4f}', every_n_train_steps=None,)
    ema_callback = EMA(decay=cfg.model.ema_rate)
    callbacks = [ckpt_callback, ema_callback]

    model = Diffusion(cfg)

    train_dataloader = DataLoader(
        get_dataset(cfg.data.name, train=True), 
        batch_size=cfg.training.batch_size, 
        shuffle=True, 
        num_workers=cfg.data.num_workers, 
        pin_memory=True, 
        persistent_workers=True,
    )

    val_dataloader = DataLoader(
        get_dataset(cfg.data.name, train=False), 
        batch_size=cfg.testing.batch_size, 
        shuffle=False, 
        num_workers=cfg.data.num_workers, 
        pin_memory=True, 
        persistent_workers=True,
    )


    trainer = pl.Trainer(
        callbacks=callbacks,
        logger=wandb_logger,
        precision=cfg.training.precision,
        max_steps=cfg.training.max_steps,
        accumulate_grad_batches=cfg.training.accumulate_grad_batches,
        accelerator="gpu", 
        devices=cfg.training.devices,
        num_sanity_val_steps=0,
        # limit_val_batches=2,
        # limit_train_batches=2,
        log_every_n_steps=cfg.training.log_every_n_steps,
        check_val_every_n_epoch=cfg.training.validation_every_n_epochs,
        gradient_clip_val=cfg.optim.grad_clip,
        benchmark=True,
        strategy = DDPStrategy(find_unused_parameters=False)
                    if isinstance(cfg.training.devices, (list, tuple)) and len(cfg.training.devices) > 1
                    else "auto",  # Ensure cfg.training.devices is a list or tuple
        # DDPStrategy(find_unused_parameters=False),
    )

    # Train
    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader, ckpt_path = cfg.training.resume_from_checkpoint)

if __name__ == '__main__':
    main()
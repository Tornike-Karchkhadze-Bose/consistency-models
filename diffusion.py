import pytorch_lightning as pl
from networks import VEPrecond, VPPrecond, EDMPrecond, CTPrecond
from loss import VELoss, VPLoss, EDMLoss, CTLoss
from torch import optim
import numpy as np
import torch
from sampler import multistep_consistency_sampling
from torchvision.utils import make_grid, save_image
import copy
from torchmetrics.image.inception import InceptionScore
# from sampler import multistep_consistency_sampling
import os
from torchmetrics.image.fid import FrechetInceptionDistance


class CosineWarmupScheduler(optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup, max_iters):
        self.warmup = warmup
        self.max_num_iters = max_iters
        super().__init__(optimizer)

    def get_lr(self):
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, epoch):
        lr_factor = 0.5 * (1 + np.cos(np.pi * epoch / self.max_num_iters))
        if epoch <= self.warmup:
            lr_factor *= epoch * 1.0 / self.warmup
        return lr_factor
    
class Diffusion(pl.LightningModule):
    def __init__(self, cfg=None):
        super().__init__()
        self.cfg = cfg
        self.save_hyperparameters()
        if cfg.diffusion.preconditioning == 'vp':
            self.loss_fn = VPLoss()
            self.net = VPPrecond(self.cfg)
        elif cfg.diffusion.preconditioning == 've':
            self.loss_fn = VELoss()
            self.net = VEPrecond(self.cfg)
        elif cfg.diffusion.preconditioning == 'edm':
            self.loss_fn = EDMLoss()
            self.net = EDMPrecond(self.cfg)
        elif cfg.diffusion.preconditioning == 'ct':
            self.loss_fn = CTLoss(self.cfg)
            self.net = CTPrecond(self.cfg, use_fp16=self.cfg.training.precision == 16)
            self.net_ema = CTPrecond(self.cfg, use_fp16=self.cfg.training.precision == 16) # no_grad or not ???
            for param in self.net_ema.parameters():
                param.requires_grad = False
            self.net_ema.load_state_dict(copy.deepcopy(self.net.state_dict()))

            self.N_and_mu =  self.cfg.diffusion.N_and_mu
        else:
            raise ValueError(f'Preconditioning {cfg.diffusion.preconditioning} does not exist')
        if self.cfg.testing.calc_inception:
            self.inception = InceptionScore()
        if self.cfg.testing.calc_fid:
            self.fid = FrechetInceptionDistance(feature=2048).cuda()
            

    def training_step(self, batch, _):
        images, _ = batch
        if self.N_and_mu == "fixed":
            loss = self.loss_fn(net=self.net, net_ema=self.net_ema, images=images, k=self.cfg.training.max_steps).mean()
        elif self.N_and_mu == "adaptive":
            loss = self.loss_fn(net=self.net, net_ema=self.net_ema, images=images, k=self.global_step).mean()

        with torch.no_grad():
            if self.N_and_mu == "fixed":
                mu = self.cfg.diffusion.mu_0
            elif self.N_and_mu == "adaptive":
                mu = self.net.mu(self.global_step)
            # update \theta_{-}
            for p, ema_p in zip(self.net.parameters(), self.net_ema.parameters()):
                ema_p.mul_(mu).add_(p, alpha=1 - mu)

        self.log("train_loss", loss, 
                    prog_bar=True,
                    logger=True,
                    on_step=True,
                    on_epoch=False,
                    )
        # return {'loss': loss}
        return loss

    def validation_step(self, batch, batch_idx):
        images, _ = batch
        if self.N_and_mu == "fixed":
            loss = self.loss_fn(net=self.net, net_ema=self.net_ema, images=images, k=self.cfg.training.max_steps).mean()
        elif self.N_and_mu == "adaptive":
            loss = self.loss_fn(net=self.net, net_ema=self.net_ema, images=images, k=self.global_step).mean()
        self.log("val_loss", loss, sync_dist=True)

        # Log one sample of the original and generated images for the first batch in the epoch
        if batch_idx == 0 and self.global_rank == 0:
            name = self.cfg.data.name
            latents = torch.randn(self.cfg.testing.samples, self.cfg.data.img_channels, self.cfg.data.img_resolution, self.cfg.data.img_resolution).to(self.device) 
            xh = multistep_consistency_sampling(self.net_ema, latents=latents, t_steps=[80])
            xh = (xh * 0.5 + 0.5).clamp(0, 1)
            generated_grid = make_grid(xh, nrow=8)

            original_image = (images[:self.cfg.testing.samples] * 0.5 + 0.5).clamp(0, 1)
            original_grid = make_grid(original_image, nrow=8)

            self.logger.log_image(f"val_samples_{name}", [generated_grid.permute(1, 2, 0).cpu().numpy(), original_grid.permute(1, 2, 0).cpu().numpy()], caption=["Generated", "Original"])


        if self.cfg.testing.calc_inception:
            latents = torch.randn(images.shape[0], self.cfg.data.img_channels, self.cfg.data.img_resolution, self.cfg.data.img_resolution).cuda() 
            xh = multistep_consistency_sampling(self.net, latents=latents, t_steps=[80])
            xh = ((xh + 1) * 127.5).clamp(0, 255).to(torch.uint8)
            self.inception.update(xh)

        if self.cfg.testing.calc_fid:
            x_batch = ((images + 1) * 127.5).clamp(0, 255).to(torch.uint8)
            self.fid.update(x_batch, real=True)

            latents = torch.randn(images.shape[0], self.cfg.data.img_channels, self.cfg.data.img_resolution, self.cfg.data.img_resolution).cuda() 
            xh = multistep_consistency_sampling(self.net_ema, latents=latents, t_steps=[80])   # self.net_ema or self.net ????
            xh = ((xh + 1) * 127.5).clamp(0, 255).to(torch.uint8)
            self.fid.update(xh, real=False)

        # return {'loss': loss}
    
    
    def on_validation_epoch_end(self):
        # latents = torch.randn(self.cfg.testing.samples, self.cfg.data.img_channels, self.cfg.data.img_resolution, self.cfg.data.img_resolution).to(self.device) 
        # name = self.cfg.data.name
        # xh = multistep_consistency_sampling(self.net_ema, latents=latents, t_steps=[80])
        # xh = (xh * 0.5 + 0.5).clamp(0, 1)
        # grid = make_grid(xh, nrow=8)
        # self.logger.log_image(f"sample_1step_{name}", [grid.permute(1, 2, 0).cpu().numpy()], caption=[""])
        # save_dir = os.path.join(self.logger.log_dir, "samples")
        # os.makedirs(save_dir, exist_ok=True)
        # save_image(grid, f"{save_dir}/ct_{name}_sample_1step_{self.global_step}.png")
        if self.cfg.testing.calc_inception:
            iscore = self.inception.compute()[0]
            self.log('iscore', iscore, sync_dist=True)
            self.inception.reset()
        if self.cfg.testing.calc_fid:
            fid = self.fid.compute().item()
            self.log('fid', fid, sync_dist=True)
            self.fid.reset()
        return super().on_validation_epoch_end()
    
    def configure_optimizers(self):
        cfg = self.cfg.optim
        if cfg.optimizer == 'radam':
            optimizer = optim.RAdam(self.net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay, betas=cfg.betas, 
                            eps=cfg.eps)
        elif cfg.optimizer == 'adam':
            optimizer = optim.Adam(self.net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay, betas=cfg.betas, 
                            eps=cfg.eps)
        elif cfg.optimizer == 'adamw':
            optimizer = optim.AdamW(self.net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay, betas=cfg.betas, 
                            eps=cfg.eps)
        elif cfg.optimizer == 'rmsprop':
            optimizer = optim.RMSprop(self.net.parameters(), lr=cfg.lr)
        elif cfg.optimizer == 'sgd':
            optimizer = optim.SGD(self.net.parameters(), lr=cfg.lr)

        lr_scheduler = CosineWarmupScheduler(optimizer=optimizer, warmup=self.cfg.training.warmup_epochs, max_iters=self.cfg.training.max_epochs)

        return {
        "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
            },
        }
    
    
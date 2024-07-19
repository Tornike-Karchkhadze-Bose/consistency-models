import pytorch_lightning as pl
from networks import VEPrecond, VPPrecond, EDMPrecond, CTPrecond, iCTPrecond
from loss import VELoss, VPLoss, EDMLoss, CTLoss, iCTLoss
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
from ema import EMA
import torch.nn as nn
from ncsnpp_models.ncsnpp import NCSNpp
from ncsnpp_models.default_cifar10_configs import get_config

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
            self.target_model = CTPrecond(self.cfg, use_fp16=self.cfg.training.precision == 16) # no_grad or not ???
            for param in self.target_model.parameters():
                param.requires_grad = False
            self.target_model.load_state_dict(copy.deepcopy(self.net.state_dict()))

            self.N_and_mu =  self.cfg.diffusion.N_and_mu
        elif cfg.diffusion.preconditioning == 'ict':
            self.loss_fn = iCTLoss(self.cfg)
            self.net = iCTPrecond(self.cfg, use_fp16=self.cfg.training.precision == 16)
            self.target_model = iCTPrecond(self.cfg, use_fp16=self.cfg.training.precision == 16) # no_grad or not ???
            for param in self.target_model.parameters():
                param.requires_grad = False
            self.target_model.load_state_dict(copy.deepcopy(self.net.state_dict()))

            # Initialize EMA models
            self.ema_models = nn.ModuleList()
            for ema_rate in self.cfg.diffusion.ema_rate:
                ema_model = iCTPrecond(self.cfg, use_fp16=self.cfg.training.precision == 16)
                for param in ema_model.parameters():
                    param.requires_grad = False
                ema_model.load_state_dict(copy.deepcopy(self.net.state_dict()))
                ema_model.eval()
                self.ema_models.append(ema_model)
            self.ema_models.eval()

            self.N_and_mu =  self.cfg.diffusion.N_and_mu
        elif cfg.diffusion.preconditioning == 'ict-ncsnpp':
            self.loss_fn = iCTLoss(self.cfg)
            config = get_config()
            self.net = NCSNpp(config, self.cfg)
            self.target_model = NCSNpp(config, self.cfg) # no_grad or not ???
            for param in self.target_model.parameters():
                param.requires_grad = False
            self.target_model.load_state_dict(copy.deepcopy(self.net.state_dict()))

            # Initialize EMA models
            self.ema_models = nn.ModuleList()
            for ema_rate in self.cfg.diffusion.ema_rate:
                ema_model = NCSNpp(config, self.cfg)
                for param in ema_model.parameters():
                    param.requires_grad = False
                ema_model.load_state_dict(copy.deepcopy(self.net.state_dict()))
                ema_model.eval()
                self.ema_models.append(ema_model)
            self.ema_models.eval()

            self.N_and_mu =  self.cfg.diffusion.N_and_mu
        else:
            raise ValueError(f'Preconditioning {cfg.diffusion.preconditioning} does not exist')
        # if self.cfg.testing.calc_inception:
        #     self.inception = InceptionScore()
        # if self.cfg.testing.calc_fid:
        #     self.fid = FrechetInceptionDistance(feature=2048) #.cuda()
            
        # FID and Inception Score instances are not registered as modules
        self.evaluation_attrs = {}
        for step in self.cfg.diffusion.denoise_steps_to_log:            
            if self.cfg.testing.calc_inception:
                self.evaluation_attrs[f'inception_student_{step}'] = InceptionScore().eval()
                for i in self.cfg.diffusion.ema_rate:
                    self.evaluation_attrs[f"inception_ema_{i}_{step}"] = InceptionScore().eval()
                self.evaluation_attrs[f'inception_target_{step}'] = InceptionScore().eval()

            if self.cfg.testing.calc_fid:
                self.evaluation_attrs[f'fid_student_{step}'] = FrechetInceptionDistance(feature=2048).eval()
                for i in self.cfg.diffusion.ema_rate:
                    self.evaluation_attrs[f"fid_ema_{i}_{step}"] = FrechetInceptionDistance(feature=2048)
                self.evaluation_attrs[f'fid_target_{step}'] = FrechetInceptionDistance(feature=2048).eval()


    def move_evaluation_attrs_to_device(self):
        # Move all eval attributes to the correct device
        for key, value in self.evaluation_attrs.items():
            self.evaluation_attrs[key] = value.to(self.device)

    def on_train_start(self):
        self.move_evaluation_attrs_to_device()

    def on_validation_start(self):
        self.move_evaluation_attrs_to_device()

    def training_step(self, batch, _):
        images, _ = batch
        if self.N_and_mu == "fixed":
            loss = self.loss_fn(net=self.net, net_ema=self.target_model, images=images, k=self.cfg.training.max_steps).mean()
        elif self.N_and_mu == "adaptive":
            loss = self.loss_fn(net=self.net, net_ema=self.target_model, images=images, k=self.global_step).mean()

        self.log("train_loss", loss, 
                    prog_bar=True,
                    logger=True,
                    on_step=True,
                    on_epoch=False,
                    )
        # return {'loss': loss}
        return loss

    def on_train_batch_end(self, out, batch, batch_idx):
        # Update EMA models manually at the end of each batch

        # Update target model
        with torch.no_grad():
            if self.N_and_mu == "fixed":
                target_ema = self.cfg.diffusion.mu_0
            elif self.N_and_mu == "adaptive":
                target_ema = self.net.mu(self.global_step)
        # update \theta_{-}
        self.update_ema(self.target_model, self.net, target_ema)

        # Update all the EMA models
        for ema_model, ema_rate in zip(self.ema_models, self.cfg.diffusion.ema_rate):
            self.update_ema(ema_model, self.net, ema_rate)

    def update_ema(self, ema_model, model, decay):
        for ema_param, param in zip(ema_model.parameters(), model.parameters()):
            ema_param.data.mul_(decay).add_(param.data, alpha=1 - decay)


    def generate_model_output(self, model, sampler, steps, prefix):
        images_to_log = []
        captions = []
        for step in steps:
            
            # TODO: this is temporary need to change and make proper when make common code for all
            if step ==1:
                step_adapt = [80]
            elif step ==2:
                step_adapt = [80, 75]


            latents = torch.randn(self.cfg.testing.samples, self.cfg.data.img_channels, self.cfg.data.img_resolution, self.cfg.data.img_resolution).to(self.device) 
            xh = multistep_consistency_sampling(model, latents=latents, t_steps=step_adapt)
            # xh = self.sampling(model=model, sampler=sampler, teacher= True if prefix == "Teacher" else False, step=step, num_samples=self.cfg.testing.samples, batch_size=self.cfg.testing.batch_size, ctm=True)
            xh = (xh * 0.5 + 0.5).clamp(0, 1)

            images_to_log.append(make_grid(xh, nrow=8).permute(1, 2, 0).cpu().numpy())
            captions.append(f"{prefix} {step} Steps")
        return images_to_log, captions

    def validation_step(self, batch, batch_idx):
        images, _ = batch
        if self.N_and_mu == "fixed":
            loss = self.loss_fn(net=self.net, net_ema=self.target_model, images=images, k=self.cfg.training.max_steps).mean()
        elif self.N_and_mu == "adaptive":
            loss = self.loss_fn(net=self.net, net_ema=self.target_model, images=images, k=self.global_step).mean()
        self.log("val_loss", loss, sync_dist=True)

        # Log one sample of the original and generated images for the first batch in the epoch
        if batch_idx == 0 and self.global_rank == 0:

            name = self.cfg.data.name
            images_to_log = []
            captions = []


            # Log student model outputs
            new_images_to_log, new_captions = self.generate_model_output(self.net, 'exact', self.cfg.diffusion.denoise_steps_to_log, "Student")
            images_to_log.extend(new_images_to_log)
            captions.extend(new_captions)

            # Log target model outputs
            new_images_to_log, new_captions = self.generate_model_output(self.target_model, 'exact', self.cfg.diffusion.denoise_steps_to_log, "Target")
            images_to_log.extend(new_images_to_log)
            captions.extend(new_captions)


            # Log EMA models outputs
            for ema_model, ema_rate in zip(self.ema_models, self.cfg.diffusion.ema_rate):
                new_images_to_log, new_captions = self.generate_model_output(ema_model, 'exact', self.cfg.diffusion.denoise_steps_to_log, f"ema_{ema_rate}")
                images_to_log.extend(new_images_to_log)
                captions.extend(new_captions)

            # log original data
            original_image = (images[:self.cfg.testing.samples] * 0.5 + 0.5).clamp(0, 1)
            original_grid = make_grid(original_image, nrow=8)
            images_to_log.append(original_grid.permute(1, 2, 0).cpu().numpy())
            captions.append("Original")
            ##################################

            self.logger.log_image(f"val_samples_{name}", images_to_log, caption=captions)


        ####################### Logging FID and Iscore ##################################

        x_batch = ((images + 1) * 127.5).clamp(0, 255).to(torch.uint8)

        # Update metric for student model
        self.update_metrics(self.net, self.cfg.diffusion.denoise_steps_to_log, 'student', x_batch)

        # Update metric for target model
        self.update_metrics(self.target_model, self.cfg.diffusion.denoise_steps_to_log, 'target', x_batch)

        # Update metric for ema models
        for ema_model, ema_rate in zip(self.ema_models, self.cfg.diffusion.ema_rate):
            self.update_metrics(ema_model, self.cfg.diffusion.denoise_steps_to_log, f'ema_{ema_rate}', x_batch)


    def update_metrics(self, model, steps, prefix, x_batch):
        for step in steps:

            # TODO: this is temporary need to change and make proper when make common code for all
            if step ==1:
                step_adapt = [80]
            elif step ==2:
                step_adapt = [80, 75]

            latents = torch.randn(self.cfg.testing.batch_size, self.cfg.data.img_channels, self.cfg.data.img_resolution, self.cfg.data.img_resolution).cuda() 
            xh = multistep_consistency_sampling(model, latents=latents, t_steps=step_adapt)
            # xh = self.sampling(model=model, step=step, num_samples=self.cfg.testing.batch_size, batch_size=self.cfg.testing.batch_size, ctm=True)
            xh = ((xh + 1) * 127.5).clamp(0, 255).to(torch.uint8).to(self.device)

            if self.cfg.testing.calc_inception:
                self.evaluation_attrs[f'inception_{prefix}_{step}'].update(xh)

            if self.cfg.testing.calc_fid:
                self.evaluation_attrs[f'fid_{prefix}_{step}'].update(x_batch, real=True)
                self.evaluation_attrs[f'fid_{prefix}_{step}'].update(xh, real=False)

    def on_validation_epoch_end(self):

        # Log Inception scores
        if self.cfg.testing.calc_inception:
            for step in self.cfg.diffusion.denoise_steps_to_log:
                iscore = self.evaluation_attrs[f'inception_student_{step}'].compute()[0]
                self.log(f'iscore/student_{step}', iscore, sync_dist=True)
                self.evaluation_attrs[f'inception_student_{step}'].reset()
                
                for i in self.cfg.diffusion.ema_rate:
                    iscore_ema = self.evaluation_attrs[f"inception_ema_{i}_{step}"].compute()[0]
                    self.log(f'iscore/ema_{i}_{step}', iscore_ema, sync_dist=True)
                    self.evaluation_attrs[f"inception_ema_{i}_{step}"].reset()
                
                iscore_target = self.evaluation_attrs[f'inception_target_{step}'].compute()[0]
                self.log(f'iscore/target_{step}', iscore_target, sync_dist=True)
                self.evaluation_attrs[f'inception_target_{step}'].reset()

        # Log FID scores
        if self.cfg.testing.calc_fid:
            for step in self.cfg.diffusion.denoise_steps_to_log:
                fid = self.evaluation_attrs[f'fid_student_{step}'].compute().item()
                self.log(f'fid/student_{step}', fid, sync_dist=True)
                self.evaluation_attrs[f'fid_student_{step}'].reset()
                
                for i in self.cfg.diffusion.ema_rate:
                    fid_ema = self.evaluation_attrs[f"fid_ema_{i}_{step}"].compute().item()
                    self.log(f'fid/ema_{i}_{step}', fid_ema, sync_dist=True)
                    self.evaluation_attrs[f"fid_ema_{i}_{step}"].reset()
                
                fid_target = self.evaluation_attrs[f'fid_target_{step}'].compute().item()
                self.log(f'fid/target_{step}', fid_target, sync_dist=True)
                self.evaluation_attrs[f'fid_target_{step}'].reset()

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
    
    
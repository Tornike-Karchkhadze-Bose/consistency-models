from diffusion import Diffusion
from main import dict2namespace
import yaml
from torchvision.utils import make_grid, save_image
from data import get_dataset
import torch
from torch.utils.data import DataLoader
from sampler import multistep_consistency_sampling
from torchmetrics.image.fid import FrechetInceptionDistance
from tqdm import tqdm
import click
import os

@torch.no_grad()
@click.command()
@click.option('--cfg', default='config.yml', help='Configuration File')
@click.option('--nrow', default=8, help='Row Count')
@click.option('--ncol', default=8, help='Column Count')
@click.option('--calc_fid', default=False, help='Calculate FID')
@click.option('--ckpt', default=None, help='Checkpoint Path')
def generate(cfg, nrow=8, ncol=8, calc_fid=False, ckpt=None):
    torch.manual_seed(123)

    with open(cfg, "r") as f:
        cfg = yaml.safe_load(f)
        cfg = dict2namespace(cfg)

    device_index = cfg.training.devices[0]
    # Create the torch device object based on the index
    device = torch.device(f'cuda:{device_index}' if torch.cuda.is_available() else 'cpu')


    model = Diffusion.load_from_checkpoint(ckpt, cfg=cfg).to(device)
    model.eval()

    # Extract the directory path
    directory_path = os.path.dirname(os.path.dirname(ckpt))

    # Create the "generated" folder within the extracted directory
    generated_folder_path = os.path.join(directory_path, 'generated')
    os.makedirs(generated_folder_path, exist_ok=True)

    latents = torch.randn(nrow * ncol, cfg.data.img_channels, cfg.data.img_resolution, cfg.data.img_resolution).to(device) #.cuda()
    xh = multistep_consistency_sampling(model.net_ema, latents=latents, t_steps=[80, 40, 20, 10, 5])
    xh = (xh * 0.5 + 0.5).clamp(0, 1)
    grid = make_grid(xh, nrow=nrow, padding=0)
    name = cfg.data.name
    save_image(grid, f"{generated_folder_path}/ct_{name}_sample_5step.png")

    # Sample 2 Steps
    xh = multistep_consistency_sampling(model.net_ema, latents=latents, t_steps=[80, 75])
    xh = (xh * 0.5 + 0.5).clamp(0, 1)
    grid = make_grid(xh, nrow=nrow, padding=0)
    save_image(grid, f"{generated_folder_path}/ct_{name}_sample_2step.png")

    # Sample 1 Step
    xh = multistep_consistency_sampling(model.net_ema, latents=latents, t_steps=[80])
    xh = (xh * 0.5 + 0.5).clamp(0, 1)
    grid = make_grid(xh, nrow=nrow, padding=0)
    save_image(grid, f"{generated_folder_path}/ct_{name}_sample_1step.png")

    if not calc_fid:
        return
    
    fid = FrechetInceptionDistance(feature=2048).to(device)
    dataloader = DataLoader(
        get_dataset(cfg.data.name, train=False), 
        batch_size=cfg.testing.batch_size, 
        shuffle=False, 
        num_workers=cfg.data.num_workers, 
        pin_memory=True, 
        persistent_workers=True,
    )

    for x_batch, _ in tqdm(dataloader):
        x_batch = ((x_batch.to(device) + 1) * 127.5).clamp(0, 255).to(torch.uint8)
        fid.update(x_batch, real=True)
    

    import torchvision.utils as vutils
    batch_size = 250
    for _ in tqdm(range(10000//batch_size)):
        latents = torch.randn(batch_size, cfg.data.img_channels, cfg.data.img_resolution, cfg.data.img_resolution).to(device)
        # latents = torch.randn(x_batch.shape[0], cfg.data.img_channels, cfg.data.img_resolution, cfg.data.img_resolution).to(device) 
        xh = multistep_consistency_sampling(model.net_ema, latents=latents, t_steps=[80])
        xh = ((xh + 1) * 127.5).clamp(0, 255).to(torch.uint8)
        fid.update(xh, real=False)
    
    print(f'FID@50K : {fid.compute().item()}')

if __name__ == '__main__':
    generate()
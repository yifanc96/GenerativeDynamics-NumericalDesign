"""Debug: track std_ratio evolution during RK4 sampling."""
import os, sys, argparse
p = argparse.ArgumentParser(); p.add_argument('--gpu', type=int, default=3)
args = p.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

import torch, math, numpy as np
sys.path.insert(0, '.')
from unet import Unet
import torch.nn as nn

device = torch.device('cuda')

data_raw, _ = torch.load('../NSdata/data_file.pt', weights_only=False)
data_raw = data_raw / 3.0679
data = data_raw.reshape(-1, 256, 256)
data = nn.functional.interpolate(data.unsqueeze(1), size=(128,128), mode='bilinear').squeeze(1)
data = data[:, None, :, :]
del data_raw
# Only keep small subsets on CPU
train_subset = data[:500].clone()  # enough for noise construction
truth_std = data[18000:18050].std().item()
del data

class Vel(nn.Module):
    def __init__(self, ch, mults):
        super().__init__()
        self.net = Unet(num_classes=1,in_channels=1,out_channels=1,dim=ch,dim_mults=mults,
                        resnet_block_groups=8,learned_sinusoidal_cond=True,random_fourier_features=False,
                        learned_sinusoidal_dim=32,attn_dim_head=32,attn_heads=4,use_classes=False)
    def forward(self,z,t): return self.net(z,t,classes=None)

N = 10

def make_noise(src, n):
    c = src - src.mean(dim=0, keepdim=True)
    xi = torch.randn(src.shape[0], n, device=src.device)
    return torch.einsum('nchw,nb->bchw', c, xi) / math.sqrt(src.shape[0])

def run(model_path, z0, label):
    model = Vel(32, (1,2,2,2)).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    tgrid = torch.linspace(1e-3, 0.999, 50).to(device)
    dt = tgrid[1] - tgrid[0]
    zt = z0.to(device)
    print(f'\n{label}:')
    for i, t_val in enumerate(tgrid):
        ones = torch.ones(N, device=device) * t_val
        k1 = model(zt, ones)
        k2 = model(zt+0.5*dt*k1, ones+0.5*dt)
        k3 = model(zt+0.5*dt*k2, ones+0.5*dt)
        k4 = model(zt+dt*k3, ones+dt)
        zt = zt + (dt/6)*(k1+2*k2+2*k3+k4)
        if i%5==0 or i==49:
            print(f'  t={t_val:.3f} std_ratio={zt.std().item()/truth_std:.4f} vel={k1.pow(2).sum(dim=(1,2,3)).mean().sqrt().item():.1f}')
    del model; torch.cuda.empty_cache()

# Data-dep noise
noise_src = train_subset[torch.randperm(500)[:100]].to(device)
z0_dep = make_noise(noise_src, N).cpu()
del noise_src; torch.cuda.empty_cache()
print(f'z0_dep std={z0_dep.std():.4f}, truth_std={truth_std:.4f}')
run('results/ns_data_dep_noise/model_final.pt', z0_dep, 'DataDep 2M')

z0_gauss = torch.randn(N,1,128,128)
print(f'z0_gauss std={z0_gauss.std():.4f}')
run('results/ns_gauss_base/model_final.pt', z0_gauss, 'Gauss-base 2M')

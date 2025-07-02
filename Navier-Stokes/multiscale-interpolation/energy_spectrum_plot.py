# spectrum plot
import torch
import numpy as np
from matplotlib import pyplot as plt

def smooth(scalars, weight = 0.6):
    """
    EMA implementation according to
    https://github.com/tensorflow/tensorboard/blob/34877f15153e1a2087316b9952c931807a122aa7/tensorboard/components/vz_line_chart2/line-chart.ts#L699
    """
    last = 0
    smoothed = scalars.clone().detach()
    num_acc = 0
    for index in range(scalars.shape[0]):
        next_val = scalars[index]
        last = last * weight + (1 - weight) * next_val
        num_acc += 1
        # de-bias
        debias_weight = 1
        if weight != 1:
            debias_weight = 1 - (weight**num_acc)
        smoothed_val = last / debias_weight
        smoothed[index] = smoothed_val

    return smoothed

def rfft_mesh(trajectory):

    nx = trajectory.shape[1]
    ny = trajectory.shape[2]

    kx_one_dim = torch.cat([torch.arange(start = 0, end = nx//2),torch.arange(start = -nx//2, end = 0) ], dim = 0)/2/np.pi
    kx = kx_one_dim.unsqueeze(-1).repeat(1,nx//2 + 1)

    ky_one_dim = torch.arange(start = 0, end = ny//2 + 1)/2/np.pi
    ky = ky_one_dim.unsqueeze(0).repeat(ny,1)

    return kx,ky


# def plot_avg_spectra_compare_Forecasting_torch(trajectory_1,trajectory_2, save_name = './energy_spectrum.jpg'):
    
#     vorticity_hat = torch.fft.rfftn(trajectory_1,dim=(1,2),norm = "forward")

#     kx, ky = rfft_mesh(trajectory_1)
#     kx = np.pi*2*kx
#     ky = np.pi*2*ky
#     k = torch.sqrt(abs(kx)**2 + abs(ky)**2)
    
#     laplace = ((np.pi*2) ** 2) * (torch.abs(kx)**2 + torch.abs(ky)**2)
#     laplace[0, 0] = 1
    
#     vorticity_hat_square = torch.abs(vorticity_hat)**2
#     vorticity_hat_square = torch.reshape(vorticity_hat_square,(trajectory_1.shape[0],-1))
    
#     vhat_square = (torch.abs(vorticity_hat)**2)/laplace
#     vhat_square = torch.reshape(vhat_square,(trajectory_1.shape[0],-1))
    

#     k = torch.ravel(k)
#     order = torch.argsort(k)

#     k = k[order]
#     energy = vhat_square[:,order]
#     enstrophy = vorticity_hat_square[:,order]

#     previous_k = k[0]
#     index_1 = 0

#     energy_spectrum_1 = torch.zeros_like(k)
#     enstrophy_spectrum_1 = torch.zeros_like(k)
#     non_repeating_k_1 = torch.zeros_like(k)

#     for i in range(k.shape[0]):

#         current_k = k[i]

#         if torch.abs(current_k - previous_k) < 1e-5:
#             energy_spectrum_1[index_1] += torch.mean(energy[:,index_1])
#             enstrophy_spectrum_1[index_1] += torch.mean(enstrophy[:,index_1])
#         else:
#             index_1 = index_1 + 1
#             non_repeating_k_1[index_1] = k[i]
#             energy_spectrum_1[index_1] = torch.mean(energy[:,index_1])
#             enstrophy_spectrum_1[index_1] = torch.mean(enstrophy[:,index_1])
#             previous_k = k[i]
            
#     fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(16, 8))
    
#     axs[0].loglog(non_repeating_k_1[:index_1].cpu().numpy(),smooth(energy_spectrum_1[:index_1]).cpu().numpy(),label = "Truth",color = '#1f77b4')
#     axs[0].loglog(non_repeating_k_1[:index_1].cpu().numpy(),energy_spectrum_1[:index_1].cpu().numpy(),color = '#1f77b4',alpha = 0.2)

#     axs[1].loglog(non_repeating_k_1[:index_1].cpu().numpy(),smooth(enstrophy_spectrum_1[:index_1]).cpu().numpy(),label = "Truth",color = '#1f77b4')
#     axs[1].loglog(non_repeating_k_1[:index_1].cpu().numpy(),enstrophy_spectrum_1[:index_1].cpu().numpy(),color = '#1f77b4',alpha = 0.2)
    
#     axs[0].set_xlabel("|k|")
#     axs[0].set_ylabel("Energy Spectra")
    
#     axs[1].set_xlabel("|k|")
#     axs[1].set_ylabel("Enstrophy Spectra")
    
#     vorticity_hat = torch.fft.rfftn(trajectory_2,dim=(1,2),norm = "forward")

#     kx, ky = rfft_mesh(trajectory_2)
#     kx = np.pi*2*kx
#     ky = np.pi*2*ky
#     k = np.sqrt(torch.abs(kx)**2 + torch.abs(ky)**2)
    
#     laplace = ((np.pi*2) ** 2) * (torch.abs(kx)**2 + torch.abs(ky)**2)
#     laplace[0, 0] = 1
    
#     vorticity_hat_square = torch.abs(vorticity_hat)**2
#     vorticity_hat_square = torch.reshape(vorticity_hat_square,(trajectory_1.shape[0],-1))
    
#     vhat_square = (torch.abs(vorticity_hat)**2)/laplace
#     vhat_square = torch.reshape(vhat_square,(trajectory_1.shape[0],-1))

#     k = torch.ravel(k)
#     order = torch.argsort(k)

#     k = k[order]
#     energy = vhat_square[:,order]
#     enstrophy = vorticity_hat_square[:,order]

#     previous_k = k[0]
#     index_2 = 0

#     energy_spectrum_2 = torch.zeros_like(k)
#     enstrophy_spectrum_2 = torch.zeros_like(k)
#     non_repeating_k_2 = torch.zeros_like(k)

#     for i in range(k.shape[0]):

#         current_k = k[i]

#         if torch.abs(current_k - previous_k) < 1e-5:
#             energy_spectrum_2[index_2] += torch.mean(energy[:,index_2])
#             enstrophy_spectrum_2[index_2] += torch.mean(enstrophy[:,index_2]) 
#         else:
#             index_2 = index_2 + 1
#             non_repeating_k_2[index_2] = k[i]
#             energy_spectrum_2[index_2] = torch.mean(energy[:,index_2])
#             enstrophy_spectrum_2[index_2] = torch.mean(enstrophy[:,index_2])
#             previous_k = k[i]
    
#     axs[0].loglog(non_repeating_k_2[:index_2].cpu().numpy(),smooth(energy_spectrum_2[:index_2]).cpu().numpy(),label = "Forecasting",color = '#2ca02c')
#     axs[1].loglog(non_repeating_k_2[:index_2].cpu().numpy(),smooth(enstrophy_spectrum_2[:index_2]).cpu().numpy(),label = "Forecasting",color = '#2ca02c')

#     axs[0].loglog(non_repeating_k_2[:index_2].cpu().numpy(),energy_spectrum_2[:index_2].cpu().numpy(),alpha = 0.2,color = '#2ca02c')
#     axs[1].loglog(non_repeating_k_2[:index_2].cpu().numpy(),enstrophy_spectrum_2[:index_2].cpu().numpy(),alpha = 0.2,color = '#2ca02c')  

#     axs[0].legend()
#     axs[1].legend()    
    
#     axs[0].set_ylim(1e-10,1e-2)
#     axs[1].set_ylim(1e-5,1e1)

    
#     # axs[0].axis('equal')
#     # axs[1].axis('equal')
# #     plt.show()
    
#     plt.savefig(save_name,dpi=300, bbox_inches='tight', transparent=True,facecolor='w')
    
#     return energy_spectrum_1[:index_1],energy_spectrum_2[:index_2]




# def plot_avg_spectra_compare_Forecasting_torch(trajectory_1,trajectory_2, save_name = './energy_spectrum.jpg'):

    
    
#     vorticity_hat = torch.fft.rfftn(trajectory_1,dim=(1,2),norm = "forward")
#     kx, ky = rfft_mesh(trajectory_1)
#     kx = np.pi*2*kx
#     ky = np.pi*2*ky
    
# #     k = torch.sqrt(abs(kx)**2 + abs(ky)**2)
#     k = 1.0*(torch.abs(kx) + torch.abs(ky))
#     laplace = ((np.pi*2) ** 2) * (torch.abs(kx)**2 + torch.abs(ky)**2)
#     laplace[0, 0] = 1

#     vorticity_hat_square = torch.abs(vorticity_hat)**2
#     vorticity_hat_square = torch.reshape(vorticity_hat_square,(trajectory_1.shape[0],-1))

#     vhat_square = (torch.abs(vorticity_hat)**2)/laplace
#     vhat_square = torch.reshape(vhat_square,(trajectory_1.shape[0],-1))


#     k = torch.ravel(k)
#     order = torch.argsort(k)

#     k = k[order]
#     energy = vhat_square[:,order]
#     enstrophy = vorticity_hat_square[:,order]

#     previous_k = k[0]
#     index_1 = 0

#     energy_spectrum_1 = torch.zeros_like(k)
#     enstrophy_spectrum_1 = torch.zeros_like(k)
#     non_repeating_k_1 = torch.zeros_like(k)

#     for i in range(k.shape[0]):

#         current_k = k[i]

#         if torch.abs(current_k - previous_k) < 1e-1:
# #             print(energy_spectrum_1[index_1])
# #             print(torch.mean(energy[:,index_1]))
#             energy_spectrum_1[index_1] += torch.mean(energy[:,i])
#             enstrophy_spectrum_1[index_1] += torch.mean(enstrophy[:,i])
#         else:
#             index_1 = index_1 + 1
#             non_repeating_k_1[index_1] = k[i]
#             energy_spectrum_1[index_1] += torch.mean(energy[:,i])
#             enstrophy_spectrum_1[index_1] += torch.mean(enstrophy[:,i])
#             previous_k = k[i]

#     fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(16, 8))

#     axs[0].loglog(non_repeating_k_1[:index_1].cpu().numpy(),(energy_spectrum_1[:index_1]).cpu().numpy(),label = "Truth",color = '#1f77b4')
# #     axs[0].loglog(non_repeating_k_1[:index_1].cpu().numpy(),energy_spectrum_1[:index_1].cpu().numpy(),color = '#1f77b4',alpha = 0.2)

#     axs[1].loglog(non_repeating_k_1[:index_1].cpu().numpy(),(enstrophy_spectrum_1[:index_1]).cpu().numpy(),label = "Truth",color = '#1f77b4')
# #     axs[1].loglog(non_repeating_k_1[:index_1].cpu().numpy(),enstrophy_spectrum_1[:index_1].cpu().numpy(),color = '#1f77b4',alpha = 0.2)

#     axs[0].set_xlabel("|k|")
#     axs[0].set_ylabel("Energy Spectra")

#     axs[1].set_xlabel("|k|")
#     axs[1].set_ylabel("Enstrophy Spectra")
    
    
    
#     vorticity_hat = torch.fft.rfftn(trajectory_2,dim=(1,2),norm = "forward")

#     kx, ky = rfft_mesh(trajectory_2)
#     kx = np.pi*2*kx
#     ky = np.pi*2*ky
# #     k = np.sqrt(torch.abs(kx)**2 + torch.abs(ky)**2)

# #     vorticity_hat = torch.fft.fft2(trajectory_2,dim=(1,2),norm = "forward")
# #     s = trajectory_2.shape[1]
# #     k_max = s // 2
# #     wavenumers = torch.cat((torch.arange(start=0, end=k_max, step=1), \
# #                             torch.arange(start=-k_max, end=0, step=1)), 0).repeat(s, 1)
# #     kx = wavenumers.transpose(0, 1)
# #     ky = wavenumers
#     k = 1.0*(torch.abs(kx) + torch.abs(ky))
    
    
#     laplace = ((np.pi*2) ** 2) * (torch.abs(kx)**2 + torch.abs(ky)**2)
#     laplace[0, 0] = 1

#     vorticity_hat_square = torch.abs(vorticity_hat)**2
#     vorticity_hat_square = torch.reshape(vorticity_hat_square,(trajectory_1.shape[0],-1))

#     vhat_square = (torch.abs(vorticity_hat)**2)/laplace
#     vhat_square = torch.reshape(vhat_square,(trajectory_1.shape[0],-1))

#     k = torch.ravel(k)
#     order = torch.argsort(k)

#     k = k[order]
#     energy = vhat_square[:,order]
#     enstrophy = vorticity_hat_square[:,order]

#     previous_k = k[0]
#     index_2 = 0
    
    
#     energy_spectrum_2 = torch.zeros_like(k)
#     enstrophy_spectrum_2 = torch.zeros_like(k)
#     non_repeating_k_2 = torch.zeros_like(k)

#     for i in range(k.shape[0]):

#         current_k = k[i]

#         if torch.abs(current_k - previous_k) < 1e-1:
#             energy_spectrum_2[index_2] += torch.mean(energy[:,i])
#             enstrophy_spectrum_2[index_2] += torch.mean(enstrophy[:,i]) 
#         else:
#             index_2 = index_2 + 1
#             non_repeating_k_2[index_2] = k[i]
#             energy_spectrum_2[index_2] += torch.mean(energy[:,i])
#             enstrophy_spectrum_2[index_2] += torch.mean(enstrophy[:,i])
#             previous_k = k[i]

#     axs[0].loglog(non_repeating_k_2[:index_2].cpu().numpy(),(energy_spectrum_2[:index_2]).cpu().numpy(),label = "Forecasting",color = '#2ca02c')
#     axs[1].loglog(non_repeating_k_2[:index_2].cpu().numpy(),(enstrophy_spectrum_2[:index_2]).cpu().numpy(),label = "Forecasting",color = '#2ca02c')

# #     axs[0].loglog(non_repeating_k_2[:index_2].cpu().numpy(),energy_spectrum_2[:index_2].cpu().numpy(),alpha = 0.2,color = '#2ca02c')
# #     axs[1].loglog(non_repeating_k_2[:index_2].cpu().numpy(),enstrophy_spectrum_2[:index_2].cpu().numpy(),alpha = 0.2,color = '#2ca02c')  

#     axs[0].legend()
#     axs[1].legend()
# #     axs[0].axis('equal')
#     plt.show()

# #     axs[0].set_ylim(1e-10,1e-2)
# #     axs[1].set_ylim(1e-4,1)
#     plt.savefig(save_name,dpi=300, bbox_inches='tight', transparent=True,facecolor='w')
    
#     return


import scipy.stats as stats
def get_energy_spectrum(vorticity_trajectory):
    vorticity_hat = torch.fft.fftn(vorticity_trajectory,dim=(1,2),norm = "forward")
    fourier_amplitudes = np.abs(vorticity_hat)**2 
    fourier_amplitudes = fourier_amplitudes.mean(dim=0)
    npix = vorticity_hat.shape[-1]
    kfreq = np.fft.fftfreq(npix) * npix
    kfreq2D = np.meshgrid(kfreq, kfreq)
    knrm = np.sqrt(kfreq2D[0]**2 + kfreq2D[1]**2)
    knrm = knrm.flatten()
    fourier_amplitudes = fourier_amplitudes.flatten()
    
    laplace = knrm ** 2
    laplace[0] = 1.0
    vhat_square = (fourier_amplitudes)/laplace
    
    kbins = np.arange(0.5, npix//2+1, 1.)
    kvals = 0.5 * (kbins[1:] + kbins[:-1])
    Abins_w, _, _ = stats.binned_statistic(knrm, fourier_amplitudes,
                                     statistic = "mean",
                                     bins = kbins)
    Abins_w *= np.pi * (kbins[1:]**2 - kbins[:-1]**2)
    
    Abins_v, _, _ = stats.binned_statistic(knrm, vhat_square,
                                     statistic = "mean",
                                     bins = kbins)
    Abins_v *= np.pi * (kbins[1:]**2 - kbins[:-1]**2)
    return kvals, Abins_w, Abins_v


def plot_avg_spectra_compare_Forecasting_torch(trajectory_1,trajectory_2, save_name = './energy_spectrum.jpg'):
    
    kvals, Abins_w1, Abins_v1 = get_energy_spectrum(trajectory_1)
    kvals, Abins_w2, Abins_v2 = get_energy_spectrum(trajectory_2)
    
    fig = plt.figure(figsize=(10,5))
    plt.subplot(121)
    plt.plot(kvals, Abins_w1, label = 'enstrophy spectrum of truth')
    plt.plot(kvals, Abins_w2, '--', label = 'enstrophy spectrum of forecasting')
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    # fig.savefig('lag1_enstrophy_spectrum.pdf', bbox_inches='tight',dpi=100,pad_inches=0.15)

    plt.subplot(122)
    plt.plot(kvals, Abins_v1, label = 'energy spectrum of truth')
    plt.plot(kvals, Abins_v2, '--', label = 'energy spectrum of forecasting')
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    plt.show()
    plt.savefig(save_name,dpi=300, bbox_inches='tight', transparent=True,facecolor='w')
    return





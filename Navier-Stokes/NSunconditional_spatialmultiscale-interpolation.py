import os
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
from torch.utils.data import DataLoader 
from torchvision import transforms as T
from torchvision.utils import make_grid
from PIL import Image
from matplotlib import pyplot as plt
import seaborn as sns
import wandb
import math
import argparse
import datetime
from time import time

################ data loading ################

def itr_merge(itrs):
    for itr in itrs:
        for v in enumerate(itr):
            yield v

def get_many_forecasting_dataloader(list_loc, time_lag, lo_size, hi_size, batch_size, train_test_split, subsampling_ratio = None):
    """
    get forecasting dataloader for 2D Stochastic NSE

    Parameters
    ----------
    list_loc: list of locations of data files, of dim num_trajectory*num_snapshots*res*res
    time_lag: creat forecasting data with time_lag 
        (i.e. x_t = data[:,:-time_lag,...] and x_{t+tau}=data[:,time_lag:,...])
    lo_size, hi_size: resolutions of x_t and x_{t+tau}
    batch_size: batch size
    train_test_split: a ratio representing the splitting of training/testing data
    subsampling_ratio: used for subsampling a small portion of data, for convenient small scale experiments
    """

    avg_pixel_norm = 3.0679163932800293 # avg data norm computed a priori
    
    print(f'[prepare dataset] time lag {time_lag}')
    
    list_len = len(list_loc)
    list_train_loaders = []
    list_test_loaders = []
    
    for i in range(list_len):
        print(f'---- [data set loc {i}] {list_loc[i]}')
        data_raw,time_raw = torch.load(list_loc[i])
        Ntj, Nts, Nx, Ny = data_raw.size() 
        tmp = torch.norm(data_raw,dim=(2,3),p='fro').mean() / np.sqrt(Nx*Ny)
        print(f'---- [dataset] average pixel norm of data set {tmp.item()}')
        data_raw = data_raw/avg_pixel_norm
        
        if time_lag > 0:
            data_pre = data_raw[:,:-time_lag,...]
            data_post = data_raw[:,time_lag:,...]
        else:
            data_pre = data_raw
            data_post = data_raw

        print(f'---- [processing] low resolution {lo_size}, high resolution {hi_size}')
        hi = torch.nn.functional.interpolate(data_post, size=(hi_size,hi_size),mode='bilinear').reshape([-1,hi_size,hi_size])
        lo = torch.nn.functional.interpolate(data_pre, size=(lo_size,lo_size),mode='bilinear')
        m = nn.Upsample(scale_factor=int(hi_size/lo_size), mode='nearest')
        lo = m(lo).reshape([-1,hi_size,hi_size])
        hi = hi[:,None,:,:] # make the data N C H W
        lo = lo[:,None,:,:] 
        
        if subsampling_ratio:
            hi = hi[:int(subsampling_ratio*hi.size()[0]),...]
            lo = lo[:int(subsampling_ratio*lo.size()[0]),...]
        
        num_train = int(lo.size()[0]*train_test_split)
        print(f'---- [processing] train_test_split {train_test_split}, num of training {num_train}, testing {lo.size()[0]-num_train}')

        train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(lo[:num_train,...],hi[:num_train,...]), batch_size=batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(lo[num_train:,...],hi[num_train:,...]), batch_size=batch_size, shuffle=False)
        
        list_train_loaders.append(train_loader)
        list_test_loaders.append(test_loader)
        del data_raw
    
    new_avg_pixel_norm = 1.0
    
    return list_train_loaders, list_test_loaders, avg_pixel_norm, new_avg_pixel_norm


################ interpolants and sampler ################

import torch
import matplotlib.pyplot as plt

class Hierarchical_Interpolant:
    def __init__(self, n, num_masks=None, device = None):
        """
        Initialize hierarchical masks for a 2^n × 2^n tensor with option to specify fewer masks.
        
        Parameters:
        n: int
            Power of 2 for tensor dimensions (tensor will be 2^n × 2^n)
        num_masks: int or None
            Number of masks to use. If None, uses n masks (default behavior).
            If num_masks < n, the coarsest masks will be aggregated.
        """
        self.n = n
        self.size = 2**n
        
        # Set the number of masks
        self.num_masks = n if num_masks is None else min(num_masks, n)
        
        # Create all scale masks first
        all_masks = []
        for k in range(n):
            # Create mask for scale k
            mask = torch.zeros((self.size, self.size), dtype=torch.float)
            
            # Generate coordinates
            y, x = torch.meshgrid(torch.arange(self.size), torch.arange(self.size), indexing='ij')
            
            # Set 1s at locations that belong to scale k
            if k == n-1:
                # Coarsest scale (2^(n-1) x 2^(n-1))
                scale_points = (y % (2**k) == 0) & (x % (2**k) == 0)
            else:
                # Scale k: points divisible by 2^k but not by 2^(k+1)
                div_by_current = (y % (2**k) == 0) & (x % (2**k) == 0)
                div_by_coarser = (y % (2**(k+1)) == 0) & (x % (2**(k+1)) == 0)
                scale_points = div_by_current & ~div_by_coarser
            
            mask[scale_points] = 1.0
            
            if device:
                self.device = device
            else:
                self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            mask = mask.to(self.device)
            
            all_masks.append(mask)
        
        # Create the final masks, potentially aggregating the coarsest scales
        self.masks = []
        
        if self.num_masks < n:
            # Keep the finest (num_masks-1) masks as they are
            for k in range(self.num_masks - 1):
                self.masks.append(all_masks[k])
            
            # Aggregate all the remaining coarsest scales into one mask
            aggregated_mask = torch.zeros_like(all_masks[0]).to(self.device)
            for k in range(self.num_masks - 1, n):
                aggregated_mask = aggregated_mask + all_masks[k]
            
            # Add the aggregated coarsest mask as the last mask
            self.masks.append(aggregated_mask)
        else:
            # Use all masks without aggregation
            self.masks = all_masks
    
    def freq_alpha(self, t):
        """
        Create mask with linear transitions between scales as t increases from 0 to num_masks.
        At t=0, all points are 1. As t increases, coarser scales transition to 0 first.
        
        Parameters:
        t: float or torch.Tensor
            Value(s) in [0, num_masks] controlling the transition
            
        Returns:
        torch.Tensor: Mask(s) with values between 0 and 1
        """
        # Handle different input types
        if isinstance(t, torch.Tensor):
            # Handle both single value tensors and batches
            if t.ndim == 0:  # If t is a scalar tensor
                t_clamped = torch.clamp(t, 0, self.num_masks)
                result = torch.zeros((1, self.size, self.size), dtype=torch.float).to(self.device)
                batch_size = 1
            else:  # If t is a batch
                # Get the batch size
                batch_size = t.shape[0]
                # Clamp t to [0, num_masks]
                t_clamped = torch.clamp(t, 0, self.num_masks)
                # Prepare a batch of results
                result = torch.zeros((batch_size, self.size, self.size), dtype=torch.float).to(self.device)
        else:
            # Scalar case
            t_clamped = max(0, min(t, self.num_masks))
            result = torch.zeros((1, self.size, self.size), dtype=torch.float).to(self.device)
            batch_size = 1
        
        for k in range(self.num_masks):
            # Reverse the order: num_masks-1-k gives index from finest to coarsest scale
            scale_index = self.num_masks - 1 - k
            
            # For each scale, determine its contribution
            if isinstance(t, torch.Tensor):
                # Condition 1: t <= k (scale fully active)
                cond1 = t_clamped <= k
                # Condition 2: k < t < k+1 (scale transitioning)
                cond2 = (t_clamped > k) & (t_clamped < k + 1)
                # Mask value for transitioning scales
                mask_val = 1 - (t_clamped - k)
                
                # Apply conditions with broadcasting
                scale_mask = self.masks[scale_index].unsqueeze(0)  # Add batch dimension
                
                # Reshape conditions properly for broadcasting
                if t.ndim == 0:
                    # For scalar tensor, no need to reshape
                    result += scale_mask * cond1
                    result += scale_mask * cond2 * mask_val
                else:
                    # For batches, reshape to (batch_size, 1, 1)
                    result += scale_mask * cond1.view(-1, 1, 1)
                    result += scale_mask * cond2.view(-1, 1, 1) * mask_val.view(-1, 1, 1)
            else:
                # Scalar case
                if t_clamped <= k:
                    result += self.masks[scale_index]
                elif t_clamped > k and t_clamped < k + 1:
                    result += self.masks[scale_index] * (1 - (t_clamped - k))
        
        return result

    def freq_beta(self, t):
        """
        Create mask with linear transitions between scales as t increases from 0 to num_masks.
        At t=0, all points are 0. As t increases, coarser scales transition to 1 first.
        
        Parameters:
        t: float or torch.Tensor
            Value(s) in [0, num_masks] controlling the transition
            
        Returns:
        torch.Tensor: Mask(s) with values between 0 and 1
        """
        # Handle different input types
        if isinstance(t, torch.Tensor):
            # Get the batch size
            batch_size = t.shape[0]
            # Clamp t to [0, num_masks]
            t_clamped = torch.clamp(t, 0, self.num_masks)
            # Prepare a batch of results
            result = torch.zeros((batch_size, self.size, self.size), dtype=torch.float).to(self.device)
        else:
            # Scalar case
            t_clamped = max(0, min(t, self.num_masks))
            result = torch.zeros((1, self.size, self.size), dtype=torch.float).to(self.device)
        
        for k in range(self.num_masks):
            # Reverse the order: num_masks-1-k gives index from finest to coarsest scale
            scale_index = self.num_masks - 1 - k
            
            # For each scale, determine its contribution
            if isinstance(t, torch.Tensor):
                # Condition 1: t >= k+1 (scale fully active)
                cond1 = t_clamped >= k + 1
                # Condition 2: k < t < k+1 (scale transitioning)
                cond2 = (t_clamped > k) & (t_clamped < k + 1)
                # Mask value for transitioning scales
                mask_val = t_clamped - k
                
                # Apply conditions with broadcasting
                scale_mask = self.masks[scale_index].unsqueeze(0)  # Add batch dimension
                
                # Use broadcasted conditions to add to result
                result += scale_mask * cond1.view(-1, 1, 1)
                result += scale_mask * cond2.view(-1, 1, 1) * mask_val.view(-1, 1, 1)
            else:
                # Scalar case
                if t_clamped >= k + 1:
                    result += self.masks[scale_index]
                elif t_clamped > k and t_clamped < k + 1:
                    result += self.masks[scale_index] * (t_clamped - k)
        
        return result
    
    def freq_alpha_dot(self, t):
        """
        Derivative of freq_alpha with respect to t.
        
        Parameters:
        t: float or torch.Tensor
            Value(s) in [0, num_masks] controlling the transition
            
        Returns:
        torch.Tensor: Derivative of mask at t
        """
        # Handle different input types
        if isinstance(t, torch.Tensor):
            # Handle both single value tensors and batches
            if t.ndim == 0:  # If t is a scalar tensor
                t_clamped = torch.clamp(t, 0, self.num_masks)
                result = torch.zeros((1, self.size, self.size), dtype=torch.float).to(self.device)
                batch_size = 1
            else:  # If t is a batch
                # Get the batch size
                batch_size = t.shape[0]
                # Clamp t to [0, num_masks]
                t_clamped = torch.clamp(t, 0, self.num_masks)
                # Prepare a batch of results
                result = torch.zeros((batch_size, self.size, self.size), dtype=torch.float).to(self.device)
            
            for k in range(self.num_masks):
                # Reverse the order: num_masks-1-k gives index from finest to coarsest scale
                scale_index = self.num_masks - 1 - k
                
                # Add batch dimension to scale mask
                scale_mask = self.masks[scale_index].unsqueeze(0)
                
                # t=0 is a special case for the finest scale (k=0)
                t_is_zero = (t_clamped == 0) & (k == 0)
                
                # t=num_masks is a special case for the coarsest scale (k=num_masks-1)
                t_is_num_masks = (t_clamped == self.num_masks) & (k == self.num_masks - 1)
                
                # Regular case: t in (k, k+1)
                t_in_range = (t_clamped > k) & (t_clamped < k + 1)
                
                # Integer case (except boundaries)
                t_is_k = (t_clamped == k) & (k > 0) & (k < self.num_masks)
                
                # Apply conditions with broadcasting
                if t.ndim == 0:
                    # For scalar tensor, no need to reshape
                    result = result - scale_mask * (t_is_zero | t_in_range | t_is_k | t_is_num_masks)
                else:
                    # For batches, reshape to (batch_size, 1, 1)
                    result = result - scale_mask * (t_is_zero | t_in_range | t_is_k | t_is_num_masks).view(-1, 1, 1)
                
        else:
            # Scalar case
            t_clamped = max(0, min(t, self.num_masks))
            result = torch.zeros((1, self.size, self.size), dtype=torch.float).to(self.device)
            
            for k in range(self.num_masks):
                # Reverse the order: num_masks-1-k gives index from finest to coarsest scale
                scale_index = self.num_masks - 1 - k
                
                # Special case for t=0 with the finest scale (k=0)
                if t_clamped == 0 and k == 0:
                    result -= self.masks[scale_index]
                # Special case for t=num_masks with the coarsest scale (k=num_masks-1)
                elif t_clamped == self.num_masks and k == self.num_masks - 1:
                    result -= self.masks[scale_index]
                # Regular case for t in (k, k+1)
                elif t_clamped > k and t_clamped < k + 1:
                    result -= self.masks[scale_index]
                # Integer case (except boundaries)
                elif t_clamped == k and k > 0 and k < self.num_masks:
                    result -= self.masks[scale_index]
        
        return result
    
    def freq_beta_dot(self, t):
        """
        Derivative of freq_beta with respect to t.
        
        Parameters:
        t: float or torch.Tensor
            Value(s) in [0, num_masks] controlling the transition
            
        Returns:
        torch.Tensor: Derivative of mask at t
        """
        # Handle different input types
        if isinstance(t, torch.Tensor):
            # Handle both single value tensors and batches
            if t.ndim == 0:  # If t is a scalar tensor
                t_clamped = torch.clamp(t, 0, self.num_masks)
                result = torch.zeros((1, self.size, self.size), dtype=torch.float).to(self.device)
                batch_size = 1
            else:  # If t is a batch
                # Get the batch size
                batch_size = t.shape[0]
                # Clamp t to [0, num_masks]
                t_clamped = torch.clamp(t, 0, self.num_masks)
                # Prepare a batch of results
                result = torch.zeros((batch_size, self.size, self.size), dtype=torch.float).to(self.device)
            
            for k in range(self.num_masks):
                # Reverse the order: num_masks-1-k gives index from finest to coarsest scale
                scale_index = self.num_masks - 1 - k
                
                # Add batch dimension to scale mask
                scale_mask = self.masks[scale_index].unsqueeze(0)
                
                # t=0 is a special case for the finest scale (k=0)
                t_is_zero = (t_clamped == 0) & (k == 0)
                
                # t=num_masks is a special case for the coarsest scale (k=num_masks-1)
                t_is_num_masks = (t_clamped == self.num_masks) & (k == self.num_masks - 1)
                
                # Regular case: t in (k, k+1)
                t_in_range = (t_clamped > k) & (t_clamped < k + 1)
                
                # Integer case (except t=num_masks)
                t_is_k = (t_clamped == k) & (k < self.num_masks)
                
                # Apply conditions with broadcasting
                if t.ndim == 0:
                    # For scalar tensor, no need to reshape
                    result = result + scale_mask * (t_is_zero | t_in_range | t_is_k | t_is_num_masks)
                else:
                    # For batches, reshape to (batch_size, 1, 1)
                    result = result + scale_mask * (t_is_zero | t_in_range | t_is_k | t_is_num_masks).view(-1, 1, 1)
                
        else:
            # Scalar case
            t_clamped = max(0, min(t, self.num_masks))
            result = torch.zeros((1, self.size, self.size), dtype=torch.float).to(self.device)
            
            for k in range(self.num_masks):
                # Reverse the order: num_masks-1-k gives index from finest to coarsest scale
                scale_index = self.num_masks - 1 - k
                
                # Special case for t=0 with the finest scale (k=0)
                if t_clamped == 0 and k == 0:
                    result += self.masks[scale_index]
                # Special case for t=num_masks with the coarsest scale (k=num_masks-1)
                elif t_clamped == self.num_masks and k == self.num_masks - 1:
                    result += self.masks[scale_index]
                # Regular case for t in (k, k+1)
                elif t_clamped > k and t_clamped < k + 1:
                    result += self.masks[scale_index]
                # Integer case (except boundaries)
                elif t_clamped == k and k < self.num_masks:
                    result += self.masks[scale_index]
        
        return result

    def It(self, D):
        """
        D is a dictionary containing 
        x0 = z0, 
        x1 = z1, 
        zt = I_t = alpha x_0 + beta x_1
        """
        z0 = D['z0']
        z1 = D['z1']
        t = D['t']

        aterm = self.freq_alpha(t)[:,None,:,:]*z0
        bterm = self.freq_beta(t)[:,None,:,:]*z1

        D['zt'] = aterm + bterm
        return D

    def Rt(self, D):
        """
        D is a dictionary containing 
        x0 = z0, 
        x1 = z1, 
        R_t = alpha_dot x_0 + beta_dot x_1
        """
        z0 = D['z0']
        z1 = D['z1']
        t = D['t']

        adotz0 = self.freq_alpha_dot(t)[:,None,:,:]*z0
        bdotz1 = self.freq_beta_dot(t)[:,None,:,:]*z1
        return adotz0 + bdotz1



import torch
import matplotlib.pyplot as plt
import numpy as np

class MaskConditionalVariance:
    def __init__(self, masks):
        """
        Initialize with a list of masks.
        
        Parameters:
        masks: list of torch.Tensor
            List of binary masks, ordered from finest to coarsest scale
        """
        self.masks = masks
        self.n_scales = len(masks)
        if self.n_scales > 0:
            self.size = masks[0].shape[0]  # Assuming square masks
    
    def estimate_variance(self, data):
        """
        Estimate the conditional variance for each mask given coarser scales.
        For each mask k, computes the variance of data at points in mask k
        that cannot be explained by the best linear prediction from coarser masks (k+1 to n).
        Uses all coarse points as features.
        
        Parameters:
        data: torch.Tensor
            Input data tensor with shape [batch, nx, ny] or [nx, ny]
            
        Returns:
        dict: Dictionary containing conditional variance estimates for each scale
        """
        if not isinstance(data, torch.Tensor):
            data = torch.tensor(data, dtype=torch.float32)
            
        # Handle batch dimension
        has_batch = len(data.shape) == 3
        
        # Ensure data has same spatial dimensions as masks
        if has_batch:
            if data.shape[-2:] != self.masks[0].shape:
                raise ValueError(f"Data spatial dimensions {data.shape[-2:]} don't match mask size {self.masks[0].shape}")
            batch_size = data.shape[0]
        else:
            if data.shape != self.masks[0].shape:
                raise ValueError(f"Data dimensions {data.shape} don't match mask size {self.masks[0].shape}")
            # Add dummy batch dimension if not present
            data = data.unsqueeze(0)
            batch_size = 1
        
        # Get total variance across batch dimension first
        # For each spatial location, compute variance across batch dimension
        if has_batch and batch_size > 1:
            # Compute variance along batch dimension (dim=0) for each spatial point
            spatial_variances = torch.var(data, dim=0)
            # Average these variances over all spatial locations
            total_var = torch.mean(spatial_variances)
        else:
            total_var = torch.tensor(0.0)  # Single batch has no variance
        
        # Initialize results
        results = {
            'total_variance': total_var.item(),
            'conditional_variances': [],
            'mask_point_counts': [],
            'scale_indices': []
        }
        
        # For each mask, compute variance conditioned on all coarser masks
        for k in range(self.n_scales):
            current_mask = self.masks[k]
            point_count = int(current_mask.sum().item())
            
            results['mask_point_counts'].append(point_count)
            results['scale_indices'].append(k)
            
            # For the coarsest mask, just use its variance directly (no conditioning)
            if k == self.n_scales - 1:
                mask_indices = torch.nonzero(current_mask, as_tuple=True)
                
                # Extract values for all batches at all mask points
                # Shape: [batch_size, n_points]
                mask_values = torch.stack([data[b][mask_indices] for b in range(batch_size)])
                
                # Compute variance across batch dimension for each point, then average
                if batch_size > 1:
                    point_variances = torch.var(mask_values, dim=0)
                    cond_var = torch.mean(point_variances).item()
                else:
                    cond_var = 0.0  # Single batch has no variance
                
                results['conditional_variances'].append(cond_var)
                continue
                
            # Create a mask for all coarser scales
            coarser_mask = torch.zeros_like(current_mask)
            for j in range(k+1, self.n_scales):
                # Use addition and then threshold to combine masks (equivalent to logical OR)
                coarser_mask = (coarser_mask + self.masks[j]) > 0
                
            # If no coarser points exist, just use the variance of current mask
            if coarser_mask.sum() == 0:
                mask_indices = torch.nonzero(current_mask, as_tuple=True)
                
                # Extract values for all batches at all mask points
                # Shape: [batch_size, n_points]
                mask_values = torch.stack([data[b][mask_indices] for b in range(batch_size)])
                
                # Compute variance across batch dimension for each point, then average
                if batch_size > 1:
                    point_variances = torch.var(mask_values, dim=0)
                    cond_var = torch.mean(point_variances).item()
                else:
                    cond_var = 0.0  # Single batch has no variance
                
                results['conditional_variances'].append(cond_var)
                continue
            
            # Extract current and coarser mask spatial indices
            current_indices = torch.nonzero(current_mask, as_tuple=True)
            coarser_indices = torch.nonzero(coarser_mask, as_tuple=True)
            
            # Number of points in each mask
            n_current_points = len(current_indices[0])
            n_coarser_points = len(coarser_indices[0])
            
            # Performance optimization for large n values
            # When n_coarser_points is large, we'll use a chunked approach to avoid memory issues
            
            # Get values at current mask points [batch_size, n_current_points]
            Y_all = torch.stack([data[b][current_indices] for b in range(batch_size)])
            
            # Decide on chunking strategy based on problem size
            # For large problems, process current points in chunks
            max_chunk_size = 1000  # Adjust based on available memory
            use_chunking = n_current_points > max_chunk_size
            
            try:
                # For large coarse point sets, we'll use a more efficient computation
                # that avoids explicitly forming the full X matrix
                if n_coarser_points > 1000:  # Large number of coarse points
                    # Instead of pseudoinverse, use normal equations with regularization
                    # This is more memory efficient for large feature sets
                    
                    # Get values at coarser mask points [batch_size, n_coarser_points]
                    coarser_values = torch.stack([data[b][coarser_indices] for b in range(batch_size)])
                    
                    # Add bias term as an additional feature
                    bias = torch.ones(batch_size, 1)
                    X_features = torch.cat([bias, coarser_values], dim=1)  # [batch_size, n_coarser_points+1]
                    
                    # Precompute X^T X and add regularization once
                    XtX = X_features.t() @ X_features
                    ridge_lambda = 1e-4 * torch.eye(XtX.shape[0])
                    XtX_reg = XtX + ridge_lambda
                    
                    # Process in chunks if needed
                    all_point_variances = []
                    
                    if use_chunking:
                        # Process chunks of current points
                        for chunk_start in range(0, n_current_points, max_chunk_size):
                            chunk_end = min(chunk_start + max_chunk_size, n_current_points)
                            chunk_size = chunk_end - chunk_start
                            
                            # Extract current chunk of Y values
                            Y_chunk = Y_all[:, chunk_start:chunk_end]  # [batch_size, chunk_size]
                            
                            # Compute X^T Y for this chunk
                            XtY = X_features.t() @ Y_chunk  # [n_features, chunk_size]
                            
                            # Solve normal equations for all points in chunk at once
                            # beta: [n_features, chunk_size]
                            beta_chunk = torch.linalg.solve(XtX_reg, XtY)
                            
                            # Make predictions for this chunk
                            Y_pred_chunk = X_features @ beta_chunk  # [batch_size, chunk_size]
                            
                            # Compute residuals for this chunk
                            residuals_chunk = Y_chunk - Y_pred_chunk  # [batch_size, chunk_size]
                            
                            # Compute variance across batch dimension for each point in chunk
                            if batch_size > 1:
                                chunk_variances = torch.var(residuals_chunk, dim=0)  # [chunk_size]
                                all_point_variances.append(chunk_variances)
                            else:
                                # Single batch case - no variance
                                all_point_variances.append(torch.zeros(chunk_size))
                    
                        # Combine all chunk variances
                        if all_point_variances:
                            point_variances = torch.cat(all_point_variances)
                            cond_var = torch.mean(point_variances).item()
                        else:
                            cond_var = 0.0
                    
                    else:
                        # No chunking needed, process all points at once
                        XtY = X_features.t() @ Y_all  # [n_features, n_current_points]
                        
                        # Solve normal equations for all points at once
                        beta_all = torch.linalg.solve(XtX_reg, XtY)
                        
                        # Make predictions for all points
                        Y_pred = X_features @ beta_all  # [batch_size, n_current_points]
                        
                        # Compute residuals for all points
                        residuals = Y_all - Y_pred  # [batch_size, n_current_points]
                        
                        # Compute variance across batch dimension for each point
                        if batch_size > 1:
                            point_variances = torch.var(residuals, dim=0)
                            cond_var = torch.mean(point_variances).item()
                        else:
                            cond_var = 0.0  # Single batch has no variance
                
                else:
                    # Original approach for smaller problems
                    # Get all coarser points' values across all batches: [batch_size, n_coarser_points+1]
                    X_all = torch.zeros((batch_size, n_coarser_points + 1))  # +1 for bias term
                    X_all[:, 0] = 1.0  # Bias term
                    for b in range(batch_size):
                        X_all[b, 1:] = data[b][coarser_indices]
                    
                    # Use normal equations instead of pseudoinverse for better performance
                    XtX = X_all.t() @ X_all
                    ridge_lambda = 1e-4 * torch.eye(XtX.shape[0])
                    XtX_reg = XtX + ridge_lambda
                    XtY = X_all.t() @ Y_all
                    
                    # Solve for all points at once
                    beta_all = torch.linalg.solve(XtX_reg, XtY)
                    
                    # Make predictions for all points
                    Y_pred = X_all @ beta_all
                    
                    # Compute residuals for all points
                    residuals = Y_all - Y_pred
                    
                    # Compute variance across batch dimension for each point
                    if batch_size > 1:
                        point_variances = torch.var(residuals, dim=0)
                        cond_var = torch.mean(point_variances).item()
                    else:
                        cond_var = 0.0  # Single batch has no variance
            
            except Exception as e:
                # Fallback using ridge regression with chunking for better memory management
                print(f"Warning: Matrix operation failed in scale {k}. Error: {str(e)}. Using fallback method.")
                
                # Only compute variances if we have multiple batches
                if batch_size > 1:
                    # Process in chunks to avoid memory issues
                    all_point_variances = []
                    chunk_size = 100  # Process points in smaller batches
                    
                    for chunk_start in range(0, n_current_points, chunk_size):
                        chunk_end = min(chunk_start + chunk_size, n_current_points)
                        chunk_variances = []
                        
                        for i in range(chunk_start, chunk_end):
                            y_i = Y_all[:, i]  # Values for current point i across all batches
                            
                            # Use a simpler model with stronger regularization for stability
                            # Add bias term
                            X_simple = torch.ones((batch_size, 2))  # Just use mean of coarser points
                            X_simple[:, 1] = torch.mean(data[:, coarser_indices[0], coarser_indices[1]], dim=1)
                            
                            # Simple closed-form ridge regression with strong regularization
                            XtX = X_simple.t() @ X_simple
                            ridge_lambda = 0.1  # Stronger regularization for stability
                            XtX_reg = XtX + ridge_lambda * torch.eye(XtX.shape[0])
                            Xty = X_simple.t() @ y_i
                            beta_i = torch.linalg.solve(XtX_reg, Xty)
                            
                            # Predict and compute residuals
                            y_pred_i = X_simple @ beta_i
                            residuals_i = y_i - y_pred_i
                            
                            # Compute variance for this point
                            point_var_i = torch.var(residuals_i).item()
                            chunk_variances.append(point_var_i)
                        
                        # Collect variances for this chunk
                        all_point_variances.extend(chunk_variances)
                    
                    # Average the point-wise variances
                    cond_var = sum(all_point_variances) / len(all_point_variances)
                else:
                    cond_var = 0.0  # Single batch has no variance
                
            except RuntimeError:
                # Fallback using ridge regression with chunking for better memory management
                print(f"Warning: Matrix operation failed in scale {k}. Using fallback method.")
                
                # Only compute variances if we have multiple batches
                if batch_size > 1:
                    # Process in chunks to avoid memory issues
                    all_point_variances = []
                    chunk_size = 100  # Process points in smaller batches
                    
                    for chunk_start in range(0, n_current_points, chunk_size):
                        chunk_end = min(chunk_start + chunk_size, n_current_points)
                        chunk_variances = []
                        
                        for i in range(chunk_start, chunk_end):
                            y_i = Y_all[:, i]  # Values for current point i across all batches
                            
                            # Use a simpler model with stronger regularization for stability
                            # Add bias term
                            X_simple = torch.ones((batch_size, 2))  # Just use mean of coarser points
                            X_simple[:, 1] = torch.mean(data[:, coarser_indices[0], coarser_indices[1]], dim=1)
                            
                            # Simple closed-form ridge regression with strong regularization
                            XtX = X_simple.t() @ X_simple
                            ridge_lambda = 0.1  # Stronger regularization for stability
                            XtX_reg = XtX + ridge_lambda * torch.eye(XtX.shape[0])
                            Xty = X_simple.t() @ y_i
                            beta_i = torch.linalg.solve(XtX_reg, Xty)
                            
                            # Predict and compute residuals
                            y_pred_i = X_simple @ beta_i
                            residuals_i = y_i - y_pred_i
                            
                            # Compute variance for this point
                            point_var_i = torch.var(residuals_i).item()
                            chunk_variances.append(point_var_i)
                        
                        # Collect variances for this chunk
                        all_point_variances.extend(chunk_variances)
                    
                    # Average the point-wise variances
                    cond_var = sum(all_point_variances) / len(all_point_variances)
                else:
                    cond_var = 0.0  # Single batch has no variance
            
            results['conditional_variances'].append(cond_var)
        
        return results

    def visualize_results(self, variance_results, figsize=(10, 6)):
        """
        Visualize the conditional variance results.
        
        Parameters:
        variance_results: dict
            Output from estimate_variance method
        figsize: tuple
            Figure size
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        scales = variance_results['scale_indices']
        variances = variance_results['conditional_variances']
        point_counts = variance_results['mask_point_counts']
        
        # Plot conditional variances
        ax.plot(scales, variances, 'o-', color='blue', label='Conditional Variance')
        
        # Add point count information
        for i, count in enumerate(point_counts):
            ax.annotate(f"{count} pts", 
                        (scales[i], variances[i]),
                        textcoords="offset points",
                        xytext=(0, 10),
                        ha='center')
        
        # Add total variance reference
        # ax.axhline(y=variance_results['total_variance'], color='red', linestyle='--', 
                  # label=f'Total Variance: {variance_results["total_variance"]:.4f}')
        
        # Add cumulative variance
        cumulative = np.cumsum(variances)
        ax.plot(scales, cumulative, 's--', color='green', label='Cumulative Variance')
        
        ax.set_xlabel('Mask Index (0=finest)')
        ax.set_ylabel('Variance')
        ax.set_title('Conditional Variance by Mask')
        ax.set_yscale('log')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return fig


################ network ################

import math
from unet import Unet
class Velocity(nn.Module):
    """ 
    This is a wrapper around any architecture
    The warpper handles the additional conditioning input by appending conditioning input as a channel
    """
    def __init__(self, config, interpolant):
        super(Velocity, self).__init__()
        self.config = config
        self._arch = Unet(
            num_classes = config.num_classes,
            in_channels = config.C + config.cond_channels,
            out_channels= config.C,
            dim = config.unet_channels,
            dim_mults = config.unet_dim_mults,
            resnet_block_groups = config.unet_resnet_block_groups,
            learned_sinusoidal_cond = config.unet_learned_sinusoidal_cond,
            random_fourier_features = config.unet_random_fourier_features,
            learned_sinusoidal_dim = config.unet_learned_sinusoidal_dim,
            attn_dim_head = config.unet_attn_dim_head,
            attn_heads = config.unet_attn_heads,
            use_classes = config.unet_use_classes,
        )
        num_params = np.sum([int(np.prod(p.shape)) for p in self._arch.parameters()])
        print("[Network] Num params in main arch for velocity is", f"{num_params:,}")
        
        self.interpolant = interpolant
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def forward(self, zt, t, y, cond=None):
        inputs = zt
        if cond is not None:
            """appending conditioning input as a channel""" 
            inputs = torch.cat([inputs, cond], dim=1)
        if not self.config.unet_use_classes:
            y = None
        out = self._arch(inputs, t, y)
        out_weighted = self.interpolant.freq_beta_dot(t)[:,None,...]*out
        return out_weighted
    

class Sampler:
    """
    sampler 
    self.interpolant: get information from the defined interpolants
    self.logger: information for uploading results to wandb, used in self.log_wandb_figure
    self.EM: EM for sampling
    """
    def __init__(self, config):
        self.config = config
        self.logger = Loggers(config)
        self.num_masks = config.num_masks
        return
    
    def wide(self, x):
        return x[:, None, None, None]

    def EM(self, D, model, steps = 60):
        print('[Sampler] Use EM samplers')
        init_condition = D['z0']
        tgrid = self.num_masks * torch.linspace(self.config.t_min_sample, self.config.t_max_sample, steps).type_as(init_condition)
        dt = tgrid[1] - tgrid[0]
        zt = D['z0']
        y = D['y']
        cond = D['cond']
        ones = torch.ones(zt.shape[0]).type_as(zt)
        for tscalar in tgrid:
            t_arr = tscalar * ones
            f = model(zt, t_arr, y, cond = cond) # note we condiition on init cond
            zt_mean = zt + f * dt
            zt = zt_mean
        return zt_mean

    # for plotting samples
    def to_grid(self, x, normalize):
        nrow = 1
        if normalize:
            kwargs = {'normalize' : True, 'value_range' : (-1, 1)}
        else:
            kwargs = {}
        return make_grid(x, nrow = nrow, **kwargs)

    # for logging sampled images to wandb
    def log_wandb_figure(self, sample, D, global_step):
        """
        plot conditioning input, x0, sampled x1, truth x1
        here D includes conditioning input, x0, truth x1
        and sample includes sampled x1
        finally, upload the figures to wandb
        """
        home = self.config.home
        if not os.path.exists(home + "images"):
            os.makedirs(home + "images")
        if self.config.use_wandb:
            def get_tensor_from_figures(tensor_w):
                plt.ioff()
                x = tensor_w.cpu()
                num = x.size()[0]
                for i in range(num):
                    WIDTH_SIZE = 3
                    HEIGHT_SIZE = 3
                    plt.ioff()
                    fig = plt.figure(figsize=(WIDTH_SIZE,HEIGHT_SIZE))
                    plt.imshow(x[i,0,...], cmap=sns.cm.icefire, vmin=-2, vmax=2.)
                    plt.axis('off')
                    plt.savefig(home + f'images/tmp{i}_{self.logger.log_base}.jpg', bbox_inches='tight')
                    plt.close("all") 
                    
                tensor_img = T.ToTensor()(Image.open(home + f'images/tmp1_{self.logger.log_base}.jpg'))
                C, H, W = tensor_img.size()
                final_tensor = torch.zeros((num,C,H,W))
                for i in range(num):
                    tensor_img = T.ToTensor()(Image.open(home + f'images/tmp{i}_{self.logger.log_base}.jpg'))
                    final_tensor[i,...] = tensor_img
                return final_tensor
            
            normalize = False
            
            num_train = config.num_reference_batch_train
            
            sample = get_tensor_from_figures(sample)
            sample_train = self.to_grid(sample[:num_train,...], normalize = normalize)
            sample_test = self.to_grid(sample[num_train:,...], normalize = normalize)
            
            z0 = get_tensor_from_figures(D['z0'])
            z0_train = self.to_grid(z0[:num_train,...], normalize = normalize)
            z0_test = self.to_grid(z0[num_train:,...], normalize = normalize)
            
            z1 = get_tensor_from_figures(D['z1'])
            z1_train = self.to_grid(z1[:num_train,...], normalize = normalize)
            z1_test = self.to_grid(z1[num_train:,...], normalize = normalize)
            
            both_train = torch.cat([z0_train, sample_train, z1_train], dim=-1)
            both_test = torch.cat([z0_test, sample_test, z1_test], dim=-1)
            
            wandb.log({'training-x0_sampledx1_truthx1': wandb.Image(both_train)}, step = global_step)
            wandb.log({'testing-x0_sampledx1_truthx1': wandb.Image(both_test)}, step = global_step)

    @torch.no_grad()
    def sample(self, D, model, global_step, wand_log = True):
        model.eval()
        if self.config.model_type == 'sde':
            zT = self.EM(D, model)
        else:
            assert False
        if self.config.use_wandb and wand_log:
            self.log_wandb_figure(zT, D, global_step)             


################ trainer ################

################ trainer ################

class Trainer:
    """
    Trainer
    self.prepare_dataset: create dataloaders using provided locations of data files
    self.time_dist: used for sampling time during training
    self.set_reference_batch: randomly picking a small training set and testing set, and test on them on the fly
    """
    def __init__(self, config):
        super(Trainer, self).__init__()
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.prepare_dataset(subsampling_ratio = config.data_subsampling_ratio)
        self.interpolant = Hierarchical_Interpolant(int(math.log2(config.hi_size)), config.num_masks)
        self.model = Velocity(self.config, self.interpolant)
        self.model.to(self.device)
        self.optimizer = self.get_optimizer(config)
        self.sampler = Sampler(config)
        self.time_dist = torch.distributions.Uniform(low=self.config.t_min_train, high=self.config.t_max_train)
        self.current_epoch = 0
        self.global_step = 0
        self.set_reference_batch(num_train = config.num_reference_batch_train, num_test = config.num_reference_batch_test) # for efficient test
        self.EMsteps = config.EMsteps
        self.home = config.home
        
        self.noise_covariance = noise_covariance
        self.noise_std = config.noise_strength * self.noise_covariance.sqrt()
        self.noise_std = self.noise_std.to(self.device)
        self.num_masks = config.num_masks
        print(f'[save_loc] will save all checkpoints and results to location to {self.home}')
        
    def prepare_dataset(self, subsampling_ratio = None):
        self.list_train_loaders, self.list_test_loaders, self.original_avg_pixel_norm, self.avg_pixel_norm = get_many_forecasting_dataloader(self.config.list_data_loc, self.config.time_lag, self.config.lo_size, self.config.hi_size, self.config.batch_size, self.config.train_test_split, subsampling_ratio)
        self.config.avg_pixel_norm = self.avg_pixel_norm
        self.config.original_avg_pixel_norm = self.original_avg_pixel_norm
        
    def set_reference_batch(self, num_train = 10, num_test = 10):
        xlo_train,xhi_train = next(iter(self.list_train_loaders[0]))
        xlo_test,xhi_test = next(iter(self.list_test_loaders[0]))
        self.ref_xlo = torch.cat((xlo_train[0:num_train,...],xlo_test[0:num_train,...]),0)
        self.ref_xhi = torch.cat((xhi_train[0:num_test,...],xhi_test[0:num_test,...]),0)
        
        
        
    @torch.no_grad()
    def prepare_batch(self, batch, time = 'unif', use_reference_batch = False):
        """
        D: a dictionary of x0, x1, z, and t, for interpolants
        here x0 = z0
             x1 = z1
             cond = z0
             t = uniform samples from [0,1]
             z = z_noise ~ N(0,I)
        """
        
        if use_reference_batch:
            xlo, xhi = self.ref_xlo, self.ref_xhi
        else:
            xlo, xhi = batch
            
        y = torch.zeros(xlo.size()[0]) # dummy variable; we do not use labels
        xlo, xhi, y = xlo.to(self.device), xhi.to(self.device), y.to(self.device)
        
        noise = self.noise_std[None,None,...] * torch.randn_like(xhi)
        noise = noise.to(self.device)
        
        
        D = {'z0': noise, 'z1': xhi, 'cond': None, 'y': y}
        
        if time == 'unif':
            D['t'] = self.num_masks * self.time_dist.sample(sample_shape = (xhi.shape[0],)).squeeze().type_as(D['z1'])
        else:
            assert False
        D = self.interpolant.It(D)
        return D
    
    def get_optimizer(self, config):
        if config.optimizer == "AdamW":
            print(f'[Optimizer] set up optimizer as {config.optimizer}')
            self.lr = self.config.base_lr
            return torch.optim.AdamW(self.model.parameters(), lr=self.config.base_lr)
    
    def target_function(self, D):
        if self.config.model_type == 'sde':
            target = self.interpolant.Rt(D)  
        else:
            assert False
        return target
    
    def loss_function(self, D):
        assert self.model.training
        model_out = self.model(D['zt'], D['t'], D['y'], cond = D['cond'])
        target = self.target_function(D)
        loss = (model_out - target).pow(2).sum(-1).sum(-1).sum(-1) # using full squared loss here
        return loss.mean()
    
    def clip_grad_norm(self, model, max_grad_norm = 1e+4):
        return torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm = max_grad_norm, norm_type = 2.0, error_if_nonfinite = False)

    def optimizer_one_step(self, max_grad_norm = 1e+4):
        self.clip_grad_norm(self.model, max_grad_norm = max_grad_norm)
        if self.global_step % self.config.print_loss_every == 0:
            grads = [ param.grad.detach().flatten() for param in self.model.parameters() if param.grad is not None]
            norm = torch.cat(grads).norm()
            print(f"[Training] Grad step {self.global_step}. Grad norm:{norm}")
            if self.config.use_wandb:
                wandb.log({"Gradnorm": norm}, step = self.global_step)
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.global_step += 1
    
    def adjust_learning_rate(self, optimizer):
        lr = self.lr
        if self.config.cosine_scheduler:
            scale = self.global_step / self.config.max_steps
            lr *= 0.5 * (1. + math.cos(math.pi * scale))
            print(f'[Cosine scheduler] lr is now {lr}')
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    
    def fit(self,):
        time_begin = time()
        print("[Training] starting training")
        self.test_model(first_batch_only = True)
        
        while self.global_step < self.config.max_steps:
            print(f"[Training] starting epoch {self.current_epoch}")
            
            for batch_idx, batch in itr_merge(self.list_train_loaders):
   
                if self.global_step >= self.config.max_steps:
                    break

                self.model.train()
                loss = self.loss_function(D = self.prepare_batch(batch, use_reference_batch = False))
                loss.backward()
                self.optimizer_one_step()     
                if self.global_step % self.config.sample_every == 0:
                    # for monitoring sampling, sampling on reference batch and uploading the results on the fly
                    D = self.prepare_batch(batch = None, use_reference_batch = True)
                    self.sampler.sample(D, self.model, self.global_step)

                if self.global_step % self.config.print_loss_every == 0:
                    total_mins = (time() - time_begin) / 60
                    print(f"[Training] Grad step {self.global_step}. Loss:{loss.item()}, finished in {total_mins:.2f} minutes")
                    if self.config.use_wandb:
                        wandb.log({"loss": loss.item()}, step=self.global_step)
                
                if self.global_step % self.config.save_model_every == 0:
                    self.save_model_checkpoint()
                if self.global_step % self.config.test_every == 0:
                    self.test_model(first_batch_only = True)
                    
            self.current_epoch += 1
            self.adjust_learning_rate(self.optimizer)
    
    #### below are testing functions, during the training processes or after training
    def save_model_checkpoint(self):
        
        if not os.path.exists(self.home + f"checkpoint/{logger.verbose_log_name}"):
            os.makedirs(self.home + f"checkpoint/{logger.verbose_log_name}")
        save_path = self.home + f"checkpoint/{logger.verbose_log_name}/model_step{self.global_step}.pth"
        torch.save(self.model.state_dict(), save_path)
        print(f'[Saving models] saving models to {save_path}')


    def sample_results(self, first_batch_only = True, which = 'test', EMsteps = 50):

        assert which in ['train', 'test']
        if which == 'test':

            loader_list = self.list_test_loaders
            #############################
            print(f'[Test] sampling on test data')
            tot_num = 0
            for batch_idx, batch in itr_merge(loader_list):
                num = batch[0].shape[0]
                tot_num = tot_num + num
                
            if first_batch_only:
                tot_num = self.config.batch_size
            print(f'[Test] in total {tot_num} test data, first batch only = {first_batch_only}')

        else:
            
            loader_list = self.list_train_loaders
            print(f'[Test] sampling on training data, first batch only = {first_batch_only}')
            tot_num = self.config.batch_size


        lo_size = config.lo_size
        hi_size = config.hi_size
        test_truth = torch.zeros(tot_num,1,hi_size,hi_size)
        test_input = torch.zeros_like(test_truth)
        test_result = torch.zeros_like(test_truth)

        self.model.eval()
        time_begin = time()
        cur_idx = 0

        for batch_idx, batch in itr_merge(loader_list):
            if first_batch_only and batch_idx > 0: break
            with torch.no_grad():
                num = batch[0].shape[0]
                D = self.prepare_batch(batch, use_reference_batch = False)
                test_input[cur_idx:cur_idx+num,...] = D['z0']
                test_truth[cur_idx:cur_idx+num,...] = D['z1']
                test_result[cur_idx:cur_idx+num,...] = self.sampler.EM(D, self.model, steps = EMsteps)
                total_mins = (time() - time_begin) / 60
                print(f'batch index {batch_idx}, finished in {total_mins:.2f} minutes')
                cur_idx = cur_idx + num

        inputs = test_input[:cur_idx,...]
        truth = test_truth[:cur_idx,...]
        results =  test_result[:cur_idx,...]
        results = torch.cat([inputs, truth, results], dim = 1) * self.original_avg_pixel_norm
        return results
    
    def plot_spectra(self, results, which = 'test'):
    
        assert which in ['train', 'test']

        # energy spectrum
        if not os.path.exists(self.home + "images"):
            os.makedirs(self.home + "images")
        
        from energy_spectrum_plot import plot_avg_spectra_compare_Forecasting_torch as plot_spectra

        spectrum_save_name = self.home + f"images/{logger.verbose_log_name}_spectrum_test_on_{which}.jpg"
        plot_spectra(results[:,1,...], results[:,2,...], save_name = spectrum_save_name)
        print(f"spectrum plot saved to {spectrum_save_name}")
        
        tensor_img = T.ToTensor()(Image.open(spectrum_save_name))

        f = lambda x: wandb.Image(x[None,...])
        if config.use_wandb:
            wandb.log({f'energy spectrum (test on {which} data)': f(tensor_img)}, step = self.global_step) 
    
    def compute_norm(self, results, which = 'test'):
        truth = results[:,1,...]
        forecast = results[:,2,...]
        truth_norm = torch.norm(truth,dim=(1,2),p='fro').mean() / self.config.hi_size
        forecast_norm = torch.norm(forecast,dim=(1,2),p='fro').mean() / self.config.hi_size
        relerr = abs(truth_norm-forecast_norm)/truth_norm
        print(f"[testing norms] on {which} data, truth norm is {truth_norm}, forecast norm is {forecast_norm}, relative err {relerr}")
        if self.config.use_wandb:
            wandb.log({f"norm err (on {which} data)":  relerr}, step = self.global_step)
    
    def test_model(self, first_batch_only = True):

        train_results = self.sample_results(first_batch_only = first_batch_only, which = 'train', EMsteps = self.EMsteps)
        test_results = self.sample_results(first_batch_only = first_batch_only, which = 'test', EMsteps = self.EMsteps)

        # norm test
        self.compute_norm(train_results, which = 'train')
        self.compute_norm(test_results, which = 'test')
        
        # spectrum test
        self.plot_spectra(train_results, which = 'train')
        self.plot_spectra(test_results, which = 'test')       

                 

################ logger ################

class Loggers:
    """
    self.log_base: date string for naming of logging files
    self.log_name: detailed information of the experiment, used for naming of logging files
    self.verbose_log_name: more verbose version for naming
    """
    def __init__(self, config):
        date = str(datetime.datetime.now())
        self.log_base = date[date.find("-"):date.rfind(".")].replace("-", "").replace(":", "").replace(" ", "_")
        self.log_name = 'noise' + str(config.noise_strength) + 'lo' + str(config.lo_size) + 'hi' + str(config.hi_size) + '_' + self.log_base
        self.verbose_log_name = 'uncond_GaussODEmsinterp_nummask'+ str(config.num_masks) + 'numdata' + str(config.num_dataset) + 'noise' + str(config.noise_strength) + 'lo' + str(config.lo_size) + 'hi' + str(config.hi_size) + 'sz' + str(config.base_lr).replace(".","") + 'max' + str(config.max_steps) + '_' + self.log_base
        
    def is_type_for_logging(self, x):
        if isinstance(x, int):
            return True
        elif isinstance(x, float):
            return True
        elif isinstance(x, bool):
            return True
        elif isinstance(x, str):
            return True
        elif isinstance(x, list):
            return True
        elif isinstance(x, set):
            return True
        else:
            return False

    def setup_wandb(self, config):
        if config.use_wandb:
            config.wandb_run = wandb.init(
                    project=config.wandb_project,
                    entity=config.wandb_entity,
                    resume=None,
                    id    =None,
                    name = self.verbose_log_name,
            )
            wandb.run.log_code(".")

            for key in vars(config):
                item = getattr(config, key)
                if self.is_type_for_logging(item):
                    setattr(wandb.config, key, item)
                    print(f'[Config] {key}: {item}')
            print("[wandb] finished wandb setup")
        else:
            print("[wandb] not using wandb setup")


################ config ################


class Config:
    def __init__(self,list_data_loc,home):
        
        # use wandb for logging
        self.use_wandb = True
        self.wandb_project = 'interpolants_forecasting_new'
        self.wandb_entity = 'yifanc96'
        self.home = home # for storing checkpoints
        
        # data
        self.list_data_loc = list_data_loc
        self.num_dataset = len(list_data_loc)
        self.C = 1
        self.num_classes = 1
        self.lo_size = 128
        self.hi_size = 128
        self.batch_size = 100
        self.num_workers = 4
        self.train_test_split = 0.9
        self.delta_t = 0.5
        self.time_lag = 2  # note that the physical lag = time_lag * delta_t = 0.5*time_lag
        self.noise_strength = 1.0
        self.data_subsampling_ratio = 1.0  # use a small amount of data, for sanity check of the code
        
        # training
        self.optimizer = 'AdamW'
        self.cosine_scheduler = True
        self.model_type = 'sde'
        self.base_lr = 2*1e-4
        self.max_steps = 100
        self.t_min_train = 0
        self.t_max_train = 1
        self.t_min_sample = 0
        self.t_max_sample = 1
        self.EMsteps = 100
        self.print_loss_every = 20 
        self.print_gradnorm_every =  20
        self.num_reference_batch_train = 10
        self.num_reference_batch_test = 10
        self.sample_every = 200 # sampling on reference batch every # iterations
        self.test_every = 200 # test energy spectrum and norm on reference batch every # iterations
        self.save_model_every = 2000 # save model checkpoints every # iterations
        
        # architecture
        self.num_masks = 4
        self.unet_use_classes = False
        # self.model_size = 'small'
        self.model_size = 'medium'
        if self.model_size == 'small':
            self.unet_channels = 8
            self.unet_dim_mults = (1, 1, 1, 1)
            self.unet_resnet_block_groups = 8
            self.unet_learned_sinusoidal_dim = 8
            self.unet_attn_dim_head = 8
            self.unet_attn_heads = 1
            self.unet_learned_sinusoidal_cond = False
            self.unet_random_fourier_features = False
        
        elif self.model_size == 'medium':
            self.unet_channels = 32
            self.unet_dim_mults = (1, 2, 2, 2)
            self.unet_resnet_block_groups = 8
            self.unet_learned_sinusoidal_dim = 32
            self.unet_attn_dim_head = 32
            self.unet_attn_heads = 4
            self.unet_learned_sinusoidal_cond = True
            self.unet_random_fourier_features = False
   
        elif self.model_size == 'large':
            self.unet_channels = 128
            self.unet_dim_mults = (1, 2, 2, 2)
            self.unet_resnet_block_groups = 8
            self.unet_learned_sinusoidal_dim = 32
            self.unet_attn_dim_head = 64
            self.unet_attn_heads = 4
            self.unet_learned_sinusoidal_cond = True
            self.unet_random_fourier_features = False
        else:
            assert False
        # self.cond_channels = self.C
        self.cond_channels = 0
        # dimension of the conditional channel; here conditioned on z_0
        # the conditioned term is appended to the input (so the final channel dim = cond_channels + input_channels)

def get_parser():
    parser = argparse.ArgumentParser(description='PyTorch framework for stochastic interpolants')
    parser.add_argument("--data_subsampling_ratio", type=float, default=1.0)
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--noise_strength", type=float, default=10.0)
    parser.add_argument("--base_lr", type=float, default=2e-4)
    parser.add_argument("--lo_size", type=int, default=128)
    parser.add_argument("--hi_size", type=int, default=128)
    parser.add_argument("--time_lag", type=int, default=2)
    parser.add_argument("--max_steps", type=int, default=50000)
    parser.add_argument("--sample_every", type=int, default=1000)
    parser.add_argument("--test_every", type=int, default=1000)
    parser.add_argument("--save_model_every", type=int, default=2000)
    parser.add_argument("--num_dataset",type=int,default=5)
    parser.add_argument("--num_masks",type=int,default=4)
    parser.add_argument('--use_wandb', type = int, default = 1) # 1 is use_wandb, and 0 is not use_wandb
    args = parser.parse_args()
    return args

random_seed = 0
torch.manual_seed(0)
np.random.seed(random_seed)
args = get_parser()
args.use_wandb = bool(args.use_wandb)


###### data location
# list_data_loc = ["/data_file.pt"]
list_suffix = [f"0{i}" for i in np.arange(1,args.num_dataset+1)]
# list_data_loc = [f"/scratch/mh5113/forecasting/new_simulations_lag_05_term" + i + ".pt" for i in list_suffix]
list_data_loc = [f"/scratch/yc3400/forecasting/NSEdata/data_file" + i + ".pt" for i in list_suffix]
# /scratch/yc3400/forecasting/NSEdata/
if args.num_dataset < len(list_data_loc): 
    list_data_loc = list_data_loc[:args.num_dataset]
    args.num_dataset = len(list_data_loc)

##### checkpoint and image storage location
# home = "./"
home = "/scratch/yc3400/forecasting/" 

config = Config(list_data_loc, home)
for arg in vars(args):
    print(f'[Argparse] change config {arg} to {getattr(args, arg)}')
    setattr(config, arg, getattr(args, arg))
logger = Loggers(config)
logger.setup_wandb(config)


#### compute conditional variance to decide the noise covariance
hi_size = config.hi_size
z1 = torch.load(list_data_loc[0])[0]
z1 = torch.nn.functional.interpolate(z1, size=(hi_size,hi_size),mode='bilinear')
z1 = z1.reshape(-1,hi_size,hi_size)
avg_pixel_norm = 3.0679163932800293
z1 = z1/avg_pixel_norm

bs, nx, nx = z1.shape

# Use masks from the Hierarchical_Interpolant
n = int(math.log2(nx))
num_masks = config.num_masks
hi = Hierarchical_Interpolant(n, num_masks = num_masks, device = 'cpu')
masks = hi.masks

# Create the variance estimator
estimator = MaskConditionalVariance(masks)

variance_results = estimator.estimate_variance(z1)

# Visualize
fig = estimator.visualize_results(variance_results)
plt.show()

# Print detailed results
print("\nConditional Variance Analysis:")
# print(f"Total variance: {variance_results['total_variance']:.6f}")
print("\nMask-specific conditional variances:")
for i, (idx, var, count) in enumerate(zip(
    variance_results['scale_indices'], 
    variance_results['conditional_variances'],
    variance_results['mask_point_counts']
)):
    print(f"Mask {idx} ({count} points): {var:.6f}")

# Calculate percentage of total variance
total_explained = sum(variance_results['conditional_variances'])
print("\nVariance distribution across masks:")
for i, (idx, var) in enumerate(zip(
    variance_results['scale_indices'],
    variance_results['conditional_variances']
)):
    pct = (var / total_explained) * 100 if total_explained > 0 else 0
    print(f"Mask {idx}: {pct:.2f}% of explained variance")

noise_covariance = torch.zeros(nx,nx)
for i in range(len(masks)):
    noise_covariance += variance_results['conditional_variances'][i]*masks[i]

noise = noise_covariance.sqrt()[None,...] * torch.randn(20000, nx,nx)

print((noise**2).mean(dim=0).mean())
print((z1**2).mean(dim=0).mean())
del z1
del noise


trainer = Trainer(config)
trainer.fit()
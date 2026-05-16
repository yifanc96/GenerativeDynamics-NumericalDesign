"""Stochastic ensemble Kalman filter (EnKF) update on a 2D vorticity field.

Two flavours:
  - vanilla : full sample covariance (rank-deficient when N << state dim)
  - localised : Schur product with a Gaspari-Cohn 5th-order taper of radius `loc_radius`
                (units = grid pixels). Standard NWP fix for small-N rank deficiency.

Both use multiplicative inflation: x <- x_mean + alpha (x - x_mean) before the update.

Observation operator: AvgPool factor `obs_factor` (matches observation.py).
"""
import torch
import torch.nn.functional as F


def gaspari_cohn(r):
    """5th-order GC taper, r in [0, infty). Compactly supported on [0, 2]; 0 outside."""
    r = r.clamp_min(0.0)
    out = torch.zeros_like(r)
    m1 = r <= 1.0
    m2 = (r > 1.0) & (r <= 2.0)
    r1 = r[m1]
    out[m1] = (-(r1 ** 5) / 4.0 + (r1 ** 4) / 2.0
               + 5.0 * (r1 ** 3) / 8.0 - 5.0 * (r1 ** 2) / 3.0 + 1.0)
    r2 = r[m2]
    out[m2] = ((r2 ** 5) / 12.0 - (r2 ** 4) / 2.0
               + 5.0 * (r2 ** 3) / 8.0 + 5.0 * (r2 ** 2) / 3.0
               - 5.0 * r2 + 4.0 - 2.0 / (3.0 * r2))
    return out


def _torus_distance(coords_a, coords_b, H):
    """Periodic |a - b| on H x H torus, returns (Na, Nb) distance."""
    da = (coords_a[:, None, :] - coords_b[None, :, :]).abs()
    da = torch.minimum(da, H - da)
    return da.pow(2).sum(dim=-1).sqrt()


def make_localisation_matrix(H, h_obs, factor, loc_radius, device):
    """Return (state_dim, obs_dim) Schur taper. State pixels & obs pixels lie on H x H
    and h_obs x h_obs grids respectively (both periodic). loc_radius is in *state-pixel* units."""
    yy, xx = torch.meshgrid(torch.arange(H, device=device), torch.arange(H, device=device), indexing='ij')
    state_coords = torch.stack([yy.flatten(), xx.flatten()], dim=-1).float()              # (H*H, 2)
    yo, xo = torch.meshgrid(torch.arange(h_obs, device=device), torch.arange(h_obs, device=device), indexing='ij')
    # Each obs pixel sits at the centre of its source factor x factor block.
    obs_coords = torch.stack([yo.flatten().float() * factor + (factor - 1) / 2.0,
                              xo.flatten().float() * factor + (factor - 1) / 2.0], dim=-1)
    d_so = _torus_distance(state_coords, obs_coords, H)
    d_oo = _torus_distance(obs_coords,   obs_coords, H)
    rho_state_obs = gaspari_cohn(d_so / loc_radius)
    rho_obs_obs   = gaspari_cohn(d_oo / loc_radius)
    return rho_state_obs, rho_obs_obs


def enkf_update(x, y_obs, obs_factor, sigma_y, inflation=1.0,
                localise=False, loc_radius=8.0, generator=None):
    """Stochastic EnKF.

    x       : (N, 1, H, H)   forecast ensemble (already de-normalised or normalised — caller chooses;
                              EnKF is scale-invariant if y_obs and sigma_y are in matching units).
    y_obs   : (1, 1, h, h)   single observation (h = H // obs_factor).
    Returns updated ensemble (N, 1, H, H).
    """
    N, _, H, _ = x.shape
    h_obs = H // obs_factor
    device = x.device

    # Inflation around the ensemble mean
    if inflation > 1.0:
        m = x.mean(dim=0, keepdim=True)
        x = m + inflation * (x - m)

    # Apply observation operator to each ensemble member
    Hx = F.avg_pool2d(x.reshape(N, 1, H, H), obs_factor, stride=obs_factor)        # (N, 1, h, h)

    # Flatten to (N, state_dim) and (N, obs_dim)
    state_dim = H * H
    obs_dim = h_obs * h_obs
    Xf = x.reshape(N, state_dim)
    Yf = Hx.reshape(N, obs_dim)
    # Anomalies
    xm = Xf.mean(dim=0, keepdim=True)
    ym = Yf.mean(dim=0, keepdim=True)
    Xa = Xf - xm                                         # (N, state)
    Ya = Yf - ym                                         # (N, obs)

    # Sample covariances (× 1/(N-1))
    n1 = max(N - 1, 1)
    Cxy = Xa.T @ Ya / n1                                 # (state, obs)
    Cyy = Ya.T @ Ya / n1                                 # (obs, obs)

    if localise:
        rho_so, rho_oo = make_localisation_matrix(H, h_obs, obs_factor, loc_radius, device)
        Cxy = Cxy * rho_so
        Cyy = Cyy * rho_oo

    # K = Cxy (Cyy + R)^-1, R = sigma_y^2 I
    R = (sigma_y ** 2) * torch.eye(obs_dim, device=device, dtype=Cxy.dtype)
    Kgain = torch.linalg.solve((Cyy + R).T, Cxy.T).T     # (state, obs)

    # Perturbed observations
    if generator is None:
        eps = sigma_y * torch.randn(N, obs_dim, device=device, dtype=Xf.dtype)
    else:
        eps = sigma_y * torch.randn(N, obs_dim, device=device, dtype=Xf.dtype, generator=generator)
    y_perturbed = y_obs.reshape(1, obs_dim) + eps        # (N, obs)

    innov = y_perturbed - Yf                             # (N, obs)
    Xf_new = Xf + innov @ Kgain.T                        # (N, state)
    return Xf_new.reshape(N, 1, H, H)

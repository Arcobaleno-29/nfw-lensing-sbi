import numpy as np
import torch
from cosmology import Cosmology
from nfw import NFWHalo

def shear_profile(theta, R, z_l, Sigma_crit, cosmology: Cosmology):
    halo = NFWHalo(theta=theta, z_l=z_l, cosmology=cosmology)

    Sigma, Sigma_mean = halo.surface_densities(R)
    
    DeltaSigma = Sigma_mean - Sigma

    kappa = Sigma / Sigma_crit
    gamma_t = DeltaSigma / Sigma_crit
    g_t = gamma_t / (1 - kappa)

    return torch.as_tensor(g_t, dtype=torch.float32)

#1D profile
def simulate(theta, R, z_l, Sigma_crit, cosmology, sigma):
    g_t = shear_profile(theta, R, z_l, Sigma_crit, cosmology)
    noise_scale = torch.as_tensor(sigma, dtype=g_t.dtype, device=g_t.device)
    return g_t + noise_scale * torch.randn_like(g_t)

#2D profile
def simulate_map(
    theta, x, y, centre, mask,
    z_l, Sigma_crit, cosmology,
    N_per_pixel, sigma_galaxy=0.3,
):
    xc, yc = centre
    R = np.hypot(x - xc, y - yc)
    phi = torch.as_tensor(
        np.arctan2(y - yc, x - xc),
        dtype=torch.float32,
    )

    # Accept one count for all pixels or a count map.
    counts = torch.broadcast_to(
        torch.as_tensor(N_per_pixel, dtype=torch.float32),
        R.shape,
    )
    valid = torch.as_tensor(mask, dtype=torch.bool) & (counts > 0)

    # Uncertainty per component after averaging galaxies.
    sigma_pixel = torch.zeros_like(counts)
    sigma_pixel[valid] = sigma_galaxy / torch.sqrt(counts[valid])

    # Expected tangential shear at each valid pixel centre.
    g_t = torch.zeros_like(counts)
    g_t[valid] = shear_profile(
        theta=theta,
        R=R[valid.numpy()],
        z_l=z_l,
        Sigma_crit=Sigma_crit,
        cosmology=cosmology,
    )

    g1 = -g_t * torch.cos(2 * phi)
    g2 = -g_t * torch.sin(2 * phi)

    # Independent noise for the two components.
    g1_obs = g1 + sigma_pixel * torch.randn_like(g1)
    g2_obs = g2 + sigma_pixel * torch.randn_like(g2)

    g_true = torch.stack([g1, g2])
    g_obs = torch.stack([g1_obs, g2_obs])

    return g_obs, g_true, sigma_pixel, valid
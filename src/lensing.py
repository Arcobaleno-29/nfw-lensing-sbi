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

def simulate(theta, R, z_l, Sigma_crit, cosmology, sigma):
    g_t = shear_profile(theta, R, z_l, Sigma_crit, cosmology)
    noise_scale = torch.as_tensor(sigma, dtype=g_t.dtype, device=g_t.device)
    return g_t + noise_scale * torch.randn_like(g_t)
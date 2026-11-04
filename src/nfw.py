import numpy as np
import math
import torch
from pathlib import Path
from dataclasses import dataclass
from cosmology import Cosmology

@dataclass
class NFWHalo:
    theta: torch.Tensor
    z_l: float
    cosmology: Cosmology

    def nfw_parameters(self):
        t = torch.as_tensor(self.theta).detach().cpu().numpy().astype(np.float64)
        log_mass = t[..., 0:1]
        concentration = t[..., 1:2]

        mass = 10.0**log_mass
        H_z = self.cosmology.H0 * np.sqrt(
            self.cosmology.Omega_m * (1 + self.z_l)**3
            + self.cosmology.Omega_lambda
        )
        rho_crit = 3 * H_z**2 / (8 * np.pi * self.cosmology.G)
        r_200c = (3 * mass / (4 * np.pi * 200 * rho_crit))**(1 / 3)
        r_s = r_200c / concentration
        rho_s = mass / (
            4 * np.pi * r_s**3
            * (np.log(1 + concentration) - concentration / (1 + concentration))
        )
        return r_s, rho_s

    @staticmethod
    def B_function(x):
        x = np.asarray(x, dtype=np.float64)
        B = np.ones_like(x)  # Handles x == 1

        below = x < 1
        above = x > 1

        u = x[below]
        B[below] = (
            2 / np.sqrt(1 - u**2)
            * np.arctanh(np.sqrt((1 - u) / (1 + u)))
        )

        u = x[above]
        B[above] = (
            2 / np.sqrt(u**2 - 1)
            * np.arctan(np.sqrt((u - 1) / (1 + u)))
        )
        return B

    def surface_densities (self, R):
        r_s, rho_s = self.nfw_parameters()
        x = np.asarray (R,dtype=np.float64) / r_s
        B_values = self.B_function(x)

        F = np.full_like(x, 1.0 / 3.0)
        regular = x != 1.0

        F[regular] = (
            (1 - B_values[regular])
            / (x[regular]**2 - 1)
        )

        Sigma = 2 * rho_s * r_s * F

        # This expression already works at x = 1 when B(1) = 1
        Sigma_mean = (
            4 * rho_s * r_s / x**2
            * (np.log(x / 2) + B_values)
        )
        return Sigma, Sigma_mean

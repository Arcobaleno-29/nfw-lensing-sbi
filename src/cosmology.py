from dataclasses import dataclass

import numpy as np
from scipy.integrate import quad


@dataclass(frozen=True)
class Cosmology:
    H0: float = 70.0
    Omega_m: float = 0.3
    Omega_lambda: float = 0.7
    G: float = 4.30091727e-9  # Mpc (km/s)^2 / solar mass
    c_light: float = 299792.458  # km/s

    def comoving_distance(self, z):
        integral, _ = quad(
            lambda zp: 1.0 / np.sqrt(
                self.Omega_m * (1 + zp)**3 + self.Omega_lambda
            ),
            0.0,
            z,
        )
        return (self.c_light / self.H0) * integral  # Mpc

    def critical_surface_density(self, z_l, z_s):
        if z_l <= 0 or z_s <= z_l:
            raise ValueError("Require 0 < z_l < z_s")

        chi_l = self.comoving_distance(z_l)
        chi_s = self.comoving_distance(z_s)

        D_l = chi_l / (1 + z_l)
        D_s = chi_s / (1 + z_s)
        D_ls = (chi_s - chi_l) / (1 + z_s)

        return (
            self.c_light**2 / (4 * np.pi * self.G)
            * D_s / (D_l * D_ls)
        )
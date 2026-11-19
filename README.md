# NFW weak-lensing inference

## Overview
This project estimates the mass (\(M_{200c}\)) and concentration (\(c_{200c}\)) of a Navarro–Frenk–White (NFW) halo from simulated weak-lensing shear. It compares a posterior evaluated on a parameter grid with one learned from simulations using neural posterior estimation (sbi). The radial analysis uses the same reference observation for both methods, allowing their mass–concentration contours to be compared directly.

## Repository layout

- `src/centred_shear_map.ipynb` — shear-map simulation, radial and sector
  compression, grid posterior, and Fisher-matrix comparison.
- `src/sbi_radial.ipynb` — neural posterior estimation, GetDist plots,
  posterior predictive checks, and calibration checks.
- `src/cosmology.py`, `src/nfw.py`, `src/lensing.py`, `src/compression.py`
  — simulation and analysis functions.
- `src/radial_reference.npz` — reference observation and posterior data
  used by the SBI notebook.
- `src/paired_mocks.npz` — saved simulated observations.

## Getting started
Install NumPy, SciPy, Matplotlib, PyTorch, `sbi`, GetDist, and Jupyter
in a Python environment. Launch Jupyter from `src/`, since the notebooks
load data and import modules relative to that folder:

```bash
cd src
jupyter lab
```
## Author
Tran Vu Minh Anh

## Documentation
* `sbi`: https://sbi.readthedocs.io/en/latest/index.html
* `GetDist`: https://getdist.readthedocs.io/en/latest/intro.html#getting-started

## License
The source code and notebooks are available under the [MIT License](LICENSE).

"""Compress existing shear maps; no halo model or random draws are needed here."""
from dataclasses import dataclass

import numpy as np


def _numpy(value):
    """Accept NumPy arrays or PyTorch tensors, returning a NumPy array."""
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


@dataclass
class CompressionResult:
    """One entry per ring, or per (ring, sector); empty bins contain NaN."""
    radius: np.ndarray
    shear: np.ndarray
    uncertainty: np.ndarray
    counts: np.ndarray
    valid: np.ndarray


def project_shear(g_map, x, y, centre):
    """Return radius, angle in [0, 2pi), tangential and cross reduced shear.

    g_map has shape (2, height, width). The centre is the analysis centre,
    which need not equal the physical halo centre. Cross sign matches the notebook.
    """
    g_map = _numpy(g_map)
    x, y = _numpy(x), _numpy(y)
    if x.shape != y.shape or g_map.shape != (2,) + x.shape:
        raise ValueError("Expected matching x/y grids and g_map shape (2, *x.shape)")
    dx, dy = x - centre[0], y - centre[1]
    radius = np.hypot(dx, dy)
    angle = np.mod(np.arctan2(dy, dx), 2 * np.pi)
    c, s = np.cos(2 * angle), np.sin(2 * angle)
    gt = -g_map[0] * c - g_map[1] * s
    gx = -g_map[0] * s + g_map[1] * c
    return radius, angle, gt, gx


def _edges(edges):
    edges = np.asarray(edges, dtype=float)
    if (edges.ndim != 1 or len(edges) < 2 or not np.isfinite(edges).all()
            or not np.all(np.diff(edges) > 0)):
        raise ValueError("Bin edges must be finite, one-dimensional and strictly increasing")
    return edges


def _compress(gt, radius, sigma, valid_pixels, bin_edges, angle=None, angle_edges=None):
    gt, radius = _numpy(gt), _numpy(radius)
    sigma = np.broadcast_to(_numpy(sigma), gt.shape)
    valid_pixels = _numpy(valid_pixels).astype(bool)
    if radius.shape != gt.shape or valid_pixels.shape != gt.shape:
        raise ValueError("Shear, radius and valid-pixel mask must have matching shapes")
    usable = (valid_pixels & np.isfinite(radius) & np.isfinite(gt)
              & np.isfinite(sigma) & (sigma > 0))
    bin_edges = _edges(bin_edges)
    sector_mode = angle is not None
    if sector_mode:
        angle = np.mod(_numpy(angle), 2 * np.pi)
        angle_edges = _edges(angle_edges)
        if angle.shape != gt.shape:
            raise ValueError("Angle and shear must have matching shapes")
        if not (np.isclose(angle_edges[0], 0) and np.isclose(angle_edges[-1], 2*np.pi)):
            raise ValueError("Angular edges must cover 0 to 2*pi")
        usable &= np.isfinite(angle)
    shape = (len(bin_edges) - 1,)
    if sector_mode:
        shape += (len(angle_edges) - 1,)
    result = CompressionResult(
        *(np.full(shape, np.nan) for _ in range(3)),
        counts=np.zeros(shape, dtype=int), valid=np.zeros(shape, dtype=bool),
    )
    for b in range(shape[0]):
        upper = radius <= bin_edges[b+1] if b == shape[0]-1 else radius < bin_edges[b+1]
        ring = usable & (radius >= bin_edges[b]) & upper
        for s in range(shape[1] if sector_mode else 1):
            index = (b, s) if sector_mode else b
            selection = ring
            if sector_mode:
                selection = ring & (angle >= angle_edges[s]) & (angle < angle_edges[s+1])
            result.counts[index] = selection.sum()
            if not result.counts[index]:
                continue
            weights = 1.0 / sigma[selection].astype(float)**2
            total = weights.sum()
            result.radius[index] = np.sum(weights * radius[selection]) / total
            result.shear[index] = np.sum(weights * gt[selection]) / total
            result.uncertainty[index] = 1.0 / np.sqrt(total)
            result.valid[index] = True
    return result


def radial_compression(gt, radius, sigma, valid_pixels, bin_edges):
    """Inverse-variance ring averages, assuming independent pixel noise.

    Returns CompressionResult with arrays of shape (number_of_rings,).
    sigma is the uncertainty of gt per pixel, NOT the scatter within a ring.
    Pass the same sigma and mask for noisy and noiseless comparisons.
    """
    return _compress(gt, radius, sigma, valid_pixels, bin_edges)


def sector_compression(gt, radius, angle, sigma, valid_pixels, bin_edges, angle_edges):
    """As radial_compression, with arrays of shape (rings, sectors).

    Angular edges are in radians and must span 0 to 2*pi.
    """
    return _compress(gt, radius, sigma, valid_pixels, bin_edges, angle, angle_edges)

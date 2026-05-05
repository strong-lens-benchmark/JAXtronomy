"""
Rectangular pixelised source reconstruction for JAXtronomy.

Solves the regularised linear inversion:

    (F^T C_D^{-1} F + λ H) s = F^T C_D^{-1} d

where:
  F   — bilinear response matrix (N_image × N_source), PSF-convolved
  s   — source pixel amplitudes (unknowns)
  d   — observed image data (flattened, 1D)
  C_D — diagonal noise covariance (per-pixel noise variance)
  H   — regularisation operator (identity, gradient, or curvature)
  λ   — regularisation strength

The response matrix F maps source pixel amplitudes to image-plane flux via
bilinear interpolation of the ray-traced source coordinates.  Each image
pixel contributes to at most 4 source pixels.  PSF blurring is applied by
convolving each column of F with the PSF kernel via batched FFT.

Usage
-----
    solver = PixelatedSourceSolver(
        lens_model, x_image, y_image,
        data, noise_map, psf_kernel,
        nx_src=30, ny_src=30,
        src_x_min=-1.5, src_x_max=1.5,
        src_y_min=-1.5, src_y_max=1.5,
        regularisation='gradient',
        lambda_reg=0.1,
    )
    source, image_model = solver.solve(kwargs_lens)
"""

from __future__ import annotations

from functools import partial

import numpy as np
import jax
import jax.numpy as jnp
from jax import jit

__all__ = ['PixelatedSourceSolver']


# ---------------------------------------------------------------------------
# Core JAX functions (called inside jit)
# ---------------------------------------------------------------------------

def _bilinear_response(beta_x, beta_y, nx_src, ny_src,
                        x_min, x_max, y_min, y_max):
    """Build bilinear response matrix F of shape (N_image, N_source).

    F[i, k] is the bilinear weight of source pixel k at the ray-traced
    source-plane position of image pixel i.  Image pixels that fall outside
    the source grid contribute zero (validity mask applied row-wise).

    Grid convention (matching TinyLensGPU): x_min and x_max are the centres
    of the first and last source pixels respectively.  Pixel spacing is
    dx = (x_max - x_min) / (nx_src - 1).  Fractional pixel index runs from
    0 (at x_min) to nx_src-1 (at x_max).
    """
    dx = (x_max - x_min) / max(nx_src - 1, 1)
    dy = (y_max - y_min) / max(ny_src - 1, 1)

    # Fractional pixel index: 0 at x_min, nx_src-1 at x_max
    ux = (beta_x - x_min) / (dx + 1e-30)
    uy = (beta_y - y_min) / (dy + 1e-30)

    # Valid rays: fall within the source grid
    valid = ((ux >= 0.0) & (ux <= float(nx_src - 1)) &
             (uy >= 0.0) & (uy <= float(ny_src - 1)))

    ix0 = jnp.floor(ux).astype(jnp.int32)
    iy0 = jnp.floor(uy).astype(jnp.int32)
    fx = ux - jnp.floor(ux)
    fy = uy - jnp.floor(uy)

    # Clamp corner indices to valid range for safe array access
    ix0_c = jnp.clip(ix0,     0, nx_src - 1)
    iy0_c = jnp.clip(iy0,     0, ny_src - 1)
    ix1_c = jnp.clip(ix0 + 1, 0, nx_src - 1)
    iy1_c = jnp.clip(iy0 + 1, 0, ny_src - 1)

    # Bilinear weights for the 4 corners
    w00 = (1.0 - fx) * (1.0 - fy)
    w10 = fx          * (1.0 - fy)
    w01 = (1.0 - fx) * fy
    w11 = fx          * fy

    N_image = beta_x.shape[0]
    N_source = nx_src * ny_src
    img_idx = jnp.arange(N_image)
    F = jnp.zeros((N_image, N_source))

    # Scatter weights — Python loop is static (4 iterations at trace time)
    for src_idx, w in [
        (iy0_c * nx_src + ix0_c, w00),
        (iy0_c * nx_src + ix1_c, w10),
        (iy1_c * nx_src + ix0_c, w01),
        (iy1_c * nx_src + ix1_c, w11),
    ]:
        F = F.at[img_idx, src_idx].add(jnp.where(valid, w, 0.0))

    return F


def _psf_convolve_response(F, psf_fft, numpix, N_source):
    """PSF-convolve every column of F via batched 2D FFT.

    F       : (N_image, N_source), N_image = numpix^2
    psf_fft : precomputed rfft2 of the PSF padded to (numpix, numpix)

    Returns F_conv : (N_image, N_source).
    """
    F_images = F.T.reshape(N_source, numpix, numpix)
    F_fft = jnp.fft.rfft2(F_images)
    F_conv = jnp.fft.irfft2(F_fft * psf_fft[jnp.newaxis], s=(numpix, numpix))
    return F_conv.reshape(N_source, numpix * numpix).T


def _regularised_solve(F_conv, data, noise_map, H, lambda_reg):
    """Solve (F^T W F + λ H) s = F^T W d for source pixel amplitudes s.

    F_conv    : (N_image, N_source)
    data      : (N_image,)
    noise_map : (N_image,)  per-pixel noise standard deviation
    H         : (N_source, N_source)
    lambda_reg: scalar
    """
    W = 1.0 / noise_map ** 2
    WF = F_conv * W[:, jnp.newaxis]
    M = WF.T @ F_conv + lambda_reg * H
    b = F_conv.T @ (W * data)
    return jnp.linalg.solve(M, b)


# ---------------------------------------------------------------------------
# Static helpers (NumPy, computed once at solver initialisation)
# ---------------------------------------------------------------------------

def _build_regularisation_matrix(nx_src, ny_src, scheme):
    """Build regularisation matrix H (N_source × N_source) in NumPy.

    scheme : 'identity' | 'gradient' | 'curvature'

    'identity'  : H = I  (ridge / L2 on amplitudes)
    'gradient'  : H = L^T L where L is the first-difference operator
                  (penalises amplitude differences between adjacent pixels)
    'curvature' : H encodes the discrete Laplacian squared
                  (penalises second-order variations)
    """
    N = nx_src * ny_src

    if scheme == 'identity':
        return np.eye(N)

    if scheme == 'gradient':
        H = np.zeros((N, N))
        for iy in range(ny_src):
            for ix in range(nx_src):
                k = iy * nx_src + ix
                for dix, diy in [(1, 0), (0, 1)]:
                    jx, jy = ix + dix, iy + diy
                    if 0 <= jx < nx_src and 0 <= jy < ny_src:
                        m = jy * nx_src + jx
                        H[k, k] += 1.0
                        H[m, m] += 1.0
                        H[k, m] -= 1.0
                        H[m, k] -= 1.0
        return H

    if scheme == 'curvature':
        # Build discrete Laplacian L, then H = L^T L (symmetric PSD by construction)
        L = np.zeros((N, N))
        for iy in range(ny_src):
            for ix in range(nx_src):
                k = iy * nx_src + ix
                nbrs = [
                    (iy + diy) * nx_src + (ix + dix)
                    for dix, diy in [(-1, 0), (1, 0), (0, -1), (0, 1)]
                    if 0 <= ix + dix < nx_src and 0 <= iy + diy < ny_src
                ]
                L[k, k] = float(len(nbrs))
                for m in nbrs:
                    L[k, m] = -1.0
        return L.T @ L

    raise ValueError(f"Unknown regularisation scheme: {scheme!r}. "
                     "Choose 'identity', 'gradient', or 'curvature'.")


def _prepare_psf_fft(psf_kernel, numpix):
    """Normalise, pad, and shift PSF; return precomputed rfft2 as a JAX array."""
    psf_norm = psf_kernel / psf_kernel.sum()
    kh, kw = psf_norm.shape
    padded = np.zeros((numpix, numpix))
    padded[:kh, :kw] = psf_norm
    # Roll to origin so that convolution is centred correctly
    padded = np.roll(padded, -(kh // 2), axis=0)
    padded = np.roll(padded, -(kw // 2), axis=1)
    return jnp.asarray(np.fft.rfft2(padded), dtype=jnp.complex128)


# ---------------------------------------------------------------------------
# Solver class
# ---------------------------------------------------------------------------

class PixelatedSourceSolver:
    """JAX-native rectangular pixelised source reconstruction.

    The source plane is covered by an nx_src × ny_src rectangular grid of
    equal-area pixels.  For each evaluation of `solve`, the lens model
    ray-shoots the image grid to the source plane, builds the bilinear
    response matrix, PSF-convolves it, and solves the regularised normal
    equations in one `jnp.linalg.solve` call.

    All of the above is JIT-compiled and fully differentiable w.r.t. the
    lens model parameters.

    Parameters
    ----------
    lens_model :
        jaxtronomy ``LensModel`` instance (must support `ray_shooting`).
    x_image, y_image : array_like, shape (N_image,)
        Flattened image-plane coordinate grids (arcsec).
    data : array_like, shape (N_image,)
        Observed image pixel values (flattened).
    noise_map : array_like or scalar
        Per-pixel noise standard deviation (arcsec or surface-brightness
        units consistent with `data`).  A scalar is broadcast to all pixels.
    psf_kernel : array_like, shape (kh, kw)
        PSF kernel (need not be normalised).
    nx_src, ny_src : int
        Number of source pixels along x and y.
    src_x_min, src_x_max : float
        Source plane x extent (arcsec).
    src_y_min, src_y_max : float
        Source plane y extent (arcsec).
    regularisation : str
        Regularisation scheme: ``'identity'``, ``'gradient'``, or
        ``'curvature'``.  Default ``'gradient'``.
    lambda_reg : float
        Regularisation strength λ.  Default 1.0.
    """

    def __init__(self, lens_model, x_image, y_image, data, noise_map, psf_kernel,
                 nx_src, ny_src, src_x_min, src_x_max, src_y_min, src_y_max,
                 regularisation='gradient', lambda_reg=1.0):
        self._lens_model = lens_model
        self._x_image = jnp.asarray(x_image, dtype=jnp.float64)
        self._y_image = jnp.asarray(y_image, dtype=jnp.float64)
        self._data = jnp.asarray(data, dtype=jnp.float64)

        noise_arr = (np.full(len(x_image), float(noise_map))
                     if np.isscalar(noise_map) else np.asarray(noise_map))
        self._noise_map = jnp.asarray(noise_arr, dtype=jnp.float64)

        self._nx_src = int(nx_src)
        self._ny_src = int(ny_src)
        self._src_x_min = float(src_x_min)
        self._src_x_max = float(src_x_max)
        self._src_y_min = float(src_y_min)
        self._src_y_max = float(src_y_max)
        self._lambda_reg = float(lambda_reg)
        self._numpix = int(round(np.sqrt(len(x_image))))

        # Static quantities precomputed in NumPy / at init
        H_np = _build_regularisation_matrix(nx_src, ny_src, regularisation)
        self._H = jnp.asarray(H_np, dtype=jnp.float64)
        self._psf_fft = _prepare_psf_fft(np.asarray(psf_kernel), self._numpix)

    @partial(jit, static_argnums=(0,))
    def solve(self, kwargs_lens):
        """Solve for source pixel amplitudes and return the reconstructed image.

        Parameters
        ----------
        kwargs_lens : list of dicts
            Lens model keyword arguments (e.g. ``[{'theta_E': 1.0, ...}]``).

        Returns
        -------
        source : jnp.ndarray, shape (ny_src, nx_src)
            Reconstructed source surface brightness.
        image_model : jnp.ndarray, shape (numpix, numpix)
            PSF-convolved model image corresponding to the reconstructed source.
        """
        beta_x, beta_y = self._lens_model.ray_shooting(
            self._x_image, self._y_image, kwargs_lens
        )

        F = _bilinear_response(
            beta_x, beta_y,
            self._nx_src, self._ny_src,
            self._src_x_min, self._src_x_max,
            self._src_y_min, self._src_y_max,
        )

        F_conv = _psf_convolve_response(
            F, self._psf_fft, self._numpix,
            self._nx_src * self._ny_src,
        )

        s = _regularised_solve(
            F_conv, self._data, self._noise_map, self._H, self._lambda_reg
        )

        source = s.reshape(self._ny_src, self._nx_src)
        image_model = (F_conv @ s).reshape(self._numpix, self._numpix)
        return source, image_model

    def source_coordinates(self):
        """Return pixel-centre coordinate arrays for the source grid.

        x_min and x_max are the centres of the boundary pixels (same
        convention as the bilinear interpolation).

        Returns
        -------
        x : np.ndarray, shape (nx_src,)
        y : np.ndarray, shape (ny_src,)
        """
        x = np.linspace(self._src_x_min, self._src_x_max, self._nx_src)
        y = np.linspace(self._src_y_min, self._src_y_max, self._ny_src)
        return x, y

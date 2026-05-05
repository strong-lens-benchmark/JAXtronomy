"""
Tests for PixelatedSourceSolver and its component functions.

Strategy
--------
1. Unit tests for _bilinear_response and _build_regularisation_matrix.
2. Integration test: build a synthetic lensed image with a known Gaussian
   source, run the solver, and verify that the chi-squared of the
   reconstruction is consistent with the noise level.
"""

import numpy as np
import jax.numpy as jnp
import pytest

from jaxtronomy.ImSim.SourceReconstruction.pixelated_source import (
    PixelatedSourceSolver,
    _bilinear_response,
    _build_regularisation_matrix,
    _prepare_psf_fft,
    _psf_convolve_response,
)
from jaxtronomy.LensModel.lens_model import LensModel


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

NUMPIX = 40          # image side length in pixels
PIXEL_SCALE = 0.05   # arcsec / pixel
NX_SRC = 20
NY_SRC = 20
SRC_EXTENT = 0.8     # source grid half-width (arcsec)
NOISE = 0.01         # uniform noise rms
LAMBDA_REG = 1e-2
PSF_SIGMA = 1.5      # PSF Gaussian sigma in pixels


def _make_image_grid(numpix=NUMPIX, scale=PIXEL_SCALE):
    """Return flattened image-plane coordinate grids centred at origin."""
    coords = (np.arange(numpix) - (numpix - 1) / 2.0) * scale
    xx, yy = np.meshgrid(coords, coords)
    return xx.ravel(), yy.ravel()


def _gaussian_psf(numpix, sigma=PSF_SIGMA):
    """Small Gaussian PSF kernel."""
    ksize = 11
    c = ksize // 2
    x = np.arange(ksize) - c
    xx, yy = np.meshgrid(x, x)
    k = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    return k / k.sum()


def _gaussian_source(x, y, amp=1.0, x0=0.05, y0=0.0, sigma=0.1):
    """Evaluate a 2D Gaussian source at (x, y)."""
    return amp * np.exp(-((x - x0)**2 + (y - y0)**2) / (2 * sigma**2))


def _make_mock_image(kwargs_lens, numpix=NUMPIX, scale=PIXEL_SCALE,
                     noise_rms=NOISE, seed=42):
    """
    Generate a synthetic lensed image.

    Ray-shoots image pixels to source plane, evaluates a Gaussian source,
    convolves with PSF, adds Gaussian noise.
    """
    from jax.scipy.signal import fftconvolve

    lens = LensModel(['EPL'])
    x_img, y_img = _make_image_grid(numpix, scale)

    beta_x, beta_y = np.array(lens.ray_shooting(
        jnp.array(x_img), jnp.array(y_img), kwargs_lens
    ))

    source_1d = _gaussian_source(beta_x, beta_y)
    source_2d = source_1d.reshape(numpix, numpix)

    psf = _gaussian_psf(numpix)
    image_clean = np.array(fftconvolve(
        jnp.array(source_2d), jnp.array(psf), mode='same'
    ))

    rng = np.random.default_rng(seed)
    noise = rng.normal(0, noise_rms, size=image_clean.shape)
    return (image_clean + noise).ravel(), x_img, y_img


# ---------------------------------------------------------------------------
# Unit tests — _bilinear_response
# ---------------------------------------------------------------------------

class TestBilinearResponse:

    def test_row_sums_at_most_one(self):
        """Image pixels that fall inside the source grid should have row sum = 1."""
        x = jnp.array([0.0, 0.1, -0.2, 0.3])
        y = jnp.array([0.0, 0.0,  0.1, -0.1])
        F = _bilinear_response(x, y, 10, 10, -1.0, 1.0, -1.0, 1.0)
        row_sums = jnp.sum(F, axis=1)
        # All test points are inside the grid → row sum should be 1
        np.testing.assert_allclose(np.array(row_sums), 1.0, atol=1e-12)

    def test_out_of_bounds_row_sum_zero(self):
        """Pixels that map outside the source grid should give zero row sum."""
        x = jnp.array([5.0, -5.0])   # well outside [-1, 1]
        y = jnp.array([5.0, -5.0])
        F = _bilinear_response(x, y, 10, 10, -1.0, 1.0, -1.0, 1.0)
        row_sums = jnp.sum(F, axis=1)
        np.testing.assert_allclose(np.array(row_sums), 0.0, atol=1e-12)

    def test_pixel_centre_gives_unit_weight(self):
        """A ray that lands exactly on a pixel centre should give weight 1 to that pixel."""
        nx, ny = 10, 10
        x_min, x_max = -1.0, 1.0
        y_min, y_max = -1.0, 1.0
        dx = (x_max - x_min) / (nx - 1)
        dy = (y_max - y_min) / (ny - 1)
        # Pixel (3, 2) centre: x_min + 3*dx, y_min + 2*dy
        xc = x_min + 3 * dx
        yc = y_min + 2 * dy
        F = _bilinear_response(jnp.array([xc]), jnp.array([yc]),
                                nx, ny, x_min, x_max, y_min, y_max)
        src_idx = 2 * nx + 3   # iy=2, ix=3
        assert float(F[0, src_idx]) == pytest.approx(1.0, abs=1e-10)
        assert float(jnp.sum(F[0])) == pytest.approx(1.0, abs=1e-10)

    def test_non_negative(self):
        """All entries of F must be non-negative (they are interpolation weights)."""
        rng = np.random.default_rng(0)
        x = jnp.array(rng.uniform(-0.8, 0.8, 50))
        y = jnp.array(rng.uniform(-0.8, 0.8, 50))
        F = _bilinear_response(x, y, 12, 12, -1.0, 1.0, -1.0, 1.0)
        assert float(jnp.min(F)) >= 0.0

    def test_shape(self):
        n_img, nx, ny = 25, 8, 10
        x = jnp.zeros(n_img)
        y = jnp.zeros(n_img)
        F = _bilinear_response(x, y, nx, ny, -1.0, 1.0, -1.0, 1.0)
        assert F.shape == (n_img, nx * ny)


# ---------------------------------------------------------------------------
# Unit tests — _build_regularisation_matrix
# ---------------------------------------------------------------------------

class TestRegularisationMatrix:

    def test_identity(self):
        H = _build_regularisation_matrix(5, 5, 'identity')
        np.testing.assert_array_equal(H, np.eye(25))

    def test_gradient_symmetric(self):
        H = _build_regularisation_matrix(4, 4, 'gradient')
        np.testing.assert_allclose(H, H.T, atol=1e-12)

    def test_gradient_psd(self):
        H = _build_regularisation_matrix(4, 4, 'gradient')
        eigs = np.linalg.eigvalsh(H)
        assert eigs.min() >= -1e-10

    def test_curvature_symmetric(self):
        H = _build_regularisation_matrix(4, 4, 'curvature')
        np.testing.assert_allclose(H, H.T, atol=1e-12)

    def test_shape(self):
        for scheme in ('identity', 'gradient', 'curvature'):
            H = _build_regularisation_matrix(6, 5, scheme)
            assert H.shape == (30, 30)

    def test_unknown_scheme_raises(self):
        with pytest.raises(ValueError, match='Unknown'):
            _build_regularisation_matrix(4, 4, 'banana')


# ---------------------------------------------------------------------------
# Integration test — full reconstruction
# ---------------------------------------------------------------------------

class TestPixelatedSourceSolver:

    @pytest.fixture(autouse=True)
    def setup(self):
        self.kwargs_lens = [{'theta_E': 1.0, 'gamma': 2.0,
                              'e1': 0.05, 'e2': 0.0,
                              'center_x': 0.0, 'center_y': 0.0}]
        self.data, self.x_img, self.y_img = _make_mock_image(self.kwargs_lens)
        self.lens_model = LensModel(['EPL'])
        self.psf = _gaussian_psf(NUMPIX)

    def _make_solver(self, regularisation='gradient', lambda_reg=LAMBDA_REG):
        return PixelatedSourceSolver(
            lens_model=self.lens_model,
            x_image=self.x_img,
            y_image=self.y_img,
            data=self.data,
            noise_map=NOISE,
            psf_kernel=self.psf,
            nx_src=NX_SRC,
            ny_src=NY_SRC,
            src_x_min=-SRC_EXTENT,
            src_x_max=SRC_EXTENT,
            src_y_min=-SRC_EXTENT,
            src_y_max=SRC_EXTENT,
            regularisation=regularisation,
            lambda_reg=lambda_reg,
        )

    def test_solve_returns_correct_shapes(self):
        solver = self._make_solver()
        source, image_model = solver.solve(self.kwargs_lens)
        assert source.shape == (NY_SRC, NX_SRC)
        assert image_model.shape == (NUMPIX, NUMPIX)

    def test_reconstruction_chi2_self_consistent(self):
        """Chi2 ≈ 1 when mock data is generated from the solver's own forward model.

        This tests the correctness of the linear solve without any PSF convention
        mismatch or pixelation error — the solver is given exactly the data it
        can represent, and with small lambda should recover it to noise level.
        """
        solver = self._make_solver(lambda_reg=1e-5)

        # True source: Gaussian evaluated at source pixel centres
        sx, sy = solver.source_coordinates()
        sxx, syy = np.meshgrid(sx, sy)
        s_true = _gaussian_source(sxx.ravel(), syy.ravel(), sigma=0.15).astype(np.float64)

        # Compute F and F_conv using the solver's own internals
        beta_x, beta_y = np.array(self.lens_model.ray_shooting(
            jnp.array(self.x_img), jnp.array(self.y_img), self.kwargs_lens
        ))
        F = _bilinear_response(
            jnp.array(beta_x), jnp.array(beta_y),
            NX_SRC, NY_SRC, -SRC_EXTENT, SRC_EXTENT, -SRC_EXTENT, SRC_EXTENT,
        )
        F_conv = _psf_convolve_response(F, solver._psf_fft, NUMPIX, NX_SRC * NY_SRC)

        # Generate data from solver's forward model and add noise
        data_clean = np.array(F_conv @ jnp.array(s_true))
        rng = np.random.default_rng(7)
        data_noisy = data_clean + rng.normal(0, NOISE, size=data_clean.shape)

        # Solve with the self-consistent data
        solver2 = PixelatedSourceSolver(
            self.lens_model, self.x_img, self.y_img, data_noisy, NOISE, self.psf,
            NX_SRC, NY_SRC, -SRC_EXTENT, SRC_EXTENT, -SRC_EXTENT, SRC_EXTENT,
            regularisation='gradient', lambda_reg=1e-5,
        )
        _, image_model = solver2.solve(self.kwargs_lens)

        residuals = data_noisy.reshape(NUMPIX, NUMPIX) - np.array(image_model)
        chi2_reduced = np.sum(residuals ** 2 / NOISE ** 2) / (NUMPIX ** 2)
        assert chi2_reduced < 3.0

    def test_source_non_negative_in_centre(self):
        """The reconstructed source should be positive near the true source."""
        solver = self._make_solver()
        source, _ = solver.solve(self.kwargs_lens)
        # The true source is a Gaussian centred near (0.05, 0.0) arcsec
        sx, sy = solver.source_coordinates()
        ix = np.argmin(np.abs(sx - 0.05))
        iy = np.argmin(np.abs(sy - 0.0))
        assert float(source[iy, ix]) > 0.0

    def test_jit_reuse(self):
        """Second call should be faster (uses JIT cache) — just check it runs."""
        solver = self._make_solver()
        solver.solve(self.kwargs_lens)  # warm-up / compile
        source2, _ = solver.solve(self.kwargs_lens)
        assert source2.shape == (NY_SRC, NX_SRC)

    def test_gradient_regularisation(self):
        solver = self._make_solver(regularisation='gradient')
        source, _ = solver.solve(self.kwargs_lens)
        assert jnp.all(jnp.isfinite(source))

    def test_curvature_regularisation(self):
        solver = self._make_solver(regularisation='curvature')
        source, _ = solver.solve(self.kwargs_lens)
        assert jnp.all(jnp.isfinite(source))

    def test_identity_regularisation(self):
        solver = self._make_solver(regularisation='identity')
        source, _ = solver.solve(self.kwargs_lens)
        assert jnp.all(jnp.isfinite(source))

    def test_source_coordinates_shape(self):
        solver = self._make_solver()
        x, y = solver.source_coordinates()
        assert x.shape == (NX_SRC,)
        assert y.shape == (NY_SRC,)

    def test_source_coordinates_range(self):
        solver = self._make_solver()
        x, y = solver.source_coordinates()
        # x_min and x_max are the centres of the boundary pixels
        assert float(x[0]) == pytest.approx(-SRC_EXTENT, rel=1e-6)
        assert float(x[-1]) == pytest.approx(SRC_EXTENT, rel=1e-6)

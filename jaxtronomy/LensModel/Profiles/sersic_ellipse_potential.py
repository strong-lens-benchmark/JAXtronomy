__author__ = 'sibirrer', 'aymgal'

from jax import config, jit, grad
import jax.numpy as jnp
import jax.scipy.special as special
import numpy as np

from jaxtronomy.Util import param_util
from lenstronomy.LensModel.Profiles.base_profile import LensProfileBase

config.update("jax_enable_x64", True)

__all__ = ["SersicEllipsePotential"]

# 64-point Gauss-Legendre nodes and weights on [0, 1] for the potential integral.
_gl_nodes, _gl_weights = np.polynomial.legendre.leggauss(64)
_GL_NODES_01 = jnp.array(0.5 * (_gl_nodes + 1.0))
_GL_WEIGHTS_01 = jnp.array(0.5 * _gl_weights)


class SersicEllipsePotential(LensProfileBase):
    """Sérsic mass profile with ellipticity defined in the lensing potential.

    Ellipticity is applied via coordinate stretching following
    Golse & Kneib (2002). The convergence is:

    .. math::
        \\kappa(R) = k_{\\rm eff}\\exp\\!\\left(-b_n\\left[(R/R_{\\rm s})^{1/n}-1\\right]\\right)

    with :math:`b_n \\approx 1.9992\\,n - 0.3271`.
    """

    param_names = ["k_eff", "R_sersic", "n_sersic", "e1", "e2", "center_x", "center_y"]
    lower_limit_default = {
        "k_eff": 0, "R_sersic": 0, "n_sersic": 0.5,
        "e1": -0.5, "e2": -0.5, "center_x": -100, "center_y": -100,
    }
    upper_limit_default = {
        "k_eff": 10, "R_sersic": 100, "n_sersic": 8,
        "e1": 0.5, "e2": 0.5, "center_x": 100, "center_y": 100,
    }

    def __init__(self):
        super(SersicEllipsePotential, self).__init__()

    @staticmethod
    @jit
    def _b_n(n):
        return jnp.maximum(1.9992 * n - 0.3271, 1e-5)

    @staticmethod
    @jit
    def _alpha_magnitude(r, n_sersic, R_sersic, k_eff):
        """Deflection angle magnitude for the spherical Sérsic profile.

        Uses jax.scipy.special.gammainc (regularised lower incomplete gamma).
        a_eff is computed in log-space to avoid intermediate overflow at large n.
        """
        R_sersic = jnp.where(R_sersic < 1e-7, 1e-7, R_sersic)
        r = jnp.where(r < 1e-8, 1e-8, r)
        b = SersicEllipsePotential._b_n(n_sersic)
        x_red = jnp.maximum(r / R_sersic, 1e-10) ** (1.0 / n_sersic)

        log_a_eff = (jnp.log(n_sersic)
                     + jnp.log(R_sersic)
                     + jnp.log(jnp.maximum(k_eff, 1e-30))
                     - 2.0 * n_sersic * jnp.log(b)
                     + b
                     + special.gammaln(2.0 * n_sersic))
        a_eff = jnp.exp(log_a_eff)
        p = special.gammainc(2.0 * n_sersic, b * x_red)
        return 2.0 * a_eff * x_red ** (-n_sersic) * p

    @staticmethod
    @jit
    def function(x, y, n_sersic, R_sersic, k_eff, e1, e2, center_x=0, center_y=0):
        """Lensing potential of the elliptical Sérsic profile.

        Computed as psi(R) = R * integral_0^1 alpha_sph(R*t) dt via
        64-point Gauss-Legendre quadrature, avoiding the large cancellations
        in the equivalent hypergeometric series at large |z|.

        :param x: x-coordinate (arcsec)
        :param y: y-coordinate (arcsec)
        :param n_sersic: Sérsic index
        :param R_sersic: half-light radius (arcsec)
        :param k_eff: convergence at R_sersic
        :param e1: eccentricity component
        :param e2: eccentricity component
        :param center_x: x-centre (arcsec)
        :param center_y: y-centre (arcsec)
        :return: lensing potential (arcsec^2)
        """
        x_, y_ = param_util.transform_e1e2_square_average(x, y, e1, e2, center_x, center_y)
        R_ = jnp.sqrt(x_**2 + y_**2 + 1e-12)

        r_quad = R_[..., jnp.newaxis] * _GL_NODES_01
        alpha_vals = SersicEllipsePotential._alpha_magnitude(
            r_quad, n_sersic, R_sersic, k_eff
        )
        return R_ * jnp.sum(_GL_WEIGHTS_01 * alpha_vals, axis=-1)

    @staticmethod
    @jit
    def derivatives(x, y, n_sersic, R_sersic, k_eff, e1, e2, center_x=0, center_y=0):
        """Deflection angles of the elliptical Sérsic profile.

        :param x: x-coordinate (arcsec)
        :param y: y-coordinate (arcsec)
        :param n_sersic: Sérsic index
        :param R_sersic: half-light radius (arcsec)
        :param k_eff: convergence at R_sersic
        :param e1: eccentricity component
        :param e2: eccentricity component
        :param center_x: x-centre (arcsec)
        :param center_y: y-centre (arcsec)
        :return: deflection angles (alpha_x, alpha_y) in arcsec
        """
        phi_G, q = param_util.ellipticity2phi_q(e1, e2)
        e = param_util.q2e(q)
        cos_phi = jnp.cos(phi_G)
        sin_phi = jnp.sin(phi_G)

        x_, y_ = param_util.transform_e1e2_square_average(x, y, e1, e2, center_x, center_y)
        r = jnp.sqrt(x_**2 + y_**2 + 1e-16)

        alpha_mag = SersicEllipsePotential._alpha_magnitude(r, n_sersic, R_sersic, k_eff)

        f_x_prim = alpha_mag * x_ / r * jnp.sqrt(1.0 - e)
        f_y_prim = alpha_mag * y_ / r * jnp.sqrt(1.0 + e)
        f_x = cos_phi * f_x_prim - sin_phi * f_y_prim
        f_y = sin_phi * f_x_prim + cos_phi * f_y_prim
        return f_x, f_y

    def hessian(self, x, y, n_sersic, R_sersic, k_eff, e1, e2, center_x=0, center_y=0):
        """Hessian of the lensing potential via JAX autodiff of derivatives.

        :param x: x-coordinate (arcsec)
        :param y: y-coordinate (arcsec)
        :param n_sersic: Sérsic index
        :param R_sersic: half-light radius (arcsec)
        :param k_eff: convergence at R_sersic
        :param e1: eccentricity component
        :param e2: eccentricity component
        :param center_x: x-centre (arcsec)
        :param center_y: y-centre (arcsec)
        :return: f_xx, f_xy, f_yx, f_yy
        """
        kwargs = (n_sersic, R_sersic, k_eff, e1, e2, center_x, center_y)

        def _alpha_x(xy):
            return self.derivatives(xy[0], xy[1], *kwargs)[0]

        def _alpha_y(xy):
            return self.derivatives(xy[0], xy[1], *kwargs)[1]

        @jit
        def _hess_single(xi, yi):
            g_x = grad(_alpha_x)(jnp.array([xi, yi]))
            g_y = grad(_alpha_y)(jnp.array([xi, yi]))
            return g_x[0], g_x[1], g_y[0], g_y[1]

        f_xx, f_xy, f_yx, f_yy = jnp.vectorize(_hess_single)(x, y)
        return f_xx, f_xy, f_yx, f_yy

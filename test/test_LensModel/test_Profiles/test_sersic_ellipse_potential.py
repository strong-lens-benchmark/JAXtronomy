import numpy as np
import numpy.testing as npt
import pytest

from jaxtronomy.LensModel.Profiles.sersic_ellipse_potential import SersicEllipsePotential
from lenstronomy.LensModel.Profiles.sersic_ellipse_potential import (
    SersicEllipsePotential as SersicEllipsePotential_ref,
)


class TestSersicEllipsePotential(object):

    def setup_method(self):
        self.prof = SersicEllipsePotential()
        self.prof_ref = SersicEllipsePotential_ref()

    def test_function(self):
        x = np.linspace(0.1, 2.0, 8)
        y = np.zeros_like(x)
        kwargs = dict(n_sersic=4.0, R_sersic=1.0, k_eff=0.5, e1=0.0, e2=0.0)

        f = self.prof.function(x, y, **kwargs)
        f_ref = self.prof_ref.function(x, y, **kwargs)
        npt.assert_array_almost_equal(f, f_ref, decimal=5)

        # elliptical
        kwargs_ell = dict(n_sersic=4.0, R_sersic=1.0, k_eff=0.5, e1=0.1, e2=0.05)
        x2 = np.array([0.5, 1.0, -0.5, 0.2])
        y2 = np.array([0.3, -0.3, 0.8, -0.7])
        f_ell = self.prof.function(x2, y2, **kwargs_ell)
        f_ell_ref = self.prof_ref.function(x2, y2, **kwargs_ell)
        npt.assert_array_almost_equal(f_ell, f_ell_ref, decimal=5)

    def test_derivatives(self):
        x = np.linspace(0.1, 2.0, 8)
        y = np.zeros_like(x)
        kwargs = dict(n_sersic=4.0, R_sersic=1.0, k_eff=0.5, e1=0.0, e2=0.0)

        f_x, f_y = self.prof.derivatives(x, y, **kwargs)
        f_x_ref, f_y_ref = self.prof_ref.derivatives(x, y, **kwargs)
        npt.assert_array_almost_equal(f_x, f_x_ref, decimal=8)
        npt.assert_array_almost_equal(f_y, f_y_ref, decimal=8)

        # elliptical
        kwargs_ell = dict(n_sersic=4.0, R_sersic=1.0, k_eff=0.5, e1=0.1, e2=0.05)
        x2 = np.array([0.5, 1.0, -0.5, 0.2])
        y2 = np.array([0.3, -0.3, 0.8, -0.7])
        f_x, f_y = self.prof.derivatives(x2, y2, **kwargs_ell)
        f_x_ref, f_y_ref = self.prof_ref.derivatives(x2, y2, **kwargs_ell)
        npt.assert_array_almost_equal(f_x, f_x_ref, decimal=8)
        npt.assert_array_almost_equal(f_y, f_y_ref, decimal=8)

    def test_derivatives_high_n(self):
        # n_sersic=8 is where hyp2f2 fails; the GL quadrature / log-space path must hold
        x = np.array([0.5, 1.0, 2.0])
        y = np.array([0.1, 0.5, 0.3])
        kwargs = dict(n_sersic=8.0, R_sersic=1.0, k_eff=0.5, e1=0.1, e2=0.05)
        f_x, f_y = self.prof.derivatives(x, y, **kwargs)
        f_x_ref, f_y_ref = self.prof_ref.derivatives(x, y, **kwargs)
        npt.assert_array_almost_equal(f_x, f_x_ref, decimal=5)
        npt.assert_array_almost_equal(f_y, f_y_ref, decimal=5)

    def test_hessian(self):
        x = np.array([0.5, 1.0, -0.5])
        y = np.array([0.3, -0.3, 0.8])
        kwargs = dict(n_sersic=4.0, R_sersic=1.0, k_eff=0.5, e1=0.1, e2=0.05)

        f_xx, f_xy, f_yx, f_yy = self.prof.hessian(x, y, **kwargs)
        f_xx_ref, f_xy_ref, f_yx_ref, f_yy_ref = self.prof_ref.hessian(x, y, **kwargs)
        npt.assert_array_almost_equal(f_xx, f_xx_ref, decimal=4)
        npt.assert_array_almost_equal(f_xy, f_xy_ref, decimal=4)
        npt.assert_array_almost_equal(f_yx, f_yx_ref, decimal=4)
        npt.assert_array_almost_equal(f_yy, f_yy_ref, decimal=4)

    def test_finite_outputs(self):
        # confirm no NaN/Inf over a range of parameters
        x = np.array([0.01, 0.1, 0.5, 1.0, 2.0, 5.0])
        y = np.zeros_like(x)
        for n in [0.5, 1.0, 2.0, 4.0, 8.0]:
            kwargs = dict(n_sersic=n, R_sersic=1.0, k_eff=0.5, e1=0.1, e2=0.05)
            f = self.prof.function(x, y, **kwargs)
            fx, fy = self.prof.derivatives(x, y, **kwargs)
            assert np.all(np.isfinite(f)), f"function not finite for n={n}"
            assert np.all(np.isfinite(fx)), f"fx not finite for n={n}"
            assert np.all(np.isfinite(fy)), f"fy not finite for n={n}"


if __name__ == "__main__":
    pytest.main()

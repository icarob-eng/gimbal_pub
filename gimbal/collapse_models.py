import numpy as np
import scipy as sc

import abc

from . import cosmology_models as csm


def delta_an_eds(vec_a: np.arange, d_i):
    r""" Analytic solution for :math:`\delta` in Einstein-de-Sitter """
    return d_i * (vec_a / vec_a[0])


class CollapseModel(abc.ABC):
    r"""
    Abstract Spherical Collapse Model.

    :param cosmology: instance of cosmology_models.CosmologyModel. Defines background cosmology
    :param z_i: initial redshift
    :param delta_m_i: initial delta
    :param d_delta_m_i: derivative of initial delta. If None, it is calculated from EdS

    :var a_i: initial scale factor
    :var vec_a: geometric space vector for scale factor
    :var vec_delta_m: vector with all values of :math:`\delta_m`
    :var vec_d_delta_m: vector with all values of :math:`\delta_m^\prime`
    """
    def __init__(self, cosmology: csm.CosmologyModel, z_i=99, delta_m_i=1e-2, d_delta_m_i=None, samples=1e5, **kwargs):
        r"""
        :param samples: number of samples to integrate
        """
        self.cosmology = cosmology

        self.z_i = z_i
        self.a_i = 1 / (1 + z_i)
        self.delta_m_i = delta_m_i
        self.d_delta_m_i = self.delta_m_i / self.a_i if d_delta_m_i is None else d_delta_m_i

        self.samples = samples

        self.vec_a = None
        self.vec_delta_m = None
        self.vec_d_delta_m = None

    def __str__(self) -> str: return f'{self.__class__.__name__}({self.cosmology.__str__()})'

    @property
    @abc.abstractmethod
    def s_i(self) -> list:
        """Returns initial value of S for differential equation solver."""

    @abc.abstractmethod
    def ds_da(self, s, a) -> list:
        r"""
        .. math:: dS/da = f(S,a)
        System of linear differential equations for the model.
        """

    def _solve_diff_eq(self) -> np.ndarray:
        self.vec_a = np.geomspace(self.a_i, 1, int(self.samples))

        return sc.integrate.odeint(
            func=self.ds_da, y0=self.s_i, t=self.vec_a, rtol=1e-10, atol=1e-10
        )

    @abc.abstractmethod
    def compute_equation(self) -> None:
        """Runs `_solve_diff_eq()` and stores the equation solution in vectors, for example, as `self.vec_delta_m`."""

    def f(self, a):
        r"""Returns interpolated :math:`f\equiv a \delta'/\delta`"""
        if self.vec_a is None or self.vec_delta_m is None or self.vec_d_delta_m is None:
            self.compute_equation()
        return np.interp(a, self.vec_a, self.vec_a * self.vec_d_delta_m / self.vec_delta_m)

    def gamma(self, a):
        r""":math:`\gamma(a)` function. Do not confound with the best fit of :math:`\gamma`."""
        return np.log(self.f(a)) / np.log(self.cosmology.omega_m(a))

    def fsigma8(self, a, gamma: float, sigma8: float):
        r""".. math:: f(z)\sigma_8(z) = \Omega_m(z)^\gamma \cdot \sigma_8(z=0) \cdot \exp[\int_1^a \Omega_m(z)^\gamma / a]"""
        f_fit = lambda a_: self.cosmology.omega_m(a_) ** gamma

        return f_fit(a) * sigma8 * np.exp(
            sc.integrate.quad(lambda a_: f_fit(a_)/a_, 1, a)[0]
        )


class SDE(CollapseModel):
    """Spherical Collapse Model with Smooth Dark Energy"""
    @property
    def s_i(self):
        return [self.delta_m_i, self.d_delta_m_i]

    def ds_da(self, s, a):
        c = self.cosmology
        delta_m, d_delta_m = s
        return [
            d_delta_m,  # (delta_m)'
            -(2 - (1 + 3 * c.w(a) * c.omega_de(a)) / 2) * d_delta_m / a + 3 / 2 * c.omega_m(a) / a ** 2 * delta_m  # (d_delta_m)'
        ]

    def compute_equation(self):
        sol = self._solve_diff_eq()

        self.vec_delta_m = sol.T[0]
        self.vec_d_delta_m = sol.T[1]


class CDE(CollapseModel):
    """Spherical Collapse Model with Clustering Dark Energy and negligible sound speed.

    :param delta_de_i: initial dark energy perturbation size
    :param q_i: initial physical sound speed divergence parameter size
    """
    def __init__(self, *args, delta_de_i=None, q_i=None, **kwargs):
        super().__init__(*args, **kwargs)

        self.q_i = - self.a_i**2 * self.d_delta_m_i if q_i is None else q_i
        self.delta_de_i = self.delta_m_to_delta_de(self.delta_m_i, self.a_i) if delta_de_i is None else delta_de_i

        self.vec_delta_de = None

    @property
    def s_i(self):
        return [self.delta_m_i, self.delta_de_i, self.q_i]

    def ds_da(self, s, a):
        c = self.cosmology
        delta_m, delta_de, q = s
        return [
            -q/a**2,   # (delta_m)'
            3 * c.w(a)/a * delta_de - (1+c.w(a)) * q/a**2,  # (delta_de)'
            1/2 * ((1 + 3 * c.w(a) * c.omega_de(a)) * q / a
                   - 3 * (c.omega_m(a) * delta_m + c.omega_de(a) * delta_de))  # q'
        ]

    def compute_equation(self):
        sol = self._solve_diff_eq()

        self.vec_delta_m = sol.T[0]
        self.vec_delta_de = sol.T[1]
        self.vec_d_delta_m = - sol.T[2] / self.vec_a ** 2

    def delta_m_to_delta_de(self, delta_m, a):
        r""".. math:: \delta_{de}=(1+w)/(1-3w) * \delta_m
        in a EdS approximation."""
        c = self.cosmology
        return (1+c.w(a)) / (1-3*c.w(a)) * delta_m

    def delta_ratio(self, z):
        r"""Linear Interpolation of :math:`\delta_{de}/\delta_m` at given `z`. If equation not computed,
        will raise exception."""
        if self.vec_a is None or self.vec_delta_m is None or self.vec_delta_de is None:
            self.compute_equation()
        return np.interp(z, 1/self.vec_a - 1, self.vec_delta_de/self.vec_delta_m)


class GeffCollapse(CollapseModel, abc.ABC):
    """Abstract spherical collapse model with Geff modified gravity considering Smooth Dark Energy."""
    @abc.abstractmethod
    def Geff(self, a) -> float:
        """.. math:: G_{eff}(a)"""

    @property
    def s_i(self):
        return [self.delta_m_i, self.d_delta_m_i]

    def ds_da(self, s, a):
        c = self.cosmology
        delta_m, d_delta_m = s
        return [
            d_delta_m,  # (delta_m)'
            -(2 - (1 + 3 * c.w(a) * c.omega_de(a)) / 2) * d_delta_m / a
            + 3 / 2 * c.omega_m(a) / a ** 2 * delta_m * self.Geff(a)
        ]

    def compute_equation(self):
        sol = self._solve_diff_eq()

        self.vec_delta_m = sol.T[0]
        self.vec_d_delta_m = sol.T[1]


class Gg0g1g2(GeffCollapse):
    def __init__(self, cosmology: csm.CosmologyModel, g0: float, g1: float, g2: float, **kwargs):
        super().__init__(cosmology, **kwargs)
        self.g0 = g0
        self.g1 = g1
        self.g2 = g2

    def Geff(self, a):
        return self.g0 + self.g1 * (a-1) + 1/2*self.g2 * (a-1)**2


class Gg0g2(GeffCollapse):
    def __init__(self, cosmology: csm.CosmologyModel, g0: float, g2: float, **kwargs):
        super().__init__(cosmology, **kwargs)
        self.g0 = g0
        self.g2 = g2

    def Geff(self, a):
        return self.g0 + 1/2*self.g2 * (a-1)**2


class Gg2(GeffCollapse):
    def __init__(self, cosmology: csm.CosmologyModel, g2: float, **kwargs):
        super().__init__(cosmology, **kwargs)
        self.g2 = g2

    def Geff(self, a):
        return 1 + self.g2 * ((a-1)**2 - 1)/2


class GDE(GeffCollapse):
    r"""Modified gravity model with :math:`G_{eff}=1 + \mu_0 \Omega_{de}(a)/\Omega_\Lambda`"""
    def __init__(self, cosmology: csm.CosmologyModel, mu0: float, **kwargs):
        super().__init__(cosmology, **kwargs)
        self.mu0 = mu0

    def Geff(self, a):
        return 1 + self.mu0 * self.cosmology.omega_de(a) / self.cosmology.omega_de(1)

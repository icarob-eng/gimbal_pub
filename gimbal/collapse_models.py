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
        """:math:`\gamma(a)` function. Do not confound with the best fit of :math:`\gamma`."""
        return np.log(self.f(a)) / np.log(self.cosmology.omega_m(a))

    def fsigma8(self, a, gamma: float, sigma8: float):
        f_fit = lambda a_: self.cosmology.omega_m(a_) ** gamma

        return sigma8 * f_fit(a) * np.exp(
            sc.integrate.quad(lambda a_: f_fit(a_)/a_, 1, a)[0]
        )


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

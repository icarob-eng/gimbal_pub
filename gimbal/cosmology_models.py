import numpy as np

import abc


class CosmologyModel(abc.ABC):
    r"""
    Abstract Cosmology Model. Represents only cosmological background mechanics. By default, assumes flat radiation free
     universe.

    :param omega_m0: matter density parameter for z=0: :math:`\Omega_{m0}`
    """
    def __init__(self, h, omega_m0, **kwargs):
        self.h = h
        self.omega_m0 = omega_m0

    @abc.abstractmethod
    def __str__(self) -> str:
        """Return string with all basic parameters of cosmology"""

    @abc.abstractmethod
    def w(self, a):
        """Equation of State parameter at scale factor `a`."""

    @abc.abstractmethod
    def de_density_eq(self, a):
        r"""Solution of the fluid equation with the Equation of State of Dark Energy: :math:`\epsilon(a)`"""

    def friedmann_eq(self, a):
        r"""Right Hand Side of Friedmann equation:
        .. math:: E = H/H_0 = \sqrt{\sum \Omega_{i0} * d(a)}
        for each component `i` of the universe, where `d(a)`is the density equation for each component.
        Assumes flat Matter-Dark Energy only universe.
        """
        return (self.omega_m0 * a**-3 + (1 - self.omega_m0) * self.de_density_eq(a)) ** (1/2)

    def omega_m(self, a):
        r"""Density parameter for matter at scale factor `a`.
        .. math:: \Omega_m(a) = (\Omega_{m0} a^{-3}) / E(a)^2
        where `E(a)^2` is the critical density parameter given by the Friedmann Equation
        """
        return self.omega_m0 * a ** -3 / self.friedmann_eq(a) ** 2

    def omega_de(self, a):
        """Density parameter for Dark Energy at scale factor `a`. Assumes flat Matter-Dark Energy only universe."""
        return 1 - self.omega_m(a)

    def H(self, a):
        return self.friedmann_eq(a) * 100 * self.h


class LCDM(CosmologyModel):
    r"""Basic :math:`\Lambda CDM` cosmology."""

    def __str__(self) -> str: return f'{self.__class__.__name__}({self.h}, {self.omega_m0})'

    def w(self, a): return -1

    def de_density_eq(self, a): return 1

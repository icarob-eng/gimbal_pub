import numpy as np
import scipy as sc

from functools import partial

from . import cosmology_models as csm
from . import collapse_models as clm


def gamma_chi2_dist(gamma: float, vec_omega_m: np.ndarray, vec_f: np.ndarray) -> float:
    r"""Compares :math:`\Omega_m^\gamma(a)` with :math:`f(a)` considering fixed :math:`\gamma`."""
    return np.sum((vec_omega_m ** gamma - vec_f) ** 2, axis=0)

a_samples_chi2 = np.linspace(1, 1 / 3, 50)  # list of scale factors for comparing numerical gamma vs gamma_samples
gamma_samples = np.linspace(0.001, 0.999, 300)  # list of possible values of gamma to compare using chi2
a_samples_residuals = np.linspace(1, 1 / 3, 10)  # list of scale factors for sample residuals
def get_best_gamma(collapse_model: clm.CollapseModel, *_) -> list[float]:
    """
    Compute Gamma by minimizing chi2 distribution. The minimization is done via the roots of the derivative of the
    spline of the curve of the chi2 distribution.

    :param collapse_model: instance of `CollapseModel` child class

    :return: best gamma value computed, root-mean-square percent residual related to gamma OR NaN, NaN if not possible
    to compute gamma
    """
    if collapse_model.vec_delta_m is None:
        collapse_model.compute_equation()

    chi2_of_gamma = partial(gamma_chi2_dist,
                            vec_omega_m=collapse_model.cosmology.omega_m(a_samples_chi2),
                            vec_f=collapse_model.f(a_samples_chi2)
                            )  # passes model data to function

    # most of the cost of this function:
    chi2_inter = sc.interpolate.InterpolatedUnivariateSpline(
        gamma_samples,
        [chi2_of_gamma(gamma) for gamma in gamma_samples],
        k=4
    )

    roots = chi2_inter.derivative().roots()

    if len(roots) != 1:
        return [np.nan, np.nan, np.nan]
    best_gamma = roots[0]

    vec_percent_residuals = (collapse_model.cosmology.omega_m(a_samples_residuals) ** best_gamma
                             / collapse_model.f(a_samples_residuals) - 1) * 100
    rms_percent_residual = np.sqrt(np.mean(np.square(vec_percent_residuals)))

    return [best_gamma, best_gamma, rms_percent_residual]

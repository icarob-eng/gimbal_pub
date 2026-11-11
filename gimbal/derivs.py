import numpy as np
import scipy as sc
from camb import camb, model
from functools import partial

from . import cosmology_models as csm
from . import collapse_models as clm


def gamma_chi2_dist(gamma: float, vec_omega_m: np.ndarray, vec_f: np.ndarray) -> float:
    r"""Compares :math:`\Omega_m^\gamma(a)` with :math:`f(a)` considering fixed :math:`\gamma`."""
    return np.sum((vec_omega_m ** gamma - vec_f) ** 2, axis=0)

a_samples_chi2 = np.linspace(1, 1 / 3, 50)  # list of scale factors for comparing numerical gamma vs gamma_samples
gamma_samples = np.linspace(0.001, 0.999, 300)  # list of possible values of gamma to compare using chi2
a_samples_residuals = np.linspace(1, 1 / 3, 10)  # list of scale factors for sample residuals
# Declaration outside function could be faster (by now, 1.2.0, doesn't make a difference)
def get_best_gamma(collapse_model: clm.CollapseModel, *_) -> list[float]:
    """
    Compute Gamma by minimizing chi2 distribution. The minimization is done via the roots of the derivative of the
    spline of the curve of the chi2 distribution. Should not differ from `get_best_gamma_optimized()`, but it's a ~25% slower, and
    it's more numerically stable.

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
    chi2_inter = sc.interpolate.make_interp_spline(
        gamma_samples,
        [chi2_of_gamma(gamma) for gamma in gamma_samples],
        k=4
    )

    # roots = chi2_inter.derivative().roots()  # BSpline.roots() is deprecated
    # pattern recommended: from https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.sproot.html
    roots = sc.interpolate.PPoly.from_spline(chi2_inter.derivative()).roots()  # Piecewise polynomial from spline

    if len(roots) != 1:
        return [np.nan, np.nan, np.nan]
    best_gamma = roots[0]

    vec_percent_residuals = (collapse_model.cosmology.omega_m(a_samples_residuals) ** best_gamma
                             / collapse_model.f(a_samples_residuals) - 1) * 100
    rms_percent_residual = np.sqrt(np.mean(np.square(vec_percent_residuals)))

    return [best_gamma, best_gamma, rms_percent_residual]


def get_best_gamma_optimized(collapse_model: clm.CollapseModel, *_) -> list[float]:
    """
    Compute Gamma by minimizing chi2 distribution via `scipy.optimize.minimize()`. Should not differ from
    `get_best_gamma()`, it's a ~25% faster but can be numerically unstable.'`

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
    results = sc.optimize.minimize(chi2_of_gamma, np.array([0.55]), bounds=[(0.3, 0.8)])

    if not results.success:
        return [np.nan, np.nan, np.nan]
    best_gamma = results.x[0]

    vec_percent_residuals = (collapse_model.cosmology.omega_m(a_samples_residuals) ** best_gamma
                             / collapse_model.f(a_samples_residuals) - 1) * 100
    rms_percent_residual = np.sqrt(np.mean(np.square(vec_percent_residuals)))

    return [best_gamma, best_gamma, rms_percent_residual]


def delta_ratio(collapse_model: clm.CollapseModel, *_) -> list[float]:
    if isinstance(collapse_model, clm.CDE):
        if collapse_model.vec_delta_m is None:
            collapse_model.compute_equation()

        return [collapse_model.delta_ratio(z=0), collapse_model.delta_ratio(z=0.5)]
    else:
        return [np.nan, np.nan]


def sigma8_0(collapse_model: clm.CollapseModel, *_) -> list[float]:
    """Computes sigma8 for z=0 using model parameters"""
    ombh2 = 0.022
    pars = camb.CAMBparams()
    omh2 = collapse_model.cosmology.omega_m0 * collapse_model.cosmology.h**2
    pars.set_cosmology(
        H0=collapse_model.cosmology.h * 100,
        ombh2=ombh2,
        omch2=omh2 - ombh2,
        mnu=0.06,
        omk=0,
        tau=0.06
    )

    w0 = collapse_model.cosmology.w(1)
    wa = collapse_model.cosmology.wa if isinstance(collapse_model.cosmology, csm.CPL) else 0
    cs2 = 0 if isinstance(collapse_model, clm.CDE) else 1

    pars.set_dark_energy(w=w0, wa=wa, cs2=cs2, dark_energy_model='ppf')
    pars.InitPower.set_params(As=2e-9, ns=0.965, r=0)

    # pars.set_for_lmax(2500, lens_potential_accuracy=0)
    pars.set_matter_power(redshifts=(0,), kmax=2.0)
    pars.NonLinear = model.NonLinear_none

    return [camb.get_results(pars).get_sigma8_0()]
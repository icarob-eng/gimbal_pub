# GIMBAL: Growth Index Modeling for Bayesian ALgorithms

This library is an Object-Oriented modeling of the Friedmann equation with a flat
Friedmann-Lemaitre-Robertson-Walker metric as used in the papers 
[Phys. Rev. D 112, 043532](https://doi.org/10.1103/71jl-gmrn) ([arXiv:2411.00963](https://arxiv.org/abs/2411.00963)), 
[Phys. Rev. D 113, 043541](https://doi.org/10.1103/gcc3-5k9d) ([arXiv:2510.15054](https://arxiv.org/abs/2510.15054)).

This library counts with some classes of cosmological backgrounds:
- $\Lambda$CDM: `LCDM(h, omega_m0)`;
- $w$CDM: `wCDM(h, omega_m0, w)`;
- $w_0w_a$CDM: `CPL(h, omega_m0, w0, wa)`;

And some growth of linear structure models:
- Smooth Dark Energy (see [arXiv:2411.00963](https://arxiv.org/abs/2411.00963) for details): `SDE(cosmology)`;
- Clustering Dark Energy (see [arXiv:2411.00963](https://arxiv.org/abs/2411.00963) for details): `CDE(cosmology)`
- $\mu$ parametrization of $G_\text{eff}(a)$ coupled with dark energy (see
[arXiv:2510.15054](https://arxiv.org/abs/2510.15054) for details): `GDE(cosmology, mu0)`;
- Some variations of the Taylor series expansion of $G_\text{eff}(a)$ (see
[arXiv:2510.15054](https://arxiv.org/abs/2510.15054) for details), like `Gg2(cosmology, g2)`;


The effective gravitational constant $G_\text{eff}(a)$, when altered, is used to compute the system of first order differential
equations for a matter density contrast $\delta_m$ in the abstract class `collapse_models.CollapseModel`,
with class defined default initial conditions.

It is then possible to access the values of
$\delta_m(a)$ and $\delta_m^\prime(a)$ for a class defined interval of the scale factor $a$.

The `CollapseModel` class also defines the methods
- `f(a)`: a interpolation of $f(a)\equiv a \delta'/\delta$;
- `gamma(a)`: $\gamma(a)$ (not the bestfit for constant $\gamma$, see bellow);
- `fsigma8(a, gamma: float, sigma8: float):`: $f(z)\sigma_8(z) = \Omega_m(z)^\gamma \cdot \sigma_8(z=0) \cdot \exp[\int_1^a \Omega_m(z)^\gamma / a]$

Finally, to compute $\gamma$ we pass a `CollapseModel`
to the function
```python
def get_best_gamma(collapse_model: clm.CollapseModel, *_) -> list[float]:
    """
    Compute Gamma by minimizing chi2 distribution. The minimization is done via the roots of the derivative of the
    spline of the curve of the chi2 distribution. Should not differ from `get_best_gamma_optimized()`, but it's a ~25% slower, and
    it's more numerically stable.

    :param collapse_model: instance of `CollapseModel` child class

    :return: best gamma value computed, root-mean-square percent residual related to gamma OR NaN, NaN if not possible
    to compute gamma
    """
```
As shown, it returns the best value of $\gamma$ and it's residuals.

Examples of the use of some of the models in this library are included in the notebook `examples/models.ipybn`.

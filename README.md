# GIMBAL: Growth Index Modeling with Background as LambdaCDM

This library has an Object-Oriented modeling of the Friedmann equation with a flat
Friedmann-Lemaitre-Robertson-Walker metric as used in the paper [...]. The current version counts only with a LCDM
background, but in the future we will publish more cosmological backgrounds.

The library has an implementation of two collapse models with the LCDM background, both with
the option of a modified effective gravitational constant, $G_\text{eff}$. The first one
is a phenomenological parametrization with the free parameter `g2`:
```python
class Gg2(GeffCollapse):
    def __init__(self, cosmology: csm.CosmologyModel, g2: float, **kwargs):
        super().__init__(cosmology, **kwargs)
        self.g2 = g2

    def Geff(self, a):
        return 1 + self.g2 * ((a-1)**2 - 1)/2
```

The second one is labeled in the code as GDE, as it associates $G_\text{eff}$ with the
Dark Energy (DE) density parameter, with the free parameter `mu0`:
```python
class GDE(GeffCollapse):
    r"""Modified gravity model with :math:`G_{eff}=1 + \mu_0 \Omega_{de}(a)/\Omega_\Lambda`"""
    def __init__(self, cosmology: csm.CosmologyModel, mu0: float, **kwargs):
        super().__init__(cosmology, **kwargs)
        self.mu0 = mu0

    def Geff(self, a):
        return 1 + self.mu0 * self.cosmology.omega_de(a) / self.cosmology.omega_de(1)
```

The effective gravitational constant is used to compute the system of first order differential
equations for a matter density contrast $\delta_m$ in the abstract class `collapse_models.CollapseModel`,
with class defined default initial conditions. It is then possible to access the values of
$\delta_m(a)$ and $\delta_m^\prime(a)$ for a class defined interval of the scale factor $a$.

The `CollapseModel` class also defines the method
```python
def f(self, a):
    r"""Returns interpolated :math:`f\equiv a \delta'/\delta`"""
```

Which is used to interpolate values of $f$. Finally, to compute $\gamma$ we pass a `CollapseModel`
to the function
```python
def get_best_gamma(collapse_model: clm.CollapseModel, *_) -> list[float]:
    """
    Compute Gamma by minimizing chi2 distribution. The minimization is done via the roots of the derivative of the
    spline of the curve of the chi2 distribution.

    :param collapse_model: instance of `CollapseModel` child class

    :return: best gamma value computed, root-mean-square percent residual related to gamma OR NaN, NaN if not possible
    to compute gamma
    """
```
As shown, it returns the best value of $\gamma$ and it's residuals.

Examples of the use of the models in this library are included in the notebook `examples/models.ipybn`.

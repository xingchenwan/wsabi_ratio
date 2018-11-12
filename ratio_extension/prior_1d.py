# Xingchen Wan
# This is a 1D implementation for the Gaussian prior defined in bayesquad.priors. Documentations can be found in that
# file.

from bayesquad.priors import Prior
from scipy.stats import norm
import numpy as np
from typing import Union


class Gaussian1D(Prior):
    def __init__(self, mean: float, variance: float):
        self.mean = mean
        self.variance = variance
        self.precision = 1 / self.variance
        self._normal = norm

        # Just to make the mean and variance amenable to matrix operations
        self.matrix_variance = np.array([[variance]])
        self.matrix_mean = np.array([[mean]])
        self.matrix_precision = np.array([[1./variance]])

    def sample(self,) -> np.ndarray:
        res = norm.rvs(loc=self.mean, scale=np.sqrt(self.variance))
        return np.array([res])

    def gradient(self, x: Union[float, list, np.ndarray]):

        def _get_derivs(x, mean, var):
            first_deriv = -(x - mean) / np.sqrt(2*np.pi) / (var ** 1.5) * \
                      np.exp(-((x - mean) ** 2) / (2 * var))
            second_deriv = 1 / (np.sqrt(2 * np.pi) * var * 2.5) * ((x - mean) ** 2 - var) * \
                      np.exp(-((x - mean) ** 2) / (2 * var))
            return first_deriv, second_deriv

        x = np.asarray(x)
        assert x.ndim == 2
        derivs = np.array([_get_derivs(x[i], self.mean, self.variance) for i in range(x.shape[0])])
        jacobian = derivs[:, 0]
        hessian = derivs[:, 1].reshape(x.shape[0], 1, 1)
        return jacobian, hessian

    def __call__(self, x: Union[np.ndarray, list]) -> np.ndarray:
        return np.array([norm.pdf(np.asscalar(each_x), loc=self.mean, scale=self.variance) for each_x in x])

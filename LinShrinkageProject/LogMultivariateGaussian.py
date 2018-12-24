# LogMultivariate Gaussian Implementation - written in compliance with the GPy priors interface
# Xingchen Wan | 2018

import numpy as np
from GPy.core.parameterization.priors import MultivariateGaussian, Prior
from GPy.util.linalg import pdinv
from paramz.domains import _POSITIVE
import weakref


class LogMultivariateGaussian(MultivariateGaussian):

    domain = _POSITIVE
    _instances = []

    def __new__(cls, mu=0, var=1):  # Singleton:
        if cls._instances:
            cls._instances[:] = [instance for instance in cls._instances if instance()]
            for instance in cls._instances:
                if np.all(instance().mu == mu) and np.all(instance().var == var):
                    return instance()
        o = super(Prior, cls).__new__(cls, mu, var)
        cls._instances.append(weakref.ref(o))
        return cls._instances[-1]()

    def __init__(self, mu, var):
        self.mu = np.array(mu).flatten()
        self.var = np.array(var)
        assert len(self.var.shape) == 2
        assert self.var.shape[0] == self.var.shape[1]
        assert self.var.shape[0] == self.mu.size
        self.input_dim = self.mu.size
        self.inv, self.hld = pdinv(self.var)
        self.constant = -0.5 * self.input_dim * np.log(2 * np.pi)

    def summary(self):
        raise NotImplementedError

    def pdf(self, x):
        return np.exp(self.lnpdf(x))

    def lnpdf(self, x):
        log_jacobian = np.log(1 / np.prod(x))
        return self.constant + log_jacobian - 0.5 * np.transpose((np.log(x) - self.mu)) @ \
               self.inv @ (np.log(x) - self.mu)

    def lnpdf_grad(self, x):
        pass

    def rvs(self, n):
        return np.exp(np.random.multivariate_normal(self.mu, self.var, n))

    def __getstate__(self):
        return self.mu, self.var

    def __setstate__(self, state):
        self.mu = state[0]
        self.var = state[1]
        assert len(self.var.shape) == 2
        assert self.var.shape[0] == self.var.shape[1]
        assert self.var.shape[0] == self.mu.size
        self.input_dim = self.mu.size



# Xingchen Wan | 2018
# Implementation of the naive Bayesian quadrature for ratios using WSABI

from bayesquad.quadrature import IntegrandModel
from bayesquad.batch_selection import select_batch
from bayesquad.gps import WsabiLGP
from bayesquad.priors import Prior
from ratio_extension.test_functions import TrueFunctions
import numpy as np
import GPy
import matplotlib.pyplot as plt


class NaiveWSABI:
    """
    Naive WSABI models the numerator and denominator integrand independently using WSABI algorithm. For
    reference, the ratio of integrals is in the form of:
    \math
        \frac{\int q(\phi)r(\phi)p(\phi)d\phi}{\int r(\phi)p(\phi)d\phi}
    \math
    where q(\phi) = p(y|z, \phi) and r(\phi) = p(z|\phi). These functions can be evaluated but are somewhat expensive.
    The objective is to infer a functional form of both r and q using Gaussian process then using Gaussian quadrature
    to complete the integration. Note that the naive method does not take into consideration of the correlation in
    the numerator and denominator.

    Since the denominator is in the form of a (fairly simple) Bayesian quadrature problem, the denominator integral is
    evaluated first. For this case, the samples selected on the hyperparameter (\phi) space are also used to evaluate
    the numerator integrand.
    """

    def __init__(self, r: TrueFunctions, q: TrueFunctions, p: Prior,
                 **options):
        assert r.dimensions == q.dimensions, \
            "The dimensions of the numerator and denominator do not match!"
        self.dim = r.dimensions
        self.r = r
        self.q = q
        self.p = p

        # Initialise the GPy GP instances and the WSABI-L model for the numerator and denominator integrands
        self.gpy_gp_den = None
        self.gpy_gp_num = None
        self.model_den = None
        self.model_num = None

        self.options = self._unpack_options(options)
        self.results = [np.nan] * self.options["num_batches"]
        self.initialise_gp()

    def _batch_iterate(self, ):
        # Active sampling by minimising the variance of the *integrand*, and then update the corresponding Gaussian
        # Process
        batch_phi = select_batch(self.model_den, self.options['batch_size'], "LOCAL_PENALISATION")
        r_sample = self.r.sample(batch_phi)
        batch_y_den = np.sqrt(r_sample)
        self.model_den.update(batch_phi, batch_y_den)
        self.gpy_gp_den.optimize()
        batch_y_num = np.sqrt(r_sample * self.q.sample(batch_phi))
        self.model_num.update(batch_phi, batch_y_num)
        self.gpy_gp_num.optimize()
        return self.model_num.integral_mean() / self.model_den.integral_mean()

    def quadrature(self):
        for i in range(self.options['num_batches']):
            self.results[i] = self._batch_iterate()
        return self.results[-1]

    def initialise_gp(self):
        """
        Initialise the Gaussian process approximations to both the numerator and denominator
        """
        init_x = np.zeros((self.dim, ))
        r_sample = self.r.sample(init_x)
        init_y_den = np.sqrt(r_sample)
        init_y_num = np.sqrt(r_sample * self.q.sample(init_x))
        # Note the square-root warping of the y values in the denominator

        self.gpy_gp_den = GPy.core.GP(init_x, init_y_den,
                                      kernel=self.options['kernel'], likelihood=self.options['likelihood'])
        warped_gp = WsabiLGP(self.gpy_gp_den)
        self.model_den = IntegrandModel(warped_gp, self.p)
        self.gpy_gp_num = GPy.core.GP(init_x, init_y_num,
                                      kernel=self.options['kernel'], likelihood=self.options['likelihood'])
        self.model_num = IntegrandModel(WsabiLGP(self.gpy_gp_num), self.p)

    def _unpack_options(self,
                        kernel: GPy.kern = None,
                        likelihood: GPy.likelihoods = GPy.likelihoods.Gaussian(variance=1e-10),
                        batch_size: int = 4,
                        num_batches: int = 25):
        if kernel is None:
            kernel = GPy.kern.RBF(self.dim, variance=2, lengthscale=2)
        return {
            "kernel": kernel,
            "likelihood": likelihood,
            'batch_size': batch_size,
            'num_batches': num_batches
        }

    def plot_result(self, true_value: float):
        if np.nan in self.results:
            raise ValueError("Quadrature has not been run!")
        res = np.array(self.results)
        rmse = (res - true_value) ** 2
        plt.plot(rmse)
        plt.xlabel("Number of batches")
        plt.ylabel("RMSE")


class NaiveBQ:
    pass


class NaiveMonteCarlo:
    pass

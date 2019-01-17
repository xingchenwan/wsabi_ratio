# Parameter Posterior
# Xingchen Wan | xingchen.wan@st-annes.ox.ac.uk | Jan 2019

from ratio.functions import Functions, Unity
from ratio.monte_carlo import MonteCarlo
from bayesquad.priors import Gaussian
import numpy as np
from typing import Union, Tuple
from scipy.stats import moment
import pymc3 as pm

import matplotlib
matplotlib.use("TkAgg")
# This is to prevent a macOS bug with matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import multivariate_normal
import theano.tensor as tt
import pandas as pd


class ParamPosterior:
    def __init__(self, regression_model: Functions, **kwargs):
        self.gpr = regression_model
        self.options = self._unpack_options(**kwargs)
        self.dimensions = self.gpr.dimensions

    def smc(self):
        """
        Using PyMC3 for Monte Carlo Sampling
        :return:
        """
        prior = Gaussian(mean=self.options['prior_mean'].reshape(-1),
                         covariance=self.options['prior_variance'])
        budget = self.options['smc_budget']
        samples = np.zeros((budget, self.dimensions))
        first_moment = np.zeros((budget, self.dimensions))
        second_moment = np.zeros((budget, self.dimensions, self.dimensions))
        kl = np.zeros((budget, ))
        # KL divergence - only applicable if the ground truth posterior distribution is somehow known a priori

        data = np.random.randn(20)

        def log_lik(x):
            """
            Log likelihood (not pdf) must be supplied here - otherwise the Monte Carlo Sampling will fail!
            :param x:
            :return:
            """
            return self.gpr.log_sample(x)

        basic_model = pm.Model()
        with basic_model:
            x = pm.Normal('x', mu=0, sd=np.sqrt(2), shape=self.dimensions)
            like = pm.DensityDist('like', log_lik,  shape=self.dimensions,)
            start = pm.find_MAP(model=basic_model)
            trace = pm.sample(10, start=start)

        samples = trace.get_values('x')
        yv = trace.get_values('like')
        self.save_result(samples, yv, file_name='SMCPosteriorSamples')

    def bq(self):
        pass

    def wsabi(self):
        pass

    # --- Evaluation Metric ---#
    def gauss_sym_kl_div(self, grd_mean, grd_cov, test_mean, test_cov):
        mu_1 = grd_mean
        sigma_1 = grd_cov
        try:
            sigma_inv_1 = np.linalg.inv(sigma_1)
        except np.linalg.LinAlgError:
            return np.nan

        mu_2 = test_mean
        sigma_2 = test_cov
        try:
            sigma_inv_2 = np.linalg.inv(sigma_2)
        except np.linalg.LinAlgError:
            return np.nan

        mu_diff = mu_1 - mu_2

        kl_1 = 0.5 * np.log(np.linalg.det(sigma_2) / np.linalg.det(sigma_1)) - self.dimensions + \
            np.trace(sigma_inv_2 @ sigma_1) + mu_diff.T @ sigma_2 @ mu_diff
        kl_2 = 0.5 * np.log(np.linalg.det(sigma_1) / np.linalg.det(sigma_2)) - self.dimensions + \
            np.trace(sigma_inv_1 @ sigma_2) + mu_diff.T @ sigma_1 @ mu_diff
        return kl_1 + kl_2

    # --- Utility Functions ---#
    def _unpack_options(self,
                        prior_mean: Union[float, np.ndarray] = 0.,
                        prior_variance: Union[float, np.ndarray] = 2.,
                        smc_budget: int = 100,
                        naive_bq_budget: int = 1000,
                        naive_bq_kern_lengthscale: float = 1.,
                        naive_bq_kern_variance: float = 1.,
                        wsabi_bq_budget: int = 1000,
                        ) -> dict:
        """
        Unpack kwargs
        :param prior_mean and prior_variance: Prior mean and variance in log-space of the likelihood function
        :return: a dictionary for the use of the object
        """
        if self.gpr.dimensions > 1 and isinstance(prior_variance, float) and isinstance(prior_mean, float):
            prior_mean = np.array([prior_mean]*(self.gpr.dimensions)).reshape(-1, 1)
            prior_variance *= np.eye(self.gpr.dimensions)
        else:
            assert len(prior_mean) == self.gpr.dimensions
            assert prior_variance.shape[0] == prior_variance.shape[1]
            assert prior_variance.shape[0] == self.gpr.dimensions
        return {
            'prior_mean': prior_mean,
            'prior_variance': prior_variance,
            'smc_budget': smc_budget,
            'naive_bq_budget': naive_bq_budget,
            'naive_bq_kern_lengthscale': naive_bq_kern_lengthscale,
            'naive_bq_kern_variance': naive_bq_kern_variance,
            'wsabi_bq_budget': wsabi_bq_budget,
        }

    def plot_iterations_mc(self, i, samples, kl_div=None):
        assert self.dimensions == 2, "Heatmap only applicable to 2D data!"
        plt.subplots(2, 1)
        plt.subplot(211)
        cmap = sns.cubehelix_palette(as_cmap=True, dark=0, light=1, reverse=False)
        x, y = samples[:i+1, 0], samples[:i+1, 1]
        a = sns.kdeplot(x, y, n_levels=10, shade=True, cmap=cmap, gridsize=200, cbar=True)
        a.set_xlabel('x')
        a.set_ylabel('y')
        if kl_div is not None:
            plt.subplot(212)
            xv = np.array([i for i in range(i+1)])
            sns.scatterplot(xv, np.log(kl_div[:i+1]))
        plt.show()

    def save_result(self, *x, file_name):
        save = pd.DataFrame([x])
        save.to_csv('~/Dropbox/4YP/Codes/wsabi_ratio/output/'+file_name+'.csv')
        return 0


# Code lifted from PyMC3 Tutorial site

# define a theano Op for our likelihood function
class LogLike(tt.Op):

    """
    Specify what type of object will be passed and returned to the Op when it is
    called. In our case we will be passing it a vector of values (the parameters
    that define our model) and returning a single "scalar" value (the
    log-likelihood)
    """
    itypes = [tt.dvector] # expects a vector of parameter values when called
    otypes = [tt.dscalar] # outputs a single scalar value (the log likelihood)

    def __init__(self, loglike, x):
        """
        Initialise the Op with various things that our log-likelihood function
        requires. Below are the things that are needed in this particular
        example.

        Parameters
        ----------
        loglike:
            The log-likelihood (or whatever) function we've defined
        data:
            The "observed" data that our log-likelihood function takes in
        x:
            The dependent variable (aka 'x') that our model requires
        sigma:
            The noise standard deviation that our function requires.
        """

        # add inputs as class attributes
        self.likelihood = loglike
        self.x = x

    def perform(self, node, inputs, outputs):
        # the method that is used when calling the Op
        theta, = inputs  # this will contain my variables

        # call the log-likelihood function
        logl = self.likelihood(self.x,)

        outputs[0][0] = np.array(logl) # output the log-likelihood

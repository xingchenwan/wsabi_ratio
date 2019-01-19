# Yacht Regression Experiment
# Xingchen Wan | xingchen.wan@st-annes.ox.ac.uk | Jan 2019

from ratio.functions import GPRegressionFromFile
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
import theano.tensor as tt
import pandas as pd

import GPy
from bayesquad.quadrature import WarpedIntegrandModel, WsabiLGP, WarpedGP, GP
from bayesquad.batch_selection import select_batch
from IPython.display import display


class RegressionQuadrature:
    def __init__(self, regression_model: GPRegressionFromFile, **kwargs):
        self.gpr = regression_model
        self.options = self._unpack_options(**kwargs)
        self.dimensions = self.gpr.dimensions

        # Parameter prior - note that the prior is in *log-space*
        self.prior = Gaussian(mean=self.options['prior_mean'].reshape(-1),
                              covariance=self.options['prior_variance'])
        self.gp = self.gpr.model

    def maximum_a_posterior(self):
        """
        Using a point estimate of the log-likelihood function
        :return: an array of predictions, y_pred
        """
        # Create a local version of the model
        local_gp = self.gp.copy()

        # Fetch and train data from the GPy GP object
        X = self.gpr.X_train
        Y = self.gpr.Y_train
        X_test = self.gpr.X_test
        local_gp.set_XY(X, Y)

        # Assign prior to the hyperparameters
        variance_prior = GPy.priors.LogGaussian(mu=0., sigma=2)
        lengthscale_prior = GPy.priors.LogGaussian(mu=0., sigma=2.)
        noise_prior = GPy.priors.LogGaussian(mu=0., sigma=2.)
        local_gp.kern.variance.set_prior(variance_prior)
        local_gp.kern.lengthscale.set_prior(lengthscale_prior)
        local_gp.Gaussian_noise.variance.set_prior(noise_prior)

        # Optimise under MAP
        local_gp.optimize_restarts(num_restarts=10, max_iters=1000)
        pred, pred_var = local_gp.predict(X_test)
        rmse = self.compute_rmse(pred, self.gpr.Y_test)
        print('Root Mean Squared Error:', rmse)
        self.visualise(pred, self.gpr.Y_test)
        return rmse

    def mc(self):
        budget = self.options['smc_budget']

        # Firstly, create a MCMC sampler for the parameter posterior distribution - same as posterior.py

        def log_lik(phi): return self.gpr.log_sample(phi)

        model = pm.Model()
        with model:
            # The prior distribution of the parameter
            phi = pm.Lognormal('phi', mu=0, sd=2, shape=self.dimensions)
            # log-likelihood surface
            lik = pm.Density('lik', log_lik, shape=self.dimensions)

            # Initialise to MAP
            start = pm.find_MAP(model=model)
            trace = pm.sample(budget, start=start)
        samples = trace.get_values('like')

        # Now draw samples from the parameter posterior distribution and do Monte Carlo integration
        y_preds = np.zeros((budget, self.gpr.Y_test.shape[0]))
        pred = np.zeros(y_preds.shape)
        rmse = np.zeros((budget, ))

        for i in range(budget):
            y_preds[i, :] = self.gpr.log_sample(phi=samples[i, :], x=self.gpr.X_test)
            pred[i, :] = y_preds.sum(axis=0) / i
            rmse[i] = self.compute_rmse(pred[i], self.gpr.Y_test)

    def wsabi(self):
        pass

    # ----- Evaluation Metric ---- #

    def compute_rmse(self, y_pred: np.ndarray, y_grd: np.ndarray) -> float:
        """
        Compute the root mean squared error between prediction (y_pred) with the ground truth y in the test set
        :param y_pred: a vector with the same length as the test set y
        :return: rmse
        """
        y_pred = y_pred.reshape(-1)
        y_grd = y_grd.reshape(-1)
        length = y_pred.shape[0]
        assert length == y_grd.shape[0], "the length of the prediction vector does " \
                                               "not match the ground truth vector!"
        rmse = np.sqrt(np.sum((y_pred - y_grd) ** 2) / length)
        return rmse

    @staticmethod
    def visualise(y_pred: np.ndarray, y_grd: np.ndarray):
        plt.plot(y_pred, ".", color='red')
        plt.plot(y_grd, ".", color='blue')
        plt.show()

    def _unpack_options(self,
                        prior_mean: Union[float, np.ndarray] = 0.,
                        prior_variance: Union[float, np.ndarray] = 2.,
                        smc_budget: int = 10000,
                        naive_bq_budget: int = 1000,
                        naive_bq_kern_lengthscale: float = 1.,
                        naive_bq_kern_variance: float = 1.,
                        wsabi_bq_budget: int = 200,
                        posterior_range=(-5., 5.),
                        # The axis ranges between which the posterior samples are drawn
                        posterior_eps: float = 0.02,
                        ):
        if self.gpr.dimensions > 1 and isinstance(prior_variance, float) and isinstance(prior_mean, float):
            prior_mean = np.array([prior_mean] * (self.gpr.dimensions)).reshape(-1, 1)
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
            'posterior_range': posterior_range,
            'posterior_eps': posterior_eps
        }


if __name__ == '__main__':
    regression_model = GPRegressionFromFile()
    rq = RegressionQuadrature(regression_model)
    rq.maximum_a_posterior()
    rq.mc()

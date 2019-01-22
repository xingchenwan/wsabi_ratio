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

    def maximum_a_posterior(self, num_restarts=10, max_iters=1000):
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
        local_gp.optimize_restarts(num_restarts=num_restarts, max_iters=max_iters)
        pred, pred_var = local_gp.predict(X_test)
        rmse = self.compute_rmse(pred, self.gpr.Y_test)
        print('Root Mean Squared Error:', rmse)
        print('MAP lengthscale vector:', local_gp.rbf.lengthscale)
        display(local_gp)
        #self.visualise(pred, self.gpr.Y_test)
        return local_gp, rmse

    def mc(self):
        budget = self.options['smc_budget']

        # Firstly, create a MCMC sampler for the parameter posterior distribution - same as posterior.py

        logl = LogLike(self.gpr.log_sample, self.gpr.X_train)

        model = pm.Model()
        with model:
            # The prior distribution of the parameter

            phi = pm.Lognormal('phi', mu=0, sd=2, shape=self.dimensions)
            # log-likelihood surface
            lik = pm.DensityDist('lik', lambda phi: logl(phi), shape=self.dimensions)

            # Initialise to MAP
            # start = pm.find_MAP(model=model)
            trace = pm.sample(budget,)
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
        # Allocating number of maximum evaluations
        budget = self.options['wsabi_budget']
        test_x = self.gpr.X_test[4:5, :]
        test_y = self.gpr.Y_test[4:5]

        # Allocate memory of the samples and results
        log_phi = np.zeros((budget, self.gpr.dimensions,)) # The log-hyperparameter sampling points
        log_r = np.zeros((budget, )) # The log-likelihood function
        q = np.zeros((test_x.shape[0], budget)) # Prediction
        log_rq = np.zeros((test_x.shape[0], budget))
        rq = np.zeros((test_x.shape[0], budget))

        # Initial points - note that as per GPML convention, the hyperparameters are expressed in log scale
        # Initialise to the MAP estimate
        map_model, _ = self.maximum_a_posterior(num_restarts=1)
        self.gpr.set_params(variance=map_model.rbf.variance, gaussian_noise=map_model.Gaussian_noise.variance)
        log_phi_initial = np.log(map_model.rbf.lengthscale).reshape(1, -1)
        log_r_initial = self.gpr.log_sample(phi=np.exp(log_phi_initial))[0]
        r_initial = np.exp(log_r_initial)
        pred = np.zeros((test_x.shape[0], ))

        # Setting up kernel - Note we only marginalise over the lengthscale terms, other hyperparameters are set to the
        # MAP values.
        kern = GPy.kern.RBF(self.dimensions,
                            variance=2.,
                            lengthscale=2.)

        log_r_gp = GPy.models.GPRegression(log_phi_initial, log_r_initial.reshape(1, -1), kern)
        log_r_model = WarpedIntegrandModel(WsabiLGP(log_r_gp), self.prior)

        # Firstly, within the given allowance, compute an estimate of the model evidence. Model evidence is the common
        # denominator for all predictive distributions.
        for i_a in range(budget):
            log_phi_i = np.array(select_batch(log_r_model, 1, )).reshape(1, -1)
            log_r_i = self.gpr.log_sample(phi=np.exp(log_phi_i))[0]

            log_r[i_a] = log_r_i
            log_phi[i_a, :] = log_phi_i

            log_r_model.update(log_phi_i, log_r_i.reshape(1, -1))
        max_log_r = max(log_r)
        r = np.exp(log_r - max_log_r)
        r_gp = GPy.models.GPRegression(log_phi, r.reshape(-1, 1), kern)
        r_model = WarpedIntegrandModel(WsabiLGP(r_gp), self.prior)
        r_gp.optimize()
        r_int = np.exp(np.log(r_model.integral_mean()[0]) + max_log_r)
        print("Estimate of model evidence: ", r_int,)
        print("Model log-evidence ", np.log(r_int))

        # Secondly, compute and marginalise the predictive distribution for each individual points
        for i_x in range(test_x.shape[0]):

            # Note that we do not active sample again for q, we just use the same samples sampled when we compute
            # the log-evidence
            _, q_initial = self.gpr.log_sample(phi=np.exp(log_phi_initial), x=test_x[i_x, :])

            # Initialise GPy GP surrogate for and q(\phi)r(\phi)
            # Sample for q values
            for i_b in range(budget):
                log_phi_i = log_phi[i_b, :]
                log_r_i, q_i = self.gpr.log_sample(phi=np.exp(log_phi_i), x=test_x[i_x, :])
                q[i_x, i_b] = q_i
                # Update the model

            # Do the same exponentiation and rescaling trick for q
            log_rq[i_x, :] = log_r + np.log(q[i_x, :])
            max_log_rq = np.max(log_rq[i_x, :])
            rq[i_x, :] = np.exp(log_rq[i_x, :] - max_log_rq)

            rq_gp = GPy.models.GPRegression(log_phi_initial, rq[i_x, :])
            rq_model = WarpedIntegrandModel(WsabiLGP(rq_gp), self.prior)
            rq_gp.optimize()

            # Now estimate the posterior
            rq_int = np.exp(rq_model.integral_mean()[0] + max_log_rq)
            pred[i_x] = rq_int / r_int

        rmse = self.compute_rmse(pred, test_y)
        print(pred, test_y)
        print('Root Mean Squared Error:', rmse)
        self.visualise(pred, test_y)
        return rmse

    def wsabi_ratio(self):
        # Implementation of the Bayesian Quadrature for Ratio paper

        # Allocating number of maximum evaluations
        budget = self.options['wsab_bq_budget']

        phi_s = np.zeros((budget, self.gpr.dimensions)) # Corresponding to phi_s in the BQR paper

        # Initial points - in log space

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
                        smc_budget: int = 1000,
                        naive_bq_budget: int = 1000,
                        naive_bq_kern_lengthscale: float = 1.,
                        naive_bq_kern_variance: float = 1.,
                        wsabi_budget: int = 200,
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
            'wsabi_budget': wsabi_budget,
            'posterior_range': posterior_range,
            'posterior_eps': posterior_eps
        }


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
        data = inputs

        # call the log-likelihood function
        logl = self.likelihood(data)

        outputs[0][0] = np.array(logl) # output the log-likelihood

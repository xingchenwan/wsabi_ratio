# Parameter Posterior
# Xingchen Wan | xingchen.wan@st-annes.ox.ac.uk | Jan 2019

from ratio.functions import Functions, Unity
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


class ParamPosterior:
    def __init__(self, regression_model: Functions, **kwargs):
        self.gpr = regression_model
        self.options = self._unpack_options(**kwargs)
        self.dimensions = self.gpr.dimensions

        prior_mean = self.options['prior_mean'].reshape(-1)
        prior_cov = self.options['prior_variance']
        self.prior = Gaussian(mean=prior_mean, covariance=prior_cov)

    def smc(self):
        """
        Using PyMC3 for Monte Carlo Sampling
        :return:
        """
        budget = self.options['smc_budget']
        kl, kl_pd = None, None
        # first_moment = np.zeros((budget, self.dimensions))
        # second_moment = np.zeros((budget, self.dimensions, self.dimensions))

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

            # Initialise the start point of the Monte Carlo sampling to the MAP estimate
            start = pm.find_MAP(model=basic_model)
            trace = pm.sample(budget, start=start)

        # Post process the samples obtained from the MCMC sampler
        samples = trace.get_values('like')
        if self.gpr.grd_evidence is not None:
            grd_mean, grd_cov = self.gpr.grd_posterior_gaussian()
            kl = np.empty((len(samples), ))
            kl[:] = np.nan

            for i in range(len(samples)): # 4 number of chains in the MCMC sampler
                if i % 100 == 0:
                    first_moment = moment(samples[:i+1, :], moment=1)
                    second_moment = moment(samples[:i+1, :], moment=2)

                    kl[i] = np.log(self.gauss_sym_kl_div(grd_mean, grd_cov, first_moment, np.diag(second_moment)))
            kl_pd = pd.Series(kl, name='kl_div')

        # Package the numpy array to pandas DataFrame for subsequent saving to local storage
        prior_pd = pd.DataFrame(trace.get_values('x'), columns=['p_' + str(i) for i in range(self.dimensions)])
        samples_pd = pd.DataFrame(trace.get_values('like'), columns=['x_' + str(i) for i in range(self.dimensions)])
        if kl is None:
            self.save_result(prior_pd, samples_pd, file_name='SMCPosteriorSamples')
        else:
            self.save_result(prior_pd, samples_pd, kl_pd, file_name='SMCPosteriorSamples')

        self.plot_iterations_mc(budget, samples, kl_div=kl)

    def bq(self):
        pass

    def wsabi(self, same_query_pts=True):
        budget = self.options['wsabi_bq_budget']

        samples = np.zeros((budget, self.gpr.dimensions))  # Array to store all the x locations of samples
        lik = np.zeros((budget,))  # Array to store all the log-likelihoods evaluated at x
        intv = np.zeros((budget, ))

        # Initial points
        initial_x = np.zeros((self.dimensions, 1)).reshape(1, -1)  # Set the initial sample to the prior mean
        initial_y = np.array(self.gpr.sample(initial_x)).reshape(1, -1)

        # Setting up kernel
        kern = GPy.kern.RBF(self.dimensions, variance=self.options['naive_bq_kern_variance'],
                            lengthscale=self.options['naive_bq_kern_lengthscale'])

        # Initial guess for the GP for BQ
        gpy_gp_lik = GPy.models.GPRegression(initial_x, initial_y, kernel=kern, )
        warped_gp = WsabiLGP(gpy_gp_lik)
        model = WarpedIntegrandModel(warped_gp, prior=self.prior)

        for i in range(budget):
            samples[i, :] = np.array(select_batch(model, 1, 'Local Penalisation')).reshape(1, -1)
            lik[i] = np.array(self.gpr.sample(samples[i, :])).reshape(1, -1)
            model.update(samples[i, :], lik[i])
            gpy_gp_lik.optimize()

        intv = model.integral_mean()[0]
        print("Integral mean estimated:", intv)

        # Generate query points meshgrid for the posterior distribution and the priors evaluated on these points
        if same_query_pts:
            query_points = np.empty((budget+1, self.gpr.dimensions))
            query_points[0] = initial_x
            query_points[1:] = samples
            prior_query_points = self.prior(query_points)
            unwarped_y = np.squeeze(warped_gp._unwarped_Y)
        else:
            query_points, prior_query_points = self._gen_meshgrid_query_points()
            # Evaluate at the query points on the likelihood surface generated by the GP surrogate
            warped_lik = gpy_gp_lik.predict(query_points)[0]
            # Unwarp the warped likelihood outputs
            unwarped_y = self.unwarp(warped_lik, warped_gp._alpha)
        posterior_query_points = (unwarped_y * prior_query_points / intv).reshape(-1, 1)

        # Initialise another GP for the posterior distribution
        gpy_gp_post = GPy.models.GPRegression(query_points, posterior_query_points, kernel=kern, )
        gpy_gp_post.optimize()

        #plt.subplot(211)
        #gpy_gp_lik.plot()
        #plt.subplot(212)
        gpy_gp_post.plot(plot_limits=[[-5, -5], [5, 5]])
        plt.show()

        # Package arrays and models for local storage
        # samples_pd = pd.DataFrame(samples, columns=['samples'+str(i) for i in range(self.dimensions)])
        # lik_pd = pd.Series(lik, name='likelihood')
        # posterior_pd = pd.Series(poste, name='posterior')

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

    def get_posterior_distribution(self, gpy_gp_lik, evidence: float):
        """
        Fit a posterior_gp from the likelihood surface obtained. This is used to generate the posterior over parameter
        distribution.
        :param gpy_gp_lik: The trained GP model for the likelihood surface
        :param evidence: The estimated evidence (marginal likelihood)
        :return:
        """
        pass

    @staticmethod
    def unwarp(warped_y: np.ndarray, alpha: float):
        """
        Unwarp the output of a WSABI GP
        :param warped_y:
        :return:
        """
        return 0.5 * (warped_y ** 2) + alpha

    # --- Utility Functions ---#
    def _gen_meshgrid_query_points(self):
        if self.dimensions == 2:
            query_points = np.mgrid[-5:5:0.1, -5:5:0.1].T.reshape(-1, 2)
            prior_evals = self.prior(query_points)
            return query_points, prior_evals
        else:
            raise NotImplemented()

    def _unpack_options(self,
                        prior_mean: Union[float, np.ndarray] = 0.,
                        prior_variance: Union[float, np.ndarray] = 2.,
                        smc_budget: int = 10000,
                        naive_bq_budget: int = 1000,
                        naive_bq_kern_lengthscale: float = 1.,
                        naive_bq_kern_variance: float = 1.,
                        wsabi_bq_budget: int = 200,
                        posterior_range = (-5., 5.),
                        # The axis ranges between which the posterior samples are drawn
                        posterior_eps: float = 0.02,
                        # Number of points to draw along each dimensions in psoterior sampling
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
            assert len(posterior_range) == 2
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

    def save_result(self, *x: Union[pd.Series, pd.DataFrame], file_name):
        lens = len(set(len(each_x) for each_x in x))
        assert lens == 1, "The lengths of the dataframes/series in the input argument do not match!"
        save = pd.concat(x, axis=1)
        save.to_csv('~/Dropbox/4YP/Codes/wsabi_ratio/output/'+file_name+'.csv')
        print("Save result successfully!")
        return 0



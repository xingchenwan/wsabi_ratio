# Yacht data manipulation
# Xingchen Wan | Dec 2018 | xingchen.wan@st-annes.ox.ac.uk
# Python 3.7


import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
# This is to prevent a macOS bug with matplotlib
import matplotlib.pyplot as plt
import GPy
from IPython.display import display
from typing import Union
from bayesquad.priors import Gaussian
from bayesquad.quadrature import OriginalIntegrandModel, WarpedIntegrandModel
from bayesquad.batch_selection import select_batch, LOCAL_PENALISATION
from bayesquad.gps import GP, WsabiLGP
from ratio_extension.yacht_integral import integral_mean_rebased, integral_mean_without_rebase
from yacht.functions import Functions, GPRegressionFromFile, Rosenbrook2D


class GPLikelihood:
    """
    Likelihood Computation of a GP Regression
    """
    def __init__(self, regression_model: Functions, **kwargs):
        self.gpr = regression_model
        self.options = self._unpack_options(**kwargs)
        self.smc_samples = None
        self.naive_bq_samples = None
        self.wsabi_samples = None
        self.dimensions = regression_model.dimensions

    # ---------------- Compute the maximum likelihood estimate of the hyper-parameters ---------- #
    def maximum_likelihood(self):
        """
        Perform optimization to obtain the MLE parameters of the model
        :return: The list collecting all hyperparmeters of the optimized model
        """
        if self.gpr.model is None:
            raise TypeError("GPy model is not found!")
        prior_model = self.gpr.model.copy()
        prior_model.kern.variance.set_prior(GPy.priors.LogGaussian(mu=0., sigma=2.))
        prior_model.kern.lengthscale.set_prior(GPy.priors.LogGaussian(mu=0., sigma=2.))
        self.gpr.model.optimize(messages=True, max_iters=self.options['max_optimisation_iterations'])
        if self.options['max_optimisation_restart'] > 1:
            self.gpr.model.optimize_restarts(num_restarts=self.options['max_optimisation_restart'])
        res = self.gpr._collect_params()
        print("Optimised parameters:")
        display(self.gpr.model)
        return res

    # ---------------- Compute the marginal likelihood marginalised by the hyperparameters \theta ------- #
    def smc(self) -> tuple:
        """
        Compute the log-evidence marginalised by hyper-parameters by using exhaustive simple Monte Carlo sampling
        :return: Computed evidence, computed log-evidence
        """
        prior_mean = self.options['prior_mean'].reshape(-1)
        prior_cov = self.options['prior_variance']
        budget = self.options['smc_budget']

        samples = np.zeros((budget, self.gpr.dimensions))
        yv = np.zeros((budget, ))
        intv = np.zeros((budget, ))
        log_intv = np.zeros((budget, ))

        for i in range(budget):
            # Draw a sample from the prior distribution
            # samples[i, :] = np.exp(np.random.multivariate_normal(mean=prior_mean, cov=prior_cov))
            samples[i, :] = np.random.multivariate_normal(mean=prior_mean, cov=prior_cov)

            # Evaluate the sample query point on the likelihood function
            yv[i] = self.gpr.log_sample(samples[i, :])
            mc_max = np.max(yv[:i+1])
            intv[i] = np.mean(np.exp(yv[:i+1] - mc_max))
            log_intv[i] = np.log(intv[i]) + mc_max

            # if i % 100 == 0:
            #   self.plot_iterations(i, log_intv, true_val=self.gpr.grd_log_evidence)

        if self.gpr.grd_log_evidence is not None:
            rmse = np.sqrt((intv - np.exp(self.gpr.grd_log_evidence)) ** 2)
            self.save_results('SMCSampling', y=yv, intv=intv, log_intv=log_intv, rmse=rmse)
        else:
            self.save_results('SMCSampling', y=yv, intv=intv, log_intv=log_intv)

        return yv[-1], log_intv[-1]

    def bmc(self):
        """
        Bayesian Monte Carlo - No Active Sampling
        :return:
        """
        prior_mean = self.options['prior_mean'].reshape(-1)
        prior_cov = self.options['prior_variance']
        prior = Gaussian(mean=prior_mean, covariance=prior_cov)

        budget = self.options['naive_bq_budget']

        samples = np.zeros((budget, self.gpr.dimensions))
        yv = np.zeros((budget,))
        yv_scaled = np.zeros((budget, ))
        intv = np.zeros((budget, ))
        log_intv = np.zeros((budget,))

        for i in range(budget):

            samples[i, :] = np.random.multivariate_normal(mean=np.array([0, 0]), cov=2*np.eye(2))
            yv[i] = np.array(self.gpr.log_sample(samples[i, :])).reshape(1, -1)
            scaling = np.max(yv[:i+1])
            print(yv[:i+1])
            yv_scaled[:i+1] = np.exp(yv[:i+1] - scaling)
            this_x = samples[:i+1]
            this_y = yv_scaled[:i+1]

            if i == 0:
                kern = GPy.kern.RBF(input_dim=self.dimensions,
                                    variance=self.options['naive_bq_kern_variance'],
                                    lengthscale=self.options['naive_bq_kern_lengthscale'])
                gpy_gp = GPy.models.GPRegression(this_x, this_y.reshape(1, -1), kernel=kern)
                gp = GP(gpy_gp)
                model = OriginalIntegrandModel(gp=gp, prior=prior)
            else:
                # Note that due to the re-scaling to max, at each step the entire x and y values are *replaced*
                # rather than updated.
                # print(this_x, this_y)
                model.replace(this_x, this_y)
                gpy_gp.optimize()
                display(gpy_gp)
                log_intv[i] = np.log((model.integral_mean())[0]) + scaling
                intv[i] = np.exp(log_intv[i])

            if i % 50 == 0:
                print(i, log_intv[i])
                self.plot_iterations(i, log_intv, true_val=self.gpr.grd_log_evidence)

        return log_intv

    def naive_bq(self) -> tuple:
        """
        Marginalise the marginal log-likelihood using naive Bayesian Quadrature
        :return:
        """
        budget = self.options['naive_bq_budget']

        naive_bq_samples = np.zeros((budget, self.gpr.dimensions))  # Array to store all the x locations of samples
        naive_bq_log_y = np.zeros((budget, ))  # Array to store all the log-likelihoods evaluated at x
        log_naive_bq_int = np.zeros((budget, ))  # Array to store the current estimate of the marginalised integral

        # Initial points
        initial_x = np.zeros((self.dimensions, 1)).reshape(1, -1)+1e-6  # Set the initial sample to the prior mean
        initial_y = np.array(self.gpr.sample(initial_x)).reshape(1, -1)

        # Prior in log space
        prior_mean = self.options['prior_mean'].reshape(-1)
        prior_cov = self.options['prior_variance']
        prior = Gaussian(mean=prior_mean, covariance=prior_cov)

        # Setting up kernel - noting the log-transformation
        kern = GPy.kern.RBF(self.dimensions, variance=self.options['naive_bq_kern_variance'],
                            lengthscale=self.options['naive_bq_kern_lengthscale'])

        # Initial guess for the GP for BQ
        gpy_gp = GPy.models.GPRegression(initial_x, initial_y, kernel=kern,)
        gp = GP(gpy_gp)
        model = OriginalIntegrandModel(gp=gp, prior=prior)
        for i in range(1, self.options['naive_bq_budget']):
            # Do active sampling
            this_x = np.array(select_batch(model, 1, "Kriging Believer")).reshape(1, -1)
            this_y = np.array(self.gpr.sample(this_x)).reshape(1, -1)
            print(this_x, this_y)

            naive_bq_samples[i, :] = this_x
            naive_bq_log_y[i] = this_y

            model.update(this_x, this_y)

            gpy_gp.optimize()
            log_naive_bq_int[i] = (model.integral_mean())[0]

            if i % 10 == 0:
                self.plot_iterations(i, naive_bq_samples, naive_bq_log_y)

                print("Step", str(i))
                print("Current estimate of Log-evidence: ", log_naive_bq_int[i])
                # print("Current values of hyperparameters: ", display(gpy_gp))
                plt.plot(log_naive_bq_int[:i], "*")
                plt.show()
        self.naive_bq_samples = naive_bq_samples
        return naive_bq_log_y[-1], log_naive_bq_int[-1]

    def wsabi_bq(self):
        """
        Marginalise the marginal log-likelihood using WSABI Bayesian Quadrature
        :return:
        """
        budget = self.options['naive_bq_budget']

        samples = np.zeros((budget, self.gpr.dimensions))  # Array to store all the x locations of samples
        yv = np.zeros((budget,))  # Array to store all the log-likelihoods evaluated at x
        intv = np.zeros((budget,))  # Array to store the current estimate of the marginalised integral

        # Initial points
        initial_x = np.zeros((self.dimensions, 1)).reshape(1, -1) + 1e-6  # Set the initial sample to the prior mean
        initial_y = np.array(self.gpr.sample(initial_x)).reshape(1, -1)

        # Prior in log space
        prior_mean = self.options['prior_mean'].reshape(-1)
        prior_cov = self.options['prior_variance']
        prior = Gaussian(mean=prior_mean, covariance=prior_cov)

        # Setting up kernel - noting the log-transformation
        kern = GPy.kern.RBF(self.dimensions, variance=self.options['naive_bq_kern_variance'],
                            lengthscale=self.options['naive_bq_kern_lengthscale'])

        # Initial guess for the GP for BQ
        gpy_gp = GPy.models.GPRegression(initial_x, initial_y, kernel=kern, )
        warped_gp = WsabiLGP(gpy_gp)
        model = WarpedIntegrandModel(warped_gp, prior=prior)
        for i in range(self.options['naive_bq_budget']):
            # Do active sampling
            this_x = np.array(select_batch(model, 1, LOCAL_PENALISATION)).reshape(1, -1)
            this_y = np.array(self.gpr.sample(this_x)).reshape(1, -1)
            print(this_x)
            samples[i, :] = this_x
            yv[i] = this_y
            model.update(this_x, this_y)
            gpy_gp.optimize()
            intv[i] = (model.integral_mean())[0]

            if i % 100 == 0:
                self.plot_iterations(i, samples, yv)
                print("Step", str(i))
                print("Current estimate of Log-evidence: ", intv[i])
                # print("Current values of hyperparameters: ", display(gpy_gp))
                plt.plot(intv[:i], "*")
                plt.show()
        self.naive_bq_samples = samples
        return yv[-1], intv[-1]

    # ----------------------- Utility function for keyword arguments -------------------------- #
    def _unpack_options(self, kernel_option: str ='rbf',
                        max_optimisation_iterations: int = 1000,
                        max_optimisation_restart: int = 20,
                        prior_mean: Union[float, np.ndarray] = 0.,
                        prior_variance: Union[float, np.ndarray] = 2.,
                        smc_budget: int = 1000,
                        naive_bq_budget: int = 1000,
                        naive_bq_kern_lengthscale: float = 1.,
                        naive_bq_kern_variance: float = 1.,
                        wsabi_bq_budget: int = 1000,
                        ) -> dict:
        """
        Unpack kwargs
        :param kernel_option: str: the name of the kernel specficied. Currently only the rbf is allowed
        :param max_optimisation_iterations: maximum evaluations of the likelihood functions for MLE optimisation
        :param max_optimisation_restart: number of restarts of the MLE optimisation to avoid the likelihood function
        being trapped in local minima
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
            'kernel_option': kernel_option,
            'max_optimisation_iterations': max_optimisation_iterations,
            'max_optimisation_restart': max_optimisation_restart,
            'prior_mean': prior_mean,
            'prior_variance': prior_variance,
            'smc_budget': smc_budget,
            'naive_bq_budget': naive_bq_budget,
            'naive_bq_kern_lengthscale': naive_bq_kern_lengthscale,
            'naive_bq_kern_variance': naive_bq_kern_variance,
            'wsabi_bq_budget': wsabi_bq_budget,
        }

    # ---------------------------- Utility functions ------------------------ #
    @staticmethod
    def save_results(file_name, **n_arrays: np.ndarray):
        """
        Package the result into a pandas data frame and save it on the local storage for later use
        :param n_arrays: dictionary of np.arrays. The keywords will be used as column headers in the pandas dataframe
        :return:
        """
        if len(n_arrays) == 0:
            raise ValueError("No array is supplied!")
        n_s = [i.shape[0] for i in n_arrays.values()]
        assert len(set(n_s)) == 1, "arrays in the n_arrays argument must be of the same first dimension!"
        df = pd.DataFrame(np.stack(n_arrays.values(), axis=-1), columns=n_arrays.keys())
        df.to_csv('output/'+file_name+'.csv')
        print("Successfully saved ", file_name)
        return 0

    @staticmethod
    def plot_iterations(i, log_lik, true_val=None):
        if i == 0:
            return
        xv = np.arange(0, i, 1)
        plt.subplot(121)
        plt.plot(xv, log_lik[:i], ".")
        if true_val is not None:
            plt.axhline(true_val)
        plt.xlabel("Number of samples")
        plt.ylabel("E(X)")

        plt.subplot(122)
        if true_val is not None:
            # rmse = np.sqrt((log_lik[:i] - true_val) ** 2)
            rmse = np.sqrt((np.exp(log_lik[:i]) - np.exp(true_val)) ** 2)

            plt.plot(xv, rmse, '.')
            plt.xlabel("Number of samples")
            plt.ylabel("RMSE")
        plt.show()


# For testing purposes only
if __name__ == '__main__':
    pr = Gaussian(mean=np.array([0, 0]), covariance=np.array([[2, 0],[0, 2]]))
    rb = Rosenbrook2D(prior=pr)

    #rb.plot_grd_posterior()

    lik = GPLikelihood(rb)
    lik.bmc()

    exit()
    gpr = GPRegressionFromFile()
    lik = GPLikelihood(gpr)
    # lik.maximum_likelihood()
    lik.bmc()

    lik.smc()




# Yacht data manipulation
# Xingchen Wan | Dec 2018 | xingchen.wan@st-annes.ox.ac.uk
# Python 3.7


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import GPy
from IPython.display import display
from typing import Union
from bayesquad.priors import Gaussian
from bayesquad.quadrature import OriginalIntegrandModel
from bayesquad.batch_selection import select_batch, LOCAL_PENALISATION
from bayesquad.gps import GP


# Some global settings
file_path = "data/yacht_hydrodynamics.data.txt"
col_headers = ['Longitudinal position of buoyancy centre',
               'Prismatic Coefficient',
               'Length-displacement Ratio',
               'Beam-draught Ratio',
               'Length-beam Ratio',
               'Froude Number',
               'Residuary Resistance Per Unit Weight of Displacement']
kernel = 'rbf'


class GPRegression:

    def __init__(self):
        self.X, self.Y = self.load_data()
        self.dimensions = self.X.shape[1]
        self.kernel_option = kernel
        self.model = self.init_gp_model()

    @staticmethod
    def load_data(plot_graph=False):
        raw_data = pd.read_csv(filepath_or_buffer=file_path, header=None, sep='\s+')
        if plot_graph:
            num_cols = len(raw_data.columns)
            for i in range(num_cols):
                plt.subplot(4, 2, i+1)
                plt.plot(raw_data.index, raw_data.loc[:, i])
                plt.title(col_headers[i])
            plt.show()
        data_X = raw_data.iloc[:, :-1].values
        data_Y = raw_data.iloc[:, -1].values

        # Refactor to 2d array
        if data_Y.ndim == 1:
            data_Y = data_Y.reshape(-1, 1)
        if data_X.ndim == 1:
            data_X = np.array([data_X])
        assert data_X.shape[0] == data_Y.shape[0]
        return data_X, data_Y

    def init_gp_model(self):

        init_len_scale = np.array([1.]*self.X.shape[1])
        init_var = 1.

        if self.kernel_option == 'rbf':
            ker = GPy.kern.RBF(input_dim=self.X.shape[1],
                               lengthscale=init_len_scale,
                               variance=init_var,
                               ARD=True,)
        else:
            raise NotImplementedError()
        m = GPy.models.GPRegression(self.X, self.Y, ker)
        return m

    # ------------ Compute the marginal log-likelihood of the model -------- #
    def log_sample(self, x: Union[np.ndarray, float, list]) -> float:
        """
        Compute the log-likelihood of the given parameter array x.
        :param x: List/array of parameters. The first parameter corresponds to the model variance. The second - penul-
        timate items correspond to the model lengthscale in each dimension. The last item corresponds to the Gaussian
        noise parameter
        The length of the parameter array must be exactly 2 more than the dimensionality of the data
        :return: the log-likelihood of the model evaluated.
        """

        # Transform to exponentiated space
        x = np.asarray(np.exp(x)).reshape(-1)
        # display(self.model)
        assert len(x) == self.dimensions + 2
        # 2 extra dimensions to accommodate the Gaussian noise and model variance parameter of the RBF kernel
        self.model.rbf.variance = x[0]
        self.model.rbf.lengthscale = x[1:-1]
        self.model.Gaussian_noise.variance = x[-1]
        return self.model.log_likelihood()

    def sample(self, x: Union[np.ndarray, float, list]) -> float:
        """
        Convert the log-likelihood back to likelihood
        :param x: List/array of parameters
        :return: the likelihood of the model evaluated
        """
        x = np.asarray(np.exp(x)).reshape(-1)
        return np.exp(-self.log_sample(x))

    def _collect_params(self) -> np.ndarray:
        """
        Collect the parameter values into a numpy n-d array
        :return: condensed numpy array of params
        """
        res = np.array([0.]*(self.dimensions+2))
        res[0] = self.model.rbf.variance
        res[1:-1] = self.model.rbf.lengthscale
        res[-1] = self.model.Gaussian_noise.variance
        # This is the parameter we are interested in for the Diego paper
        return res


class GPLikelihood:
    """
    Likelihood Computation of a GP Regression
    """
    def __init__(self, regression_model: GPRegression, **kwargs):
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
        self.gpr.model.optimize(messages=True, max_iters=self.options['max_optimisation_iterations'])
        if self.options['max_optimisation_restart'] > 1:
            self.gpr.model.optimize_restarts(num_restarts=self.options['max_optimisation_restart'])
        res = self.gpr._collect_params()
        print("Optimised parameters:")
        display(self.gpr.model)
        return res

    # ---------------- Compute the marginal likelihood marginalised by the hyperparameters \theta ------- #
    def smc(self, display_noise=False) -> tuple:
        """
        Compute the log-evidence marginalised by hyper-parameters by using exhaustive simple Monte Carlo sampling
        :return: Computed evidence, computed log-evidence
        """
        prior_mean = self.options['prior_mean'].reshape(-1)
        prior_cov = self.options['prior_variance']
        budget = self.options['smc_budget']

        mc_samples = np.zeros((budget, self.gpr.dimensions+2))
        mc_int = np.zeros((budget, ))
        mc_out = np.zeros((budget, ))
        log_mc_int = np.zeros((budget, ))

        for i in range(budget):
            # Draw a sample from the prior distribution
            mc_samples[i, :] = np.exp(np.random.multivariate_normal(mean=prior_mean, cov=prior_cov))
            # Evaluate the sample query point on the likelihood function
            mc_out[i] = self.gpr.log_sample(mc_samples[i, :])
            mc_max = np.max(mc_out[:i+1])
            mc_int[i] = np.mean(np.exp(mc_out[:i+1] - mc_max))
            log_mc_int[i] = np.log(mc_int[i]) + mc_max

            if i % 10 == 0:
                self.plot_iterations(i, mc_samples, mc_out)
                print("Step", str(i))
                if display_noise:
                    pass
                else:
                    print('samples', mc_samples[i, :])
                    print("Current estimate of Log-evidence: ", log_mc_int[i])
                    plt.plot(log_mc_int[:i])
                plt.show()
        self.smc_samples = log_mc_int
        return mc_int[-1], log_mc_int[-1]

    def ais(self):
        """
        Annealed Importance Sampling
        :return:
        """
        pass

    def naive_bq(self) -> tuple:
        """
        Marginalise the marginal log-likelihood using naive Bayesian Quadrature
        :return:
        """
        budget = self.options['naive_bq_budget']

        naive_bq_samples = np.zeros((budget, self.gpr.dimensions+2))  # Array to store all the x locations of samples
        naive_bq_log_y = np.zeros((budget, ))  # Array to store all the log-likelihoods evaluated at x
        naive_bq_y = np.zeros((budget, ))  # Array to store the likelihoods evaluated at x
        log_naive_bq_int = np.zeros((budget, ))  # Array to store the current estimate of the marginalised integral

        # Initial points
        initial_x = np.zeros((self.dimensions+2, 1)).reshape(1, -1)+1e-6  # Set the initial sample to the prior mean
        initial_y = np.array(self.gpr.log_sample(initial_x)).reshape(1, -1)

        # Prior in log space
        prior_mean = self.options['prior_mean'].reshape(1, -1)
        prior_cov = self.options['prior_variance']

        # Setting up kernel - noting the log-transformation
        kern = GPy.kern.RBF(self.dimensions+2, variance=np.log(self.options['naive_bq_kern_variance']),
                            lengthscale=np.log(self.options['naive_bq_kern_lengthscale']))

        # Initial guess for the GP for BQ
        lik = GPy.likelihoods.Gaussian(variance=1e-10)
        prior = Gaussian(mean=prior_mean.reshape(-1), covariance=prior_cov)
        gpy_gp = GPy.core.GP(initial_x, initial_y, kernel=kern, likelihood=lik)
        gp = GP(gpy_gp)
        model = OriginalIntegrandModel(gp=gp, prior=prior)
        for i in range(1, self.options['naive_bq_budget']):
            # Do active sampling
            this_x = np.array(select_batch(model, 1, LOCAL_PENALISATION)).reshape(1, -1)
            naive_bq_samples[i, :] = this_x
            naive_bq_log_y[i] = np.array(self.gpr.log_sample(this_x)).reshape(1, -1)

            # Compute the scaling
            log_scaling = np.max(naive_bq_log_y[:i])

            # Scaling batch by max and exponentiate
            naive_bq_y[:i] = np.exp(naive_bq_log_y[:i] - log_scaling)
            this_y = naive_bq_y[i]

            model.update(this_x, this_y)
            gpy_gp.optimize()
            naive_bq_int, _, _= model.integral_mean(log_transform=True)
            log_naive_bq_int[i] = naive_bq_int + log_scaling
            print("samples", np.exp(this_x))
            print("eval", log_naive_bq_int[i])
            if i % 1 == 0:
                self.plot_iterations(i, naive_bq_samples, naive_bq_log_y)
                print("Step", str(i))
                print("Current estimate of Log-evidence: ", log_naive_bq_int[i])
                # print("Current values of hyperparameters: ", display(gpy_gp))
                plt.plot(log_naive_bq_int[:i])
                plt.show()
        self.naive_bq_samples = naive_bq_samples
        return naive_bq_log_y[-1], log_naive_bq_int[-1]

    def wsabi_bq(self):
        """
        Marginalise the marginal log-likelihood using WSABI Bayesian Quadrature
        :return:
        """
        pass

    # ----------------------- Utility function for keyword arguments -------------------------- #
    def _unpack_options(self, kernel_option: str ='rbf',
                        max_optimisation_iterations: int = 1000,
                        max_optimisation_restart: int = 20,
                        prior_mean: Union[float, np.ndarray] = 0.,
                        prior_variance: Union[float, np.ndarray] = 2.,
                        smc_budget: int = 100000,
                        naive_bq_budget: int = 1000,
                        naive_bq_kern_lengthscale: float = 2.,
                        naive_bq_kern_variance: float = 2.,
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
            prior_mean = np.array([prior_mean]*(self.gpr.dimensions+2)).reshape(-1, 1)
            prior_variance *= np.eye(self.gpr.dimensions+2)
        else:
            assert len(prior_mean) == self.gpr.dimensions + 2
            assert prior_variance.shape[0] == prior_variance.shape[1]
            assert prior_variance.shape[0] == self.gpr.dimensions + 2
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
    def save_results(self):
        pass

    def plot_iterations(self, i, samples, log_lik, log_lik_int=None, noise_only=False):
        if i == 0:
            return
        if noise_only:
            if i > 20:
                plt.plot(samples[:i-20, -1], log_lik[:i - 20], "x", 'gray')
                plt.plot(samples[i-20:i, -1], log_lik[:i - 20], "x", "red")
            else:
                plt.plot(samples[:i, -1], log_lik[:i], "x", 'red')
                plt.title(col_headers[-1])
        else:
            for j in range(0, self.dimensions+1):
                plt.subplot(2, 5, j+1)
                if i > 20:
                    plt.plot(samples[:i-20, j], log_lik[:i - 20], "x", 'gray')
                    plt.plot(samples[i-20:i, j], log_lik[:i - 20], "x", "red")
                else:
                    plt.plot(samples[:i, j], log_lik[:i], "x", 'red')
                    plt.title(col_headers[j])
            if log_lik_int:
                plt.subplot(2, 5, self.dimensions+1)
                plt.plot(log_lik_int[:i])
        plt.show()


class GPPosterior:
    pass


# For testing purposes only
if __name__ == '__main__':
    gpr = GPRegression()
    lik = GPLikelihood(gpr)
    lik.naive_bq()

    lik.smc()


import numpy as np
from scipy.stats import norm, multivariate_normal
import matplotlib.pyplot as plt
from typing import Union


class GaussMixture:
    """
    A test function consists of a mixture (summation) of Gaussians (so used because it allows the evaluation of the integration
    exactly as a benchmark for other quadrature methods.
    """
    def __init__(self, means: np.ndarray, covs: np.ndarray,):
        assert means.shape[0] == covs.shape[0], "Mean and Covariance List mismatch!"
        assert means.ndim == 2
        self.means = means
        self.covs = covs
        self.dimensions = means.shape[1]
        self.mixture_count = len(means)

    def sample(self, x, ):
        y = 0
        for i in range(self.mixture_count):
            if self.dimensions == 1:
                y += self.one_d_normal(x, self.means[i], self.covs[i])
            else:
                y += self.multi_d_gauss(x, self.means[i], self.covs[i])
        return y

    @staticmethod
    def one_d_normal(x, mean, var):
        return norm.pdf(x, mean, var)

    @staticmethod
    def multi_d_gauss(x, mean, cov):
        return multivariate_normal.pdf(x, mean=mean, cov=cov)

    def plot(self, plot_range: list):
        range_min, range_step, range_max = plot_range[0], plot_range[1], plot_range[2]
        plot_x = np.arange(range_min, range_max, range_step + 0.0)
        plot_y = np.array([self.sample(x) for x in plot_x])
        plt.plot(plot_x, plot_y)

    def add_gaussian(self, means: Union[np.ndarray, float], var: Union[np.ndarray, float]):
        assert means.shape == self.means.shape[1:]
        assert var.shape == self.covs.shape[1:]
        self.means = np.append(self.means, means)
        self.covs = np.append(self.covs, var)


def evidence_integral(gauss_mix: GaussMixture, prior_mean: np.ndarray, prior_var: np.ndarray):
    """
    Returns the analytical integral of the evidence integral given by the mixture (sum) of Gaussian multiplied
    by a Gaussian prior.
    :param gauss_mix: an instance of Gaussian Mixture defined in the GaussMixture instance
    :param prior_mean: :param prior_var: mean and covariance of the prior (Gaussian) distribution
    :return: the *exact* integral of the gaussian mixture by analytical methods
    """
    assert prior_mean.shape[0] == gauss_mix.dimensions
    res = 0.
    for i in range(gauss_mix.mixture_count):
        mu = gauss_mix.means[i]
        sigma = gauss_mix.covs[i]
        if gauss_mix.dimensions > 1:
            res += (2 * np.pi)**(-gauss_mix.dimensions/2) * np.linalg.det(prior_var + sigma) ** -0.5 * \
                np.exp(-0.5 * np.transpose(mu - prior_mean) * np.linalg.inv(sigma + prior_var) * (mu - prior_mean))
            # The integral of the resulting un-normalised Gaussian over the entire real number domain is given by
            # the normalising factor of the Gaussian.
        else:
            res += 1 / np.sqrt(2 * np.pi * (sigma + prior_var)) * np.exp(- ((mu - prior_mean) ** 2) /
                                                                            (2 * (sigma + prior_var)))
            # Ditto for the uni-variate case
    return res


def predictive_integral(gauss_mix_1: GaussMixture, gauss_mix_2: GaussMixture,
                        prior_mean: np.ndarray, prior_var: np.ndarray):
    pass
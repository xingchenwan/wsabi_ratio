# Xingchen Wan | 2018
# Generation of the test functions that are made from Gaussian mixture for evaluation of the quality of integration
# of various numerical integration techniques. Due to the use of Gaussian mixtures these integrals are analytically
# tractable; these exact values form the benchmark of comparison.


import numpy as np
from scipy.stats import norm, multivariate_normal
import matplotlib.pyplot as plt
from typing import Union


class TrueFunctions:
    def __init__(self, ):
        self.dimensions = None

    def sample(self, x: Union[np.ndarray, float, list]):
        x = np.asarray(x)
        if x.ndim == 0 or x.ndim == 1:
            assert self.dimensions == 1, "Scalar input is only permitted if the function is of dimension!"
        else:
            assert x.ndim == 2
            assert x.shape[1] == self.dimensions, "Dimension of function is of"+str(self.dimensions)+\
                                                  " ,but the dimension of input is "+str(x.shape[1])


class GaussMixture(TrueFunctions):
    """
    A test function consists of a mixture (summation) of Gaussians (so used because it allows the evaluation of
    the integration exactly as a benchmark for other quadrature methods.
    """
    def __init__(self, means: Union[np.ndarray, float, list], covs: Union[np.ndarray, float, list],):
        super(GaussMixture, self).__init__()
        self.means = np.asarray(means)
        self.covs = np.asarray(covs)
        assert self.means.shape[0] == self.covs.shape[0], "Mean and Covariance List mismatch!"
        if self.means.ndim == 1:
            self.dimensions = 1
        else:
            self.dimensions = self.means.shape[1]
        self.mixture_count = len(self.means)

    def sample(self, x: Union[np.ndarray, float, list], ):
        """
        Sample from the true function either with one query point or a list of points
        :param x: the coordinate(s) of the query point(s)
        :return: the value of the true function evaluated at the query point(s)
        """
        x = np.asarray(x)
        if x.ndim <= 1:
            y = 0
            for i in range(self.mixture_count):
                if self.dimensions == 1:
                    y += self.one_d_normal(x, self.means[i], self.covs[i])
                else:
                    y += self.multi_d_gauss(x, self.means[i], self.covs[i])
        else:
            y = np.zeros((x.shape[0], ))
            for j in range(x.shape[0]):
                for i in range(self.mixture_count):
                    if self.dimensions == 1:
                        y[j] += self.one_d_normal(x, self.means[i], self.covs[i])
                    else:
                        y[j] += self.multi_d_gauss(x, self.means[i], self.covs[i])
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


def evidence_integral(gauss_mix: GaussMixture,
                      prior_mean: Union[np.ndarray,float], prior_var: Union[np.ndarray, float]):
    """
    Returns the analytical integral of the evidence integral given by the mixture (sum) of Gaussian multiplied
    by a Gaussian prior.
    :param gauss_mix: an instance of Gaussian Mixture defined in the GaussMixture instance
    :param prior_mean: :param prior_var: mean and covariance of the prior (Gaussian) distribution
    :return: the *exact* integral of the gaussian mixture by analytical methods
    """
    prior_var = np.asarray(prior_var)
    prior_mean = np.asarray(prior_mean)
    if prior_mean.ndim == 0:
        assert gauss_mix.dimensions == 1
    else:
        assert prior_mean.shape[0] == gauss_mix.dimensions
    res = 0.
    for i in range(gauss_mix.mixture_count):
        mu = gauss_mix.means[i]
        sigma = gauss_mix.covs[i]
        _, _, scaling_factor = gauss_product(mu, prior_mean, sigma, prior_var)
        res += scaling_factor
    return res


def predictive_integral(gauss_mix_1: GaussMixture, gauss_mix_2: GaussMixture,
                        prior_mean: Union[np.ndarray, float], prior_var: Union[np.ndarray, float]):
    """
    Return the analytical integral of the posterior integral given by the two mixture of Guassians multipled by a
    Gaussian prior.
    :param gauss_mix_1:
    :param gauss_mix_2:
    :param prior_mean:
    :param prior_var:
    :return:
    """
    prior_mean = np.asarray(prior_mean)
    prior_var = np.asarray(prior_var)
    if prior_mean.ndim == 0:
        # Supplied argument is number
        assert gauss_mix_1.dimensions == 1 and gauss_mix_2.dimensions == 1
    else:
        assert prior_mean.shape[0] == gauss_mix_1.dimensions
        assert gauss_mix_1.dimensions == gauss_mix_2.dimensions
    res = 0.
    for i in range(gauss_mix_1.mixture_count):
        mu1 = gauss_mix_1.means[i]
        sigma1 = gauss_mix_1.covs[i]
        for j in range(gauss_mix_2.mixture_count):
            mu2 = gauss_mix_2.means[j]
            sigma2 = gauss_mix_2.covs[j]
            mu_product, sigma_product, scale_product = gauss_product(mu1, mu2, sigma1, sigma2)
            _, _, scale_with_prior = gauss_product(mu_product, prior_mean, sigma_product, prior_var)
            res += scale_with_prior * scale_product
    return res


def gauss_product(mean1: Union[np.ndarray, float], mean2: Union[np.ndarray, float],
                  cov1: Union[np.ndarray, float], cov2: Union[np.ndarray, float]):
    """
    Given the mean and variance/covariance matrices of two Gaussian distribution, this function computes the mean,
    variance/covariance matrix of the resultant product (which is a scaled version of another Gaussian distribution).
    The scaling factor is also returned to normalise the product to a proper pdf
    :param mean1 :param mean2: Means
    :param cov1: :param cov2: Variance/Covariance Matrices
    :return: resultant mean, variance/covariance matrix, scaling factor
    """
    # Cast float (if any) data types to numpy arrays
    mean1 = np.asarray(mean1)
    mean2 = np.asarray(mean2)
    cov1 = np.asarray(cov1)
    cov2 = np.asarray(cov2)

    # Sanity Checks
    assert mean1.shape == mean2.shape, "mean1 has shape "+str(mean1.shape)+", but mean2 has shape "+str(mean2.shape)
    assert cov1.shape == cov2.shape, "cov1 has shape "+str(cov1.shape)+", but cov2 has shape "+str(cov2.shape)
    try:
        dim = mean1.shape[0]
    except IndexError:
        # Input is a number
        dim = 1

    # Multivariate Gaussian - Use Matrix Algebra
    if dim > 1:
        # Precision matrices
        precision1 = np.linalg.inv(cov1)
        precision2 = np.linalg.inv(cov2)

        # Product Covariance Matrix
        product_cov = np.linalg.inv(precision1 + precision2)
        product_mean = product_cov @ (precision1 @ mean1 + precision2 @ mean2)
        scaling_factor = (2 * np.pi) ** (-dim / 2) * np.linalg.det(cov1 + cov2) ** -0.5 * \
            np.exp(-0.5 * np.transpose(mean1 - mean2) @ np.linalg.inv(cov1 + cov2) @ (mean1 - mean2))

    # Univariate Gaussian - Simply use number operations
    elif dim == 1:
        product_cov = 1 / (1/cov1 + 1/cov2)
        product_mean = product_cov * (mean1/cov1 + mean2/cov2)
        scaling_factor = 1 / np.sqrt(2 * np.pi * (cov1 + cov2)) * np.exp(- ((mean1 - mean2) ** 2) /
                                                                            (2 * (cov1 + cov2)))
    else:
        raise ValueError("Invalid input shape")
    return product_mean, product_cov, scaling_factor

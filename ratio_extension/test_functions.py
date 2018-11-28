# Xingchen Wan | 2018
# Generation of the test functions that are made from Gaussian mixture for evaluation of the quality of integration
# of various numerical integration techniques. Due to the use of Gaussian mixtures these integrals are analytically
# tractable; these exact values form the benchmark of comparison.


import numpy as np
from scipy.stats import norm, multivariate_normal
import matplotlib.pyplot as plt
from typing import Union
from bayesquad.priors import Prior
from scipy.integrate import quad


class TrueFunctions:
    """Abstract class for True Functions"""
    def __init__(self, ):
        self.dimensions = None

    def sample(self, x: Union[np.ndarray, float, list]) -> np.ndarray:
        pass

    def log_sample(self, x:Union[np.ndarray, float, list]) -> np.ndarray:
        return np.log(self.sample(x))

    def _sample_test(self, x: Union[np.ndarray, float, list]):
        x = np.asarray(x)
        if x.ndim == 0 or x.ndim == 1:
            assert self.dimensions == 1, "Scalar input is only permitted if the function is of dimension!"
        else:
            # If plotting a list of points, the ndim of the supplied x np array must be 2
            assert x.ndim == 2
            assert x.shape[1] == self.dimensions, "Dimension of function is of" + str(self.dimensions) + \
                                                  " ,but the dimension of input is " + str(x.shape[1])
        return x

    def plot(self, plot_range: tuple = (-3., 0.01, 3.), **matplot_options):
        assert self.dimensions <= 2, "Plotting higher dimension functions are not supperted!"
        range_min, range_step, range_max = plot_range[0], plot_range[1], plot_range[2]
        plot_x = np.arange(range_min, range_max, range_step + 0.0)
        plot_y = np.array(self.sample(plot_x))
        plt.plot(plot_x, plot_y, **matplot_options)


class ProductOfGaussianMixture(TrueFunctions):
    """
    A test function that is product of n Gaussian mixtures (defined below)
    """
    def __init__(self, *gauss_mixtures: TrueFunctions):
        super(ProductOfGaussianMixture, self).__init__()
        gauss_mixtures_dims = []
        for each_mixture in gauss_mixtures:
            assert isinstance(each_mixture, GaussMixture), "Invalid Type: GaussMixture object(s) expected"
            gauss_mixtures_dims.append(each_mixture.dimensions)
        assert len(set(gauss_mixtures_dims)) == 1, "There are different dimensions in the GaussMixture objects!"
        self.dimensions = gauss_mixtures_dims[0]
        self.gauss_mixtures = gauss_mixtures
        self.gauss_mixtures_count = len(gauss_mixtures)

    def sample(self, x: Union[np.ndarray, float, list]):
        x = self._sample_test(x)
        if x.ndim <= 1:
            y_s = np.array([each_mixture.sample(x) for each_mixture in self.gauss_mixtures])
            print(y_s)
            return np.prod(y_s)
        else:
            y_s = []
            for j in range(x.shape[0]):
                y_s.append([each_mixture.sample(x[j]) for each_mixture in self.gauss_mixtures])
            y_s = np.asarray(y_s)
            return np.prod(y_s, axis=1)


class GaussMixture(TrueFunctions):
    """
    A test function consists of a mixture (summation) of Gaussians (so used because it allows the evaluation of
    the integration exactly as a benchmark for other quadrature methods.
    """
    def __init__(self, means: Union[np.ndarray, float, list], covariances: Union[np.ndarray, float, list],
                 weights: Union[np.ndarray, list, float]=None):
        super(GaussMixture, self).__init__()

        self.means = np.asarray(means)
        self.covs = np.asarray(covariances)
        self.mixture_count = len(self.means)

        if self.means.ndim == 1:
            self.dimensions = 1
        else:
            self.dimensions = self.means.shape[1]
        if weights is None:
            # For unspecified weights, each individual Gaussian distribution within the mixture will receive
            # an equal weight
            weights = np.array([1./self.mixture_count]*self.mixture_count)
        self.weights = np.asarray(weights)
        assert self.means.shape[0] == self.covs.shape[0], "Mean and Covariance List mismatch!"
        assert self.means.shape[0] == self.weights.shape[0]
        assert self.weights.ndim <= 1, "Weight vector must be a 1D array!"

    def sample(self, x: Union[np.ndarray, float, list], ):
        """
        Sample from the true function either with one query point or a list of points
        :param x: the coordinate(s) of the query point(s)
        :return: the value of the true function evaluated at the query point(s)
        """
        x = self._sample_test(x)
        if x.ndim <= 1:
            y = 0
            for i in range(self.mixture_count):
                if self.dimensions == 1:
                    y += self.weights[i] * self.one_d_normal(x, self.means[i], self.covs[i])
                else:
                    y += self.weights[i] * self.multi_d_gauss(x, self.means[i], self.covs[i])
        else:
            x = np.squeeze(x, axis=1)
            y = np.zeros((x.shape[0], ))
            for i in range(self.mixture_count):
                if self.dimensions == 1:
                    y += self.weights[i] * self.one_d_normal(x, self.means[i], self.covs[i])
                else:
                    y += self.weights[i] * self.multi_d_gauss(x, self.means[i], self.covs[i])
        return y

    @staticmethod
    def one_d_normal(x: np.ndarray, mean, var) -> np.ndarray:
        assert x.ndim == 1
        return np.array([norm.pdf(x[i], mean, var) for i in range(x.shape[0])])

    @staticmethod
    def multi_d_gauss(x: np.ndarray, mean, cov) -> np.ndarray:
        assert x.ndim == 2
        return np.array([multivariate_normal.pdf(x[i], mean=mean, cov=cov) for i in range(x.shape[0])])

    def add_gaussian(self, means: Union[np.ndarray, float], var: Union[np.ndarray, float], weight: Union[np.ndarray, float]):
        assert means.shape == self.means.shape[1:]
        assert var.shape == self.covs.shape[1:]
        self.means = np.append(self.means, means)
        self.covs = np.append(self.covs, var)
        self.weights = np.append(self.weights, weight)

    def _rebase_weight(self):
        self.weights = self.weights / np.sum(self.weights)


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
        weight = gauss_mix.weights[i]
        _, _, scaling_factor = gauss_product(mu, prior_mean, sigma, prior_var)
        res += scaling_factor * weight
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
        weight1 = gauss_mix_1.weights[i]
        for j in range(gauss_mix_2.mixture_count):
            mu2 = gauss_mix_2.means[j]
            sigma2 = gauss_mix_2.covs[j]
            weight2 = gauss_mix_2.weights[j]
            mu_product, sigma_product, scale_product = gauss_product(mu1, mu2, sigma1, sigma2)
            _, _, scale_with_prior = gauss_product(mu_product, prior_mean, sigma_product, prior_var)
            res += scale_with_prior * scale_product * weight1 * weight2
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


def approx_integrals(p: Prior, q: TrueFunctions, r: TrueFunctions) -> tuple:
    def pr(x: float) -> float:
        x = np.array([[x]])
        return np.asscalar(p(x) * r.sample(x))

    def pqr(x: float) -> float:
        x = np.array([[x]])
        return np.asscalar(p(x) * q.sample(x) * r.sample(x))

    integral_pr = quad(pr, -10, 10, )
    integral_pqr = quad(pqr, -10, 10, )
    ratio = integral_pqr[0]/integral_pr[0]
    print("Denominator Integral:", str(integral_pr))
    print('Numerator Integral: ', str(integral_pqr))
    print('Ratio: ',str(ratio))
    return integral_pqr[0], integral_pr[0], ratio

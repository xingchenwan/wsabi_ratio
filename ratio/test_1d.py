# Xingchen Wan | 2018
# Generation of the test functions that are made from Gaussian mixture for evaluation of the quality of integration
# of various numerical integration techniques. Due to the use of Gaussian mixtures these integrals are analytically
# tractable; these exact values form the benchmark of comparison.


import numpy as np
from typing import Union
from bayesquad.priors import Prior
from scipy.integrate import quad
from ratio.functions import Functions, GaussMixture, ProductOfGaussianMixture


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


def approx_integrals(p: Prior, q: Functions, r: Functions) -> tuple:
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

from ratio_extension.test_functions import *
import matplotlib.pyplot as plt
from ratio_extension.naive_bq import NaiveWSABI
from bayesquad.priors import Gaussian
from ratio_extension.prior_1d import Gaussian1D

if __name__ == "__main__":
    a = GaussMixture(means=[-5, 2], covariances=[4, 2])

    b = GaussMixture([2, 3], [1, 2])

    prior = Gaussian1D(mean=0, variance=1)

    posterior = predictive_integral(a, b, prior_mean=0, prior_var=1) / \
                evidence_integral(a, prior_mean=0, prior_var=1)
    print(posterior)

    naive_bq = NaiveWSABI(a, b, prior, )
    naive_bq.quadrature()
    naive_bq.plot_result(posterior)
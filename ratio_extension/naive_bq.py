# Xingchen Wan | 2018
# Implementation of the naive Bayesian quadrature for ratios using WSABI

from bayesquad.quadrature import IntegrandModel


class NaiveWSABI:
    """
    Naive WSABI models the numerator and denominator integrand independently using WSABI algorithm. For
    reference, the ratio of integrals is in the form of:
    \math
        \frac{\int q(\phi)r(\phi)p(\phi)d\phi}{\int r(\phi)p(\phi)d\phi}
    \math
    where q(\phi) = p(y|z, \phi) and r(\phi) = p(z|\phi). These functions can be evaluated but are somewhat expensive.
    The objective is to infer a functional form of both r and q using Gaussian process then using Gaussian quadrature
    to complete the integration. Note that the naive method does not take into consideration of the correlation in
    the numerator and denominator.

    Since the denominator is in the form of a (fairly simple) Bayesian quadrature problem, the denominator integral is
    evaluated first. For this case, the samples selected on the hyperparameter (\phi) space are also used to evaluate
    the numerator integrand.
    """

    def __init__(self, numerator_integrand: IntegrandModel, denominator_integrand: IntegrandModel):
        self.numerator_integrand = numerator_integrand
        self.denominator_integrand = denominator_integrand

    def active_sample(self):
        pass


class NaiveBQ:
    pass


class NaiveMonteCarlo:
    pass

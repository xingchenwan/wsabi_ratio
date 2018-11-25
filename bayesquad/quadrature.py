"""Provides a model of the integrand, with the capability to perform Bayesian quadrature."""
# Modified by Xingchen Wan to adapt to the WSABI for ratio project - changed the class hierarchy and added support for
# the integrand model of a vanilla (non-WSABI) Bayesian Quadrature method

from typing import Tuple, Union

import numpy as np
from GPy.kern import Kern, RBF
# from multimethod import multimethod
from numpy import ndarray, newaxis

from .decorators import flexible_array_dimensions
from .gps import WarpedGP, WsabiLGP, GP
from .maths_helpers import jacobian_of_f_squared_times_g, hessian_of_f_squared_times_g
from .priors import Gaussian, Prior
from ratio_extension.prior_1d import Gaussian1D
from abc import abstractmethod
from scipy.stats import multivariate_normal, norm
from scipy.linalg import cho_solve, cho_factor


class IntegrandModel:
    """
    Added a base class to accommodate both the WSABI and vanilla Bayesian quadrature methods.
    Addition by Xingchen Wan - 11 Nov 2018
    """
    def __init__(self, gp, prior: Prior):
        self.gp = gp
        self.prior = prior
        self.dimensions = gp.dimensions

    @flexible_array_dimensions
    def posterior_mean_and_variance(self, x: ndarray) -> Tuple[ndarray, ndarray]:
        """Get the posterior mean and variance of the product of warped GP and prior at a point, or a set of points.

        Parameters
        ----------
        x
            The point(s) at which to evaluate the posterior mean and variance. A 2D array of shape
            (num_points, num_dimensions), or a 1D array of shape (num_dimensions).

        Returns
        -------
        mean : ndarray
            A 1D array of shape (num_points) if the input was 2D, or a 0D array if the input was 1D. The :math:`i`-th
            element is the posterior mean at the :math:`i`-th point of `x`.
        variance : ndarray
            A 1D array of shape (num_points) if the input was 2D, or a 0D array if the input was 1D. The :math:`i`-th
            element is the posterior variance at the :math:`i`-th point of `x`.
        """
        gp_mean, gp_var = self.gp.posterior_mean_and_variance(x)
        prior = self.prior(x)

        mean = gp_mean * prior
        variance = gp_var * prior ** 2

        return mean, variance

    @abstractmethod
    def posterior_variance_jacobian(self, x: ndarray) -> ndarray:
        """Get the jacobian of the posterior variance of the product of warped GP and prior at a point or set of points.

        Parameters
        ----------
        x
            The point(s) at which to evaluate the jacobian. A 2D array of shape (num_points, num_dimensions), or a 1D
            array of shape (num_dimensions).

        Returns
        -------
        jacobian : ndarray
            A 2D array of shape (num_points, num_dimensions) if the input was 2D, or a 1D array of shape
            (num_dimensions) if the input was 1D. The :math:`(i, j)`-th element is the :math:`j`-th component of the
            jacobian of the posterior variance at the :math:`i`-th point of `x`.

        Notes
        -----
        Writing :math:`\\pi(x)` for the prior, and :math:`V(x)` for the posterior variance, the posterior variance of
        the product is :math:`\\pi(x)^2 V(x)`.
        """
        pass

    @abstractmethod
    def posterior_variance_hessian(self, x: ndarray) -> ndarray:
        """Get the hessian of the posterior variance of the product of warped GP and prior at a point, or set of points.

        Parameters
        ----------
        x
            The point(s) at which to evaluate the hessian. A 2D array of shape (num_points, num_dimensions), or a 1D
            array of shape (num_dimensions).

        Returns
        -------
        hessian : ndarray
            A 3D array of shape (num_points, num_dimensions, num_dimensions) if the input was 2D, or a 2D array of shape
            (num_dimensions, num_dimensions) if the input was 1D. The :math:`(i, j, k)`-th element is the
            :math:`(j, k)`-th mixed partial derivative of the posterior variance at the :math:`i`-th point of `x`.

        Notes
        -----
        Writing :math:`\\pi(x)` for the prior, and :math:`V(x)` for the posterior variance, the posterior variance of
        the product is :math:`\\pi(x)^2 V(x)`.
        """
        pass

    def update(self, x: ndarray, y: ndarray):
        """Add new data to the model.

        Parameters
        ----------
        x
            A 2D array of shape (num_points, num_dimensions), or a 1D array of shape (num_dimensions).
        y
            A 1D array of shape (num_points). If X is 1D, this may also be a 0D array or float.

        Raises
        ------
        ValueError
            If the number of points in `x` does not equal the number of points in `y`.
        """
        self.gp.update(x, y)

    def integral_mean(self) -> float:
        """Compute the mean of the integral of the function under this model."""
        if isinstance(self.prior, Gaussian) and isinstance(self.gp.kernel, RBF):
            return self._compute_mean(self.prior, self.gp, self.gp.kernel)
        elif isinstance(self.prior, Gaussian1D) and isinstance(self.gp.kernel, RBF):
            return self._compute_mean(self.prior, self.gp, self.gp.kernel)
        else:
            raise NotImplementedError()

    @staticmethod
    @abstractmethod
    def _compute_mean(prior, gp, kernel) -> float: pass

    def fantasise(self, x, y):
        self.gp.fantasise(x, y)

    def remove_fantasies(self):
        self.gp.remove_fantasies()


class WarpedIntegrandModel(IntegrandModel):
    """Represents the product of a warped Gaussian Process and a prior.

    Typically, this product is the function that we're interested in integrating."""

    def __init__(self, warped_gp: WarpedGP, prior: Prior):
        super(WarpedIntegrandModel, self).__init__(warped_gp, prior)

    @flexible_array_dimensions
    def posterior_variance_jacobian(self, x: ndarray) -> ndarray:
        _, gp_variance = self.gp.posterior_mean_and_variance(x)
        gp_variance_jacobian = self.gp.posterior_variance_jacobian(x)

        prior = self.prior(x)
        prior_jacobian, _ = self.prior.gradient(x)
        return jacobian_of_f_squared_times_g(
            f=prior, f_jacobian=prior_jacobian,
            g=gp_variance, g_jacobian=gp_variance_jacobian)

    @flexible_array_dimensions
    def posterior_variance_hessian(self, x: ndarray) -> ndarray:
        _, gp_variance = self.gp.posterior_mean_and_variance(x)
        gp_variance_jacobian = self.gp.posterior_variance_jacobian(x)
        gp_variance_hessian = self.gp.posterior_variance_hessian(x)

        prior = self.prior(x)
        prior_jacobian, prior_hessian = self.prior.gradient(x)

        return hessian_of_f_squared_times_g(
            f=prior, f_jacobian=prior_jacobian, f_hessian=prior_hessian,
            g=gp_variance, g_jacobian=gp_variance_jacobian, g_hessian=gp_variance_hessian)

    @staticmethod
    def _compute_mean(prior: Union[Gaussian, Gaussian1D], gp: WarpedGP, kernel: RBF) -> float:
        dimensions = gp.dimensions

        alpha = gp._alpha
        kernel_lengthscale = kernel.lengthscale.values[0]
        kernel_variance = kernel.variance.values[0]

        X_D = gp._gp.X

        if isinstance(prior, Gaussian1D):
            mu = prior.matrix_mean
            sigma = prior.matrix_variance
            sigma_inv = prior.matrix_precision
        else:
            mu = prior.mean
            sigma = prior.covariance
            sigma_inv = prior.precision

        nu = (X_D[:, newaxis, :] + X_D[newaxis, :, :]) / 2
        A = gp._gp.posterior.woodbury_vector

        L = np.exp(
            -(np.linalg.norm(X_D[:, newaxis, :] - X_D[newaxis, :, :], axis=2) ** 2) / (4 * kernel_lengthscale ** 2))
        L = kernel_variance ** 2 * L
        L = np.linalg.det(2 * np.pi * sigma) ** (-1 / 2) * L

        C = sigma_inv + 2 * np.eye(dimensions) / kernel_lengthscale ** 2

        C_inv = np.linalg.inv(C)
        if dimensions == 1:
            gamma_part = 2 * nu / kernel_lengthscale ** 2 + (sigma_inv @ mu)[newaxis, :]
        else:
            gamma_part = 2 * nu / kernel_lengthscale ** 2 + (sigma_inv @ mu)[newaxis: newaxis, :]
        gamma = np.einsum('kl,ijl->ijk', C_inv, gamma_part)

        k_1 = 2 * np.einsum('ijk,ijk->ij', nu, nu) / kernel_lengthscale ** 2
        k_2 = mu.T @ sigma_inv @ mu
        k_3 = np.einsum('ijk,kl,ijl->ij', gamma, C, gamma)

        k = k_1 + k_2 - k_3

        K = np.exp(-k / 2)

        return alpha + (np.linalg.det(2 * np.pi * np.linalg.inv(C)) ** 0.5) / 2 * (A.T @ (K * L) @ A)


class OriginalIntegrandModel(IntegrandModel):
    """
    This class serves similar functions as the previous Integrand Model but is used for the usual GP (rather than warped
    GP for the case of WSABI

    Addition by Xingchen Wan - 11 Nov 2018
    """
    def __init__(self, gp: GP, prior: Prior):
        super(OriginalIntegrandModel, self).__init__(gp=gp, prior=prior)

    @flexible_array_dimensions
    def posterior_variance_jacobian(self, x: ndarray) -> ndarray:
        _, gp_variance = self.gp.posterior_mean_and_variance(x)
        _, gp_variance_jacobian = self.gp.posterior_jacobians(x)

        prior = self.prior(x)
        prior_jacobian, _ = self.prior.gradient(x)
        return jacobian_of_f_squared_times_g(
            f=prior, f_jacobian=prior_jacobian,
            g=gp_variance, g_jacobian=gp_variance_jacobian)

    @flexible_array_dimensions
    def posterior_variance_hessian(self, x: ndarray) -> ndarray:
        _, gp_variance = self.gp.posterior_mean_and_variance(x)
        _, gp_variance_jacobian = self.gp.posterior_jacobians(x)
        _, gp_variance_hessian = self.gp.posterior_hessians(x)

        prior = self.prior(x)
        prior_jacobian, prior_hessian = self.prior.gradient(x)

        return hessian_of_f_squared_times_g(
            f=prior, f_jacobian=prior_jacobian, f_hessian=prior_hessian,
            g=gp_variance, g_jacobian=gp_variance_jacobian, g_hessian=gp_variance_hessian)

    @staticmethod
    def _compute_mean(prior: Union[Gaussian, Gaussian1D], gp: GP, kernel: RBF) -> float:
        # todo: add the support to compute the variance of the quadrature estimation as well.
        from GPy.util.linalg import jitchol
        # w, h are the lengthscale and variance of the RBF kernel - see Equation 7.1.4 in Mike's DPhil Dissertation
        w = kernel.lengthscale.values[0]
        h = kernel.variance.values[0]

        X_D = gp._gpy_gp.X
        Y_D = gp._gpy_gp.Y
        n, d = X_D.shape
        # n: number of samples, d: dimensionality of each sample

        if isinstance(prior, Gaussian1D):
            mu = prior.matrix_mean
            sigma = prior.matrix_variance
        else:
            mu = prior.mean
            sigma = prior.covariance

        # Defined in Equations 7.1.7
        n_s = np.zeros((n, ))

        if d == 1:
            W = float(sigma) + w ** 2
            mu = np.asscalar(mu)
            for i in range(n):
                n_s[i] = h * norm.pdf(X_D[i, :], loc=mu, scale=np.sqrt(W))
        else:
            W = sigma
            W_2 = 2 * sigma
            for i in range(d):
                W[i, i] += w[i] ** 2
                W_2[i, i] += w[i] ** 2
            for i in range(n):
                n_s[i] = h * multivariate_normal(X_D[i, :], mean=mu, cov=W)
        K_xx = kernel.K(X_D)
        # Find the inverse of K_xx matrix via Cholesky decomposition (with jitter)
        K_xx_cho = jitchol(K_xx,)
        choleksy_inverse = np.linalg.inv(K_xx_cho)
        K_xx_inv = choleksy_inverse.T @ choleksy_inverse
        mean = n_s.T @ K_xx_inv @ Y_D
        return mean




"""
# Omitted due to lack of compatibility of the multimethod package
@multimethod
def _compute_mean(prior: Prior, gp: WarpedGP, kernel: Kern) -> float:
    ""Compute the mean of the integral for the given prior, warped GP, and kernel.

    This method will delegate to other methods of the same name defined in this module, based on the type of the
    arguments. If no implementation is found for the provided types, this default implementation will raise an error.""
    raise NotImplementedError("Integration is not supported for this combination of prior, warping and kernel.\n\n"
                              "Prior was of type {}.\n"
                              "Warped GP was of type {}.\n"
                              "Kernel was of type {}."
                              .format(type(prior), type(gp), type(kernel)))


@multimethod

def _compute_mean(prior: Gaussian, gp: WsabiLGP, kernel: RBF) -> float:
    ""Compute the mean of the integral for a WSABI-L GP with a squared exponential kernel against a Gaussian prior.""
    dimensions = gp.dimensions

    alpha = gp._alpha
    kernel_lengthscale = kernel.lengthscale.values[0]
    kernel_variance = kernel.variance.values[0]

    X_D = gp._gp.X

    mu = prior.mean
    sigma = prior.covariance
    sigma_inv = prior.precision

    nu = (X_D[:, newaxis, :] + X_D[newaxis, :, :]) / 2
    A = gp._gp.posterior.woodbury_vector

    L = np.exp(-(np.linalg.norm(X_D[:, newaxis, :] - X_D[newaxis, :, :], axis=2) ** 2)/(4 * kernel_lengthscale**2))
    L = kernel_variance ** 2 * L
    L = np.linalg.det(2 * np.pi * sigma) ** (-1/2) * L

    C = sigma_inv + 2 * np.eye(dimensions) / kernel_lengthscale ** 2

    C_inv = np.linalg.inv(C)
    gamma_part = 2 * nu / kernel_lengthscale ** 2 + (sigma_inv @ mu)[newaxis, newaxis, :]

    gamma = np.einsum('kl,ijl->ijk', C_inv, gamma_part)

    k_1 = 2 * np.einsum('ijk,ijk->ij', nu, nu) / kernel_lengthscale ** 2
    k_2 = mu.T @ sigma_inv @ mu
    k_3 = np.einsum('ijk,kl,ijl->ij', gamma, C, gamma)

    k = k_1 + k_2 - k_3

    K = np.exp(-k/2)

    return alpha + (np.linalg.det(2 * np.pi * np.linalg.inv(C)) ** 0.5)/2 * (A.T @ (K * L) @ A)
"""
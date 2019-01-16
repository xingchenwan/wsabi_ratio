import numpy as np
from scipy.stats import multivariate_normal, norm
from GPy.util.linalg import jitchol


def integral_mean_rebased(gpy_gp, prior_mean, prior_var, compute_var=False):
    X = gpy_gp.X
    Y = gpy_gp.Y

    n, d = X.shape[0], X.shape[1]
    assert prior_mean.ndim == 1
    assert prior_var.ndim == 2
    assert prior_mean.shape[0] == d
    assert prior_var.shape[0] == d
    assert prior_var.shape[0] == prior_var.shape[1]

    scaling = np.max(Y)
    # print(scaling)
    Y = np.exp(Y - scaling)

    mu = prior_mean

    # Kernel parameters
    w = np.exp(gpy_gp.kern.lengthscale.values)
    h = np.exp(gpy_gp.kern.variance.values[0])

    if len(w) == 1:
        w = np.array([w]*d).reshape(-1)
    W = np.diag(w)        # Assuming isotropic covariance, build the W matrix from w parameters
    V = prior_var

    n_s = np.zeros((n, ))

    for i in range(n):
        n_s[i] = h * multivariate_normal.pdf(X[i, :], mean=mu, cov=W+V)
    # print(Y)
    c_f = np.linalg.det(2 * np.pi * (2 * W + V)) ** -0.5

    K_xx = gpy_gp.kern.K(X)
    # Find the inverse of K_xx matrix via Cholesky decomposition (with jitter)
    K_xx_cho = jitchol(K_xx, )
    choleksy_inverse = np.linalg.inv(K_xx_cho)
    K_xx_inv = choleksy_inverse.T @ choleksy_inverse

    unscaled_integral_mean = n_s.T @ K_xx_inv @ Y

    if compute_var:
        unscaled_integral_var = c_f - n_s.T @ K_xx_inv @ n_s
        scaled_var = np.log(unscaled_integral_var) + 2 * scaling
    else:
        scaled_var = np.nan
    scaled_mean = np.log(unscaled_integral_mean) + scaling

    return scaled_mean, scaled_var


def integral_mean_without_rebase(gpy_gp, prior_mean, prior_var, compute_var=False):
    X = gpy_gp.X
    Y = gpy_gp.Y

    n, d = X.shape[0], X.shape[1]
    assert prior_mean.ndim == 1
    assert prior_var.ndim == 2
    assert prior_mean.shape[0] == d
    assert prior_var.shape[0] == d
    assert prior_var.shape[0] == prior_var.shape[1]

    mu = prior_mean

    # Kernel parameters
    w = gpy_gp.kern.lengthscale.values
    h = gpy_gp.kern.variance.values[0]


    if len(w) == 1:
        w = np.array([w] * d).reshape(-1)
    W = np.diag(w ** 2)  # Assuming isotropic covariance, build the W matrix from w parameters
    V = prior_var

    n_s = np.zeros((n,))

    for i in range(n):
        n_s[i] = h * multivariate_normal.pdf(X[i, :], mean=mu, cov=W + V)
    # print(Y)
    c_f = np.linalg.det(2 * np.pi * (2 * W + V)) ** -0.5

    K_xx = gpy_gp.kern.K(X)
    # Find the inverse of K_xx matrix via Cholesky decomposition (with jitter)
    K_xx_cho = jitchol(K_xx, )
    choleksy_inverse = np.linalg.inv(K_xx_cho)
    K_xx_inv = choleksy_inverse.T @ choleksy_inverse

    unscaled_integral_mean = n_s.T @ K_xx_inv @ Y

    if compute_var:
        unscaled_integral_var = c_f - n_s.T @ K_xx_inv @ n_s
    else:
        unscaled_integral_var = np.nan

    return unscaled_integral_mean, unscaled_integral_var

# Eigenvalue Linear Shrinkage Gaussian Process Experiment - Xingchen Wan 2018

import pandas as pd
import numpy as np
import GPy
from IPython.display import display
from LinShrinkageProject.LogMultivariateGaussian import LogMultivariateGaussian
import matplotlib.pyplot as plt
import seaborn as sns

file_path = "data/yacht_hydrodynamics.data.txt"
MLE_optimisation_restart = 20
MLE_optimisation_iteration = 1000
MCMC_samples = 10
MCMC_burn_in = 2


def load_data(validation_ratio: float=0.3):
    """
    Load data-set, do relevant processing and split the data into training and validation sets
    :param validation_ratio: the fraction of data that will be reserved as validation sets
    :return: the data (X) and labels(Y) of the training and validation data
    """
    assert validation_ratio < 1.
    raw_data = pd.read_csv(filepath_or_buffer=file_path, header=None, sep='\s+')
    data_X = raw_data.iloc[:, :-1].values
    data_Y = raw_data.iloc[:, -1].values
    if data_Y.ndim == 1:
        data_Y = data_Y.reshape(-1, 1)
    if data_X.ndim == 1:
        data_X = np.array([data_X])
    assert data_X.shape[0] == data_Y.shape[0]

    train_data_length = int(data_X.shape[0] * (1. - validation_ratio))
    train_data_X = data_X[:train_data_length, :]
    validation_data_X = data_X[train_data_length:, :]
    train_data_Y = data_Y[:train_data_length, :]
    validation_data_Y = data_Y[train_data_length:, :]
    return train_data_X, train_data_Y, validation_data_X, validation_data_Y


def fit_gp(data_X: np.ndarray, data_Y: np.ndarray, init_params=None) -> GPy.models.GPRegression:
    """
    Fit a rbf kernel Gaussian process model to the data
    :return:
    """
    assert data_X.shape[0] == data_Y.shape[0], "Lengths of the x and y mismatch"
    if init_params is None:
        init_params = np.array([1.] * (data_X.shape[1]+2))
    else:
        assert len(init_params) == data_X.shape[1] + 2, \
            "Shape of the initial parameter vector must be " + \
            str(data_X.shape[1])+" ,i.e. kernel lengthscale of the data's dimension + kernel variance + noise param"
    dimension = data_X.shape[1]
    kern = GPy.kern.RBF(input_dim=dimension,
                        lengthscale=init_params[:-2],
                        variance=init_params[-2],
                        ARD=True)
    model = GPy.models.GPRegression(data_X, data_Y, kernel=kern)
    model.Gaussian_noise.variance = init_params[-1]
    return model


def param_maximum_likelihood(gpy_gp: GPy.models.GPRegression, test_model: bool = True,
                             test_X: np.ndarray = None, test_Y: np.ndarray = None,
                             variance_prior: GPy.priors.Prior = None,
                             lengthscale_prior: GPy.priors.Prior = None,
                             noise_prior: GPy.priors.Prior = None,):
    """
    Find the maximum likelihood estimates of the hyperparameters. The last element in the vector returned is the
    Gaussian noise hyperparameter that we are interested in.
    The hyperparameters correspond to
    $
    \theta^* = argmin(-log(P(Y|\theta)))
    $
    If prior_mean and prior_var arguments (the parameter prior and covariance, assuming a Gaussian parameter prior
    $p(\theta)$ are provided, the optimised hyperparameters corrspond to the MAP-II estimate:
    $
    \theta^* = argmin(-log(P(Y|/theta) - log(P(\theta))
    $

    :param gpy_gp: An initialised GPy GPRegression object
    :param test_model: toggle whether to display the information of the fitted GPRegression model
    :return: The vector of hyperparameters found from maximum likelihood estimate
    """
    if test_model and (test_X is None or test_Y is None):
        raise ValueError()
    if variance_prior is not None:
        gpy_gp.kern.variance.set_prior(variance_prior)
    if lengthscale_prior is not None:
        gpy_gp.kern.lengthscale.set_prior(lengthscale_prior)
    if noise_prior is not None:
        gpy_gp.Gaussian_noise.variance.set_prior(noise_prior)
        # todo: check this
    gpy_gp.optimize(messages=True, max_iters=MLE_optimisation_iteration)
    gpy_gp.optimize_restarts(num_restarts=MLE_optimisation_restart)
    res = [gpy_gp.rbf.variance, gpy_gp.rbf.lengthscale, gpy_gp.Gaussian_noise.variance]
    rmse = np.nan
    if test_model:
        display(gpy_gp)
        rmse = test_gp(gpy_gp, test_X, test_Y, display_model=True)
    return gpy_gp, res, rmse


def param_manual(gpy_gp: GPy.models.GPRegression,
                 manual_params,
                 test_model: bool = False,
                 test_X: np.ndarray = None, test_Y: np.ndarray = None):
    """
    Introduce a manual perturbation term. All other hyperparameters by default are the same as the MLE estimates
    :param gpy_gp:
    :param test_model:
    :param test_X:
    :param test_Y:
    :return:
    """
    assert len(manual_params) == gpy_gp.input_dim + 2
    gpy_gp.rbf.lengthscale = manual_params[:-2]
    gpy_gp.rbf.variance = manual_params[-2]
    gpy_gp.Gaussian_noise.variance = manual_params[-1]
    rmse = np.nan
    if test_model:
        display(gpy_gp)
        rmse = test_gp(gpy_gp, test_X, test_Y, display_model=True)
    return gpy_gp, manual_params, rmse


def param_hmc(gpy_gp: GPy.models.GPRegression,
              test_model: bool = True,
              variance_prior: GPy.priors.Prior = None,
              lengthscale_prior: GPy.priors.Prior = None,
              noise_prior: GPy.priors.Prior = None,
              test_X: np.ndarray = None, test_Y: np.ndarray = None,
              plot_distributions: bool = False):
    """
    Compute the posterior distribution of the parameters using hybrid Monte Carlo, and then set the hyperparameters
    to the mean of each term. This is a more Bayesian approach over MLE-II or MAP-II.
    :param gpy_gp:
    :param test_model:
    :param test_X:
    :param test_Y:
    :return:
    """
    if test_model and (test_X is None or test_Y is None):
        raise ValueError()
    if variance_prior is not None:
        gpy_gp.kern.variance.set_prior(variance_prior)
    if lengthscale_prior is not None:
        gpy_gp.kern.lengthscale.set_prior(lengthscale_prior)
    if noise_prior is not None:
        gpy_gp.Gaussian_noise.variance.set_prior(noise_prior)
    display(gpy_gp)
    hmc = GPy.inference.mcmc.HMC(gpy_gp, stepsize=5e-2)
    t = hmc.sample(num_samples=MCMC_samples)
    if plot_distributions:
        df = pd.DataFrame(t, columns=gpy_gp.parameter_names_flat())
        ax = sns.distplot(df.iloc[:, -1], color='r', )
        plt.show()
    samples = t[MCMC_burn_in:, :]
    gpy_gp.kern.variance[:] = samples[:, -2].mean()
    gpy_gp.kern.lengthscale[:] = samples[:, :-2].mean()
    gpy_gp.kern.Gaussian_noise[:] = samples[:, -1].mean()
    res = [gpy_gp.rbf.variance, gpy_gp.rbf.lengthscale, gpy_gp.Gaussian_noise.variance]
    rmse = np.nan
    if test_model:
        display(gpy_gp)
        rmse = test_gp(gpy_gp, test_X, test_Y, display_model=True)
    return gpy_gp, res, rmse


def param_lin_shrinkage(gpy_gp: GPy.models.GPRegression, test_model: bool = True,
                        test_X: np.ndarray = None, test_Y: np.ndarray = None
                        ):
    """
    Use linear shrinkage method to estimate the Gaussian noise hyperparameter
    :param gpy_gp:
    :param test_model:
    :return:
    """
    pass


def test_gp(gpy_gp: GPy.models.GPRegression,
            data_X: np.ndarray, data_Y: np.ndarray, display_model: bool = False) -> float:
    """
    Evaluate the goodness of the model fit by RMSE value
    :param gpy_gp: the GPy regression model
    :param data_X: the query data points. Typically of x of validation data-set
    :param data_Y: the labeels. Typically of the y of validation data-set
    :param display_model: bool
    :return:
    """
    assert data_X.shape[0] == data_Y.shape[0], "Lengths of x and labels mismatch"
    assert data_X.shape[1] == gpy_gp.input_dim, "Dimension of x and the model dimension mismatch"
    mean_pred, var_pred = gpy_gp.predict(Xnew=data_X)
    rmse = np.sqrt(((mean_pred - np.squeeze(data_Y, -1)) ** 2).mean())
    if display_model is True:
        print("Root Mean Squared Error (RMSE): ", str(rmse))
    return rmse


if __name__ == '__main__':
    # Load the training and validation set data
    train_x, train_y, validate_x, validate_y = load_data()

    # Initialise a GP object on the training data
    m = fit_gp(train_x, train_y)

    # Save the input dimension
    input_dim = m.input_dim

    # Do a MLE-II estimate of the hyperparameters.
    m_mle, params_mle, _ = param_maximum_likelihood(m, test_X=validate_x, test_Y=validate_y)

    # Do a MAP-II estimate of the hyperparameters. Here we can specify a prior over $\theta$, the hyperparameter vector
    # For this case, we use an isotropic zero-mean Gaussian prior with variance of 2 in the *log-space* for the length-
    # scale and variance hyperparameters, and equivalently a Gaussian prior of mean of -6 and variance 2 for the noise
    # hyperparameter in the *log-space*
    variance_prior = GPy.priors.LogGaussian(mu=0., sigma=2.)

    lengthscale_prior_mu = np.array([0.]*input_dim)
    lengthscale_prior_var = np.eye(input_dim) * 2.
    # lengthscale_prior = LogMultivariateGaussian(mu=lengthscale_prior_mu, var=lengthscale_prior_var)
    lengthscale_prior = None

    noise_prior = GPy.priors.LogGaussian(mu=-4, sigma=2)

    # Use HMC sampling - we can also infer a posterior distribution of the hyperparameters using Monte Carlo technique
    m_hmc, params_hmc, _ = param_hmc(m_mle, lengthscale_prior=lengthscale_prior,
                                     variance_prior=variance_prior,
                                     noise_prior=noise_prior,
                                     test_X=validate_x, test_Y=validate_y,
                                     plot_distributions=True)


    m_map, params_map, _ = param_maximum_likelihood(m,
                                                    lengthscale_prior=lengthscale_prior,
                                                    variance_prior=variance_prior,
                                                    noise_prior=noise_prior,
                                                    test_X=validate_x, test_Y=validate_y)



    # Alternatively, we can manually supply the values of the hyperparameters
    # Here we use the MAP estimates for all hyperparameters apart from the Gaussian noise, and set a fixed value for the
    # Gaussian noise
    params_manual = params_map[:]
    params_manual[-1] = 1e-6  # The last element of the hyperparameter is noise
    m_manual, _, _ = param_manual(m, manual_params=params_manual)



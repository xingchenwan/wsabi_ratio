import numpy as np
import scipy.stats
import GPy
import matplotlib
matplotlib.use("TkAgg")
# This is to prevent a macOS bug with matplotlib
import matplotlib.pyplot as plt
from functools import partial
import matplotlib.cm as cm
import pandas as pd
from emukit.model_wrappers.gpy_quadrature_wrappers import BaseGaussianProcessGPy, RBFGPy
from emukit.quadrature.kernels import QuadratureRBF, IntegralBounds
from emukit.quadrature.methods import VanillaBayesianQuadrature
import logging
from bayesquad.quadrature import WarpedIntegrandModel, WsabiLGP, WarpedGP, GP
from bayesquad.priors import Gaussian

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
prune_length = 100


def BOCPD_GP(data, hazard_func, mode='mc',):
    maxes = np.zeros(len(data) + 1)

    R = np.zeros((len(data) + 1, len(data) + 1))
    Z = np.zeros((len(data)+1, 1))
    R[0, 0] = 1

    #gp_X = np.arange(0, data.shape[0], 1).reshape(-1, 1)
    #gp_Y = data.reshape(-1, 1)
    #kern = GPy.kern.Matern52(input_dim=1, lengthscale=10, variance=5)
    #gp = GPy.models.GPRegression(gp_X, gp_Y, kern)
    #gp.Gaussian_noise.variance = 1e-1

    gp = None
    for t, x in enumerate(data):
        # Evaluate the predictive distribution for the new datum under each of
        # the parameters.  This is the standard thing from Bayesian inference.
        if t == 0:
            gp_X = np.array(0).reshape(1, 1)
            gp_Y = np.array(x).reshape(1, 1)
            kern = GPy.kern.Matern32(input_dim=1, lengthscale=5., variance=10.)
            gp = GPy.models.GPRegression(gp_X, gp_Y, kern)
            gp.Gaussian_noise.variance = 1e-1
            if mode == 'fix':
                pred_mean, pred_var = fix_params(gp, x)
            elif mode == 'mc':
                pred_mean, pred_var = monte_carlo(gp, x)
            elif mode == 'bq':
                pred_mean, pred_var = bq(gp, x)
            elif mode == 'wsabi':
                pred_mean, pred_var = wsabi(gp, x)
            else:
                raise ValueError
            pred_mean = pred_mean.reshape(-1)
            pred_var = pred_var.reshape(-1)
        else:
            gp_X = np.concatenate([gp.X, np.array(t).reshape(1, 1)], axis=0)
            gp_Y = np.concatenate([gp.Y, np.array(x).reshape(1, 1)], axis=0)
            if gp_X.shape[0] > prune_length:
                gp_X = gp_X[1:, :]
                gp_Y = gp_Y[1:, :]
                assert gp_X.shape[0] == prune_length,"Expected length is" + str(prune_length) + \
                                                     "but got"+str(gp_X.shape[0])
            gp.set_XY(gp_X, gp_Y)

            try:
                if mode == 'fix':
                    pred_mean_t, pred_var_t = fix_params(gp, x)
                elif mode == 'mc':
                    pred_mean_t, pred_var_t = monte_carlo(gp, x)
                elif mode == 'bq':
                    pred_mean_t, pred_var_t = bq(gp, x)
                elif mode == 'wsabi':
                    pred_mean_t, pred_var_t = wsabi(gp, x)
                else:
                    raise ValueError
            except np.linalg.linalg.LinAlgError:
                logging.warning("Error at iteration"+str(t)+". Skipped")
                continue

            pred_mean = np.concatenate([pred_mean, pred_mean_t.reshape(-1)])
            pred_var = np.concatenate([pred_var, pred_var_t.reshape(-1)])
            logging.info("Progress: "+str(t) + "/" + str(data.shape[0]))

        predprobs = scipy.stats.norm.pdf(x, loc=pred_mean, scale=np.sqrt(pred_var))
        # print('probability', predprobs)
        # Evaluate the hazard function for this interval
        H = hazard_func(np.array(range(t + 1)))

        # Evaluate the growth probabilities - shift the probabilities down and to
        # the right, scaled by the hazard function and the predictive
        # probabilities.
        R[1:t + 2, t + 1] = R[0:t + 1, t] * predprobs * (1 - H)

        # Evaluate the probability that there *was* a changepoint and we're
        # accumulating the mass back down at r = 0.
        R[0, t + 1] = np.sum(R[0:t + 1, t] * predprobs * H)

        # Renormalize the run length probabilities for improved numerical
        # stability.
        Z[t] = np.sum(R[:, t+1]) # Evidence at time t
        R[:, t + 1] = R[:, t + 1] / Z[t]
        # print('R', R[:, t])
        # Update the parameter sets for each possible run length.

        maxes[t] = R[:, t].argmax()
    return R, maxes


def constant_hazard(lam, r):
    return 1 / lam * np.ones(r.shape)


def logistic_hazard(lam, r):
    pass


def fix_params(gpy_gp: GPy.core.gp, x):
    """GP regression with a fix set of hyperparameters"""
    x = np.array(x).reshape(1, 1)
    return gpy_gp.predict_noiseless(x)


def monte_carlo(gpy_gp: GPy.core.gp, x, budget=100):
    """Marginalise the GP Hyperparamters using Hybrid Monte Carlo"""
    from ratio.posterior_mc_inference import PosteriorMCSampler

    x = np.array(x).reshape(1, 1)
    sampler = PosteriorMCSampler(gpy_gp)
    samples = sampler.hmc(num_iters=budget, mode='gpy')
    pred_means = np.empty((budget, ))
    pred_vars = np.empty((budget, ))

    for i in range(samples.shape[0]):
        gpy_gp = _set_model(gpy_gp, samples[i, :])
        pred_means[i], pred_vars[i] = gpy_gp.predict_noiseless(x)
    return pred_means.sum(axis=0)/budget, pred_vars.sum(axis=0)/budget


def bq(gpy_gp: GPy.core.gp, x):
    """Initial grid sampling, followed by vanilla Bayesian Quadrature"""

    def _wrap_emukit(gpy_gp: GPy.core.GP):
        """
        Wrap GPy GP around Emukit interface to enable subsequent quadrature
        :param gpy_gp:
        :return:
        """
        gpy_gp.optimize()
        rbf = RBFGPy(gpy_gp.kern)
        qrbf = QuadratureRBF(rbf, integral_bounds=[(-10., 10.)])
        model = BaseGaussianProcessGPy(kern=qrbf, gpy_model=gpy_gp)
        method = VanillaBayesianQuadrature(base_gp=model)
        return method

    x = np.array(x).reshape(1, 1)
    log_params = np.mgrid[-2:2.1:1, -2:2.1:1, -4:1.1:1].reshape(3, -1).T
    logging.info(str(log_params.shape[0])+" samples obtained.")
    params = np.exp(log_params)
    log_liks = np.empty((params.shape[0],))
    pred_means = np.empty((params.shape[0], ))
    pred_vars = np.empty((params.shape[0], ))
    for i in range(params.shape[0]):
        gpy_gp = _set_model(gpy_gp, params[i, :])
        log_liks[i] = gpy_gp.log_likelihood()
        pred_means[i], pred_vars[i] = gpy_gp.predict_noiseless(x)

    rq = np.exp(log_liks) * pred_means
    rvar = np.exp(log_liks) * pred_vars

    init_lik_kern = GPy.kern.RBF(input_dim=params.shape[1], lengthscale=1., variance=1.)
    r_gp = GPy.models.GPRegression(log_params, np.exp(log_liks).reshape(-1, 1), init_lik_kern)
    r_model = _wrap_emukit(r_gp)
    r_int = r_model.integrate()[0]

    rq_gp = GPy.models.GPRegression(log_params, rq.reshape(-1, 1), init_lik_kern)
    rq_model = _wrap_emukit(rq_gp)
    rq_int = rq_model.integrate()[0]

    rvar_gp = GPy.models.GPRegression(log_params, rvar.reshape(-1, 1), init_lik_kern)
    rvar_model = _wrap_emukit(rvar_gp)
    rvar_int = rvar_model.integrate()[0]

    return rq_int/r_int, rvar_int/r_int


def wsabi(gpy_gp: GPy.core.GP, x,):
    """Initial grid sampling, followed by WSABI quadrature"""
    from ratio.posterior_mc_inference import PosteriorMCSampler
    budget = 50
    x = np.array(x).reshape(1, 1)
    sampler = PosteriorMCSampler(gpy_gp)
    log_params = sampler.hmc(num_iters=budget, mode='gpy')
    #log_params = np.empty((budget, 3))
    #for i in range(budget):
    #   log_params[i, :] = scipy.stats.multivariate_normal.rvs(mean=np.zeros((3, )), cov=4*np.eye(3))
    # log_params = np.mgrid[-1:3.1:1, -2:3.1:2, -4:0.1:1].reshape(3, -1).T
    # print(log_params)
    budget = log_params.shape[0]

    kern = GPy.kern.RBF(input_dim=1, variance=1., lengthscale=1.)
    prior = Gaussian(mean=np.zeros((3, )), covariance=4*np.eye(3))
    log_phis = np.empty((budget, 3))
    log_liks = np.empty((budget, ))
    pred_means = np.empty((budget, ))
    pred_vars = np.empty((budget, ))

    for i in range(log_params.shape[0]):
        log_phi = log_params[i, :]
        _set_model(gpy_gp, log_phi)
        log_lik = gpy_gp.log_likelihood()
        log_phis[i] = log_phi
        log_liks[i] = log_lik
        pred_means[i], pred_vars[i] = gpy_gp.predict_noiseless(x)

    r_gp = GPy.models.GPRegression(log_params[:1, :], np.sqrt(2 * np.exp(log_liks[0])).reshape(1, -1), kern)
    r_model = WarpedIntegrandModel(WsabiLGP(r_gp), prior)
    r_model.update(log_phis[1:, :], np.exp(log_liks[1:]).reshape(1, -1))
    r_gp.optimize()
    r_int = r_model.integral_mean()[0]

    q_min = np.min(pred_means)
    pred_means -= q_min

    rq = np.exp(log_liks) * pred_means
    rq_gp = GPy.models.GPRegression(log_phis[:1, :], np.sqrt(2 * rq[0]).reshape(1, -1), kern)
    rq_model = WarpedIntegrandModel((WsabiLGP(rq_gp)), prior)
    rq_model.update(log_phis[1:, :], rq[1:].reshape(1, -1))
    rq_gp.optimize()
    rq_int = rq_model.integral_mean()[0] + q_min * r_int

    rvar = np.exp(log_liks) * pred_vars
    rvar_gp = GPy.models.GPRegression(log_phis[:1, :], np.sqrt(2 * rvar[0]).reshape(1, -1), kern)
    rvar_model = WarpedIntegrandModel((WsabiLGP(rvar_gp)), prior)
    rvar_model.update(log_phis[1:, :], rvar[1:].reshape(1, -1))
    rvar_gp.optimize()
    rvar_int = rvar_model.integral_mean()[0]

    return rq_int / r_int, rvar_int / r_int


def _set_model(gpy_gp: GPy.core.GP, params):
    params = np.squeeze(params)
    assert params.ndim == 1
    assert params.shape[0] == gpy_gp.input_dim + 2
    gpy_gp.Mat32.lengthscale = params[0]
    gpy_gp.Mat32.variance = params[1]
    gpy_gp.Gaussian_noise.variance = params[2]
    return gpy_gp


def demo():
    file_path = '/Users/xingchenwan/Dropbox/4YP/Codes/wsabi_ratio/data/nile.csv'
    dji_2006 = '/Users/xingchenwan/Dropbox/4YP/Codes/wsabi_ratio/data/DJI_FinCrisis.csv'

    def generate_normal_time_series(num, minl=10, maxl=1000):
        data = np.array([], dtype=np.float64)
        partition = np.random.randint(minl, maxl, num)
        for p in partition:
            mean = np.random.randn() * 10
            var = np.random.randn() * 5
            if var < 0:
                var = var * -1
            tdata = np.random.normal(mean, var, p)
            data = np.concatenate((data, tdata))
        return data

    def load_nile_data():
        raw_data = pd.read_csv(file_path, ).values
        raw_data = raw_data[0:700, :]
        year = raw_data[:, 0]
        levels = raw_data[:, 1]
        return year, levels

    def load_dji_crisis():
        raw_data = pd.read_csv(dji_2006).values
        raw_data = raw_data[500:1000, :]
        log_rtn = raw_data[:, -1]
        daily_vol = raw_data[:, -2]
        return log_rtn, daily_vol



    # = generate_normal_time_series(7, 10, 11)

    year, data = load_nile_data()
    R, maxes = BOCPD_GP(data, partial(constant_hazard, 50))

    f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    ax1.plot(data,)
    #ax1.axvline(722-year[0], color='r', linewidth=2, linestyle='--')
    ax1.set_ylabel("Nile water level (m)")
    sparsity = 1  # only plot every fifth data for faster display
    ax2.pcolor(np.array(range(0, R.shape[0], sparsity)),
              np.array(range(0, prune_length, sparsity)),
              -np.log(R[:prune_length:sparsity, ::sparsity]),
              cmap=cm.hot, vmin=0, vmax=150)
    #ax2.axvline(722-year[0], color='r', linewidth=2, linestyle='--')
    ax2.set_xlabel("Number of years since AD 622")
    ax2.set_ylabel("Posterior Run Length (yr)")
    plt.show()
    # ax = fig.add_subplot(3, 1, 3, sharex=ax)
    # Nw = 10
    # plt.plot(R[Nw, Nw:-1])
    # plt.show()
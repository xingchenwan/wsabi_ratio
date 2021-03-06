# Xingchen Wan | xingchen.wan@st-annes.ox.ac.uk | Jan 2019

from ratio.functions import RBFGPRegression, PeriodicGPRegression
from bayesquad.priors import Gaussian
import numpy as np
from typing import Union, Tuple

import matplotlib
matplotlib.use("TkAgg")
# This is to prevent a macOS bug with matplotlib
import matplotlib.pyplot as plt
#  import seaborn as sns
#  import theano.tensor as tt
import pandas as pd

import GPy
from bayesquad.quadrature import WarpedIntegrandModel, WsabiLGP, WarpedGP, GP
from bayesquad.batch_selection import select_batch
from IPython.display import display

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
import time

from emukit.model_wrappers.gpy_quadrature_wrappers import BaseGaussianProcessGPy, RBFGPy
from emukit.quadrature.kernels import QuadratureRBF, IntegralBounds
from emukit.quadrature.methods import VanillaBayesianQuadrature
from emukit.quadrature.loop import VanillaBayesianQuadratureLoop
from ratio.functions import Functions
from random import randint


class ChangepointModel(Functions):
    def __init__(self,
                 file_path='/Users/xingchenwan/Dropbox/4YP/Codes/wsabi_ratio/data/nile.csv',
                 window_size=25,
                 method='mc'):
        super(ChangepointModel, self).__init__()
        self.file_path = file_path
        self.dimensions = 1
        self.param_dim = 5
        self.window_size = window_size
        #self.X, self.Y, self.n = self.load_data()
        self.X, self.Y, self.n = self.load_test_data()
        self.model = self.init_gp_model()
        self.method = method

    def init_gp_model(self):
        ker = GPy.kern.ChangepointRBF(input_dim=1, lengthscale1=1., lengthscale2=2., variance1=1., variance2=2., xc=0)
        x_init = self.X[:self.window_size].reshape(1, -1)
        y_init = self.Y[:self.window_size].reshape(1, -1)
        m = GPy.models.GPRegression(x_init, y_init, ker)
        return m

    def load_data(self):
        raw_data = pd.read_csv(self.file_path,)
        all_Y = raw_data.iloc[:, 1].values
        n = all_Y.shape[0]
        all_X = np.array([_ for _ in range(n)])
        return all_X, all_Y, n

    def load_test_data(self):
        X = np.array([_ for _ in range(80)])
        Y = np.empty(X.shape)
        for i in range(X.shape[0]):
            if i < 40:
                Y[i] = 1 + 0.02 * np.sin(X[i])
            else:
                Y[i] = 4 + 1 * np.sin(X[i])
        #plt.plot(X, Y)
        #plt.show()
        return X, Y, 80

    def online_pred(self, start_idx):
        # Do one-step-forward prediction based on the data x and y. along with the sufficient statistics of the
        # posterior distribution of parameter x_c (i.e. the run length)
        if start_idx < self.window_size:
            logging.info("Insuffcient run length. Skipped")
            return np.nan, np.nan, np.nan
        elif start_idx < self.n - self.window_size-1:
            x_new = self.X[start_idx: start_idx+self.window_size]
            y_new = self.Y[start_idx: start_idx+self.window_size]
            x_pred = self.X[start_idx+self.window_size]
            y_actual = self.Y[start_idx+self.window_size]
        else:
            x_new = self.X[start_idx:-1]
            y_new = self.Y[start_idx:-1]
            x_pred = self.X[-1] # last point
            y_actual = self.Y[-1]
        self.model.set_XY(X=x_new.reshape(-1, 1), Y=y_new.reshape(-1, 1))
        pred, pred_var, = self.predict(x_pred, y_actual, self.method)
        return pred, pred_var, y_actual, None

    def predict(self, x_pred, y, method='mc'):
        """
        Do prediction given x and y data
        :param x: Input data
        :param y: Input label (target)n
        :param method: marginalisation technique. mc=monte carlo, bmc=bayesian (grid) monte carlo, wsabi=wsabi quadrature
        :return: Tuple[prediction of next time step, run length mean, run length var]
        """
        return wsabi(x_pred, y, self.log_sample)

    def log_sample(self, phi:np.ndarray, x_pred: np.ndarray=None):
        #print(phi)
        phi = np.squeeze(phi)
        assert phi.ndim == 1, "ndim of phi must be 1!"
        self.model.chngpt.lengthscale1 = phi[0]
        self.model.chngpt.lengthscale2 = phi[1]
        self.model.chngpt.variance1 = phi[2]
        self.model.chngpt.variance2 = phi[3]
        self.model.chngpt.xc = phi[4]
        self.model.Gaussian_noise.variance = 1e-3
        #logging.info("Parameter changed")
        log_lik = self.model.log_likelihood()
        # display(self.model)
        if x_pred is None:
            return log_lik, None, None
        x_pred = x_pred.reshape(1, 1)
        pred, var = self.model.predict(Xnew=x_pred)
        # print(pred,var)
        return log_lik, pred, var


def wsabi(X_pred, y_grd, log_lik_handle, param_dim=5,
          prior_mean=np.zeros((5, 1)), prior_var=100*np.eye(5)):
        # Allocating number of maximum evaluations
        start = time.time()
        prior = Gaussian(mean=prior_mean.reshape(-1),covariance=prior_var)

        # Initial grid sampling
        log_phis = np.mgrid[-1:1.1:1, -1:1.1:1, -1:1.1:1, -1:1.1:1, 0:25:5].reshape(5, -1).T
        n = log_phis.shape[0]
        phis = log_phis.copy()
        phis[:, :-1] = np.exp(phis[:, :-1])

        # Allocate memory of the samples and results
        log_r = np.zeros((n, 1))  # The log-likelihood function
        q = np.zeros((n, 1)) # Prediction
        # var = np.zeros((n, ))  # Posterior variance

        for i in range(n):
            log_r[i, :], q[i, :], _ = log_lik_handle(phi=phis[i, :], x_pred=X_pred)
            print(phis[i, :], log_r[i, :], q[i, :])
        r = np.exp(log_r)
        # Setting up kernel - Note we only marginalise over the lengthscale terms, other hyperparameters are set to the
        # MAP values.
        kern = GPy.kern.RBF(param_dim,
                            variance=1.,
                            lengthscale=1.)
        # kern.plot(ax=plt.gca())
        r_gp = GPy.models.GPRegression(phis[:1,:], r[:1, :], kern)
        r_model = WarpedIntegrandModel(WsabiLGP(r_gp), prior)
        r_model.update(phis[1:, :], r[1:, :])
        r_gp.optimize()

        r_int = r_model.integral_mean()[0] # Model evidence
        log_r_int = np.log(r_int) # Model log-evidence

        print("Estimate of model evidence: ", r_int,)
        print("Model log-evidence ", log_r_int)

        # Enforce positivity in q
        q_min = np.min(q)
        if q_min < 0:
            q -= q_min
        else:
            q_min = 0

        # Do the same exponentiation and rescaling trick for q
        log_rq_x = log_r + np.log(q)
        max_log_rq = np.max(log_rq_x)
        rq = np.exp(log_rq_x - max_log_rq)
        rq_gp = GPy.models.GPRegression(phis, np.sqrt(2 * rq.reshape(-1, 1)), kern)
        rq_model = WarpedIntegrandModel(WsabiLGP(rq_gp), prior)
        rq_model.update(phis, rq)
        rq_gp.optimize()

        # Now estimate the posterior
        # rq_int = rq_model.integral_mean()[0] + q_min * r_int
        rq_int = np.exp(np.log(rq_model.integral_mean()[0]) + max_log_rq) + q_min * r_int
        print("rq_int", rq_int)
        # Similar for variance
        #log_rvar_x = log_r + np.log(var)
        #max_log_rvar = np.max(log_rvar_x)
        #rvar = np.exp(log_rvar_x - max_log_rvar)
        #rvar_gp = GPy.models.GPRegression(phis[:1, :], np.sqrt(2 * rvar[0].reshape(1, 1)), kern)
        #rvar_model = WarpedIntegrandModel(WsabiLGP(rvar_gp), prior)
        #rvar_model.update(phis[1:, :], rvar[1:].reshape(-1, 1))
        #rvar_gp.optimize()

        #rvar_int = np.exp(np.log(rvar_model.integral_mean()[0]) + max_log_rvar)

        pred = rq_int / r_int
        #pred_var = rvar_int / r_int
        print('pred', pred)
        print('actual', y_grd)

        end = time.time()
        print("Total Time: ", end-start)
        return pred, None


def do_experiment():
    n_iteration = 5
    cm = ChangepointModel()
    res = np.empty(n_iteration)
    true = np.empty(n_iteration)
    for i in range(n_iteration):
        tmp = cm.online_pred(i+25)
        res[i], true[i] = tmp[0], tmp[2],
    plt.plot(res)
    plt.plot(true)
    plt.show()


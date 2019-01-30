# Xingchen Wan | xingchen.wan@st-annes.ox.ac.uk | Jan 2019

from ratio.functions import RBFGPRegression
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


class RegressionQuadrature:
    def __init__(self, regression_model: RBFGPRegression, **kwargs):
        self.gpr = regression_model
        self.options = self._unpack_options(**kwargs)
        self.dimensions = self.gpr.param_dim

        # Parameter prior - note that the prior is in *log-space*
        self.prior = Gaussian(mean=self.options['prior_mean'].reshape(-1),
                              covariance=self.options['prior_variance'])
        self.gp = self.gpr.model

    def maximum_a_posterior(self, num_restarts=10, max_iters=1000, verbose=True):
        """
        Using a point estimate of the log-likelihood function
        :return: an array of predictions, y_pred
        """
        # Create a local version of the model
        self.gpr.reset_params(reset_gaussian_noise=True, reset_lengthscale=True, reset_variance=True)
        local_gp = self.gp.copy()

        # Fetch and train data from the GPy GP object
        X = self.gpr.X_train[:, :]
        Y = self.gpr.Y_train[:]
        X_test = self.gpr.X_test[:50, :]
        Y_test = self.gpr.Y_test[:50]
        local_gp.set_XY(X, Y)

        # Assign prior to the hyperparameters
        variance_prior = GPy.priors.LogGaussian(mu=0., sigma=2)
        lengthscale_prior = GPy.priors.LogGaussian(mu=0., sigma=2.)
        noise_prior = GPy.priors.LogGaussian(mu=0., sigma=2.)
        local_gp.kern.variance.set_prior(variance_prior)
        local_gp.kern.lengthscale.set_prior(lengthscale_prior)
        local_gp.Gaussian_noise.variance.set_prior(noise_prior)

        # Optimise under MAP
        local_gp.optimize_restarts(num_restarts=num_restarts, max_iters=max_iters)
        pred, pred_var = local_gp.predict(X_test)
        rmse = self.compute_rmse(pred, Y_test)
        ll, cs = self.compute_ll_cs(pred, pred_var, Y_test)
        print('Root Mean Squared Error:', rmse)
        print('Log-likelihood:', ll)
        print('Calibration Score:', cs)
        # print('MAP lengthscale vector:', local_gp.rbf.lengthscale)
        if verbose:
            display(local_gp)
            self.visualise(pred, pred_var, Y_test)
        return local_gp, rmse

    def mc(self, verbose=True):
        """
        Monte carlo marginalisation
        The algorithm first samples from the likelihood surface, then use the different hyperparameter vectors obtained
        to obtain predictions given data. Then we average over the different predictions to give the final result.
        :param verbose: toggle whether to display a graph of the predicted values vs the ground truth
        :return: float: root mean squared error
        """
        budget = self.options['smc_budget']
        test_x = self.gpr.X_test[:50, :]
        test_y = self.gpr.Y_test[:50]
        test_length = test_x.shape[0]

        # Fetch MAP values of variance and Gaussian noise
        map_model, _ = self.maximum_a_posterior(num_restarts=1, verbose=False)
        variance_map = map_model.rbf.variance
        gaussian_noise_map = map_model.Gaussian_noise.variance
        self.gpr.reset_params()
        self.gpr.set_params(variance=variance_map, gaussian_noise=gaussian_noise_map)

        # Sample the parameter posterior, note that we discard the first column (variance) and last column
        # (gaussian noise) since we are interested in marginalising the lengthscale hyperparameters only
        samples = sample_from_param_posterior(self.gpr.model, 1000, 1, 'hmc', False)
        samples = samples[:, 1:-1]
        logging.info("Parameter posterior sampling completed")

        # Now draw samples from the parameter posterior distribution and do Monte Carlo integration
        y_preds = np.zeros((budget,test_length))
        y_preds_var = np.zeros((budget, test_length))

        for i in range(budget):
            self.gpr.set_params(samples[i, :])
            tmp_res = self.gpr.model.predict_noiseless(test_x)
            y_preds[i, :] = (tmp_res[0]).reshape(-1)
            y_preds_var[i, :] = (tmp_res[1]).reshape(-1)
            logging.info("Progress: "+str(i)+" / "+str(budget))
        pred = y_preds.sum(axis=0) / budget
        pred_var = y_preds_var.sum(axis=0) / budget
        rmse = self.compute_rmse(pred, test_y)
        ll, cs = self.compute_ll_cs(pred, pred_var, test_y)
        print("Root Mean Squared Error", rmse)
        print("Log-likelihood", ll)
        print("Calibration Score", cs)
        if verbose:
            self.visualise(pred, pred_var, test_y)
        return rmse, ll

    def bq(self, verbose=True):
        """
        Marginalisation using vanilla Bayesian Quadrature - we use Amazon Emukit interface for this purpose
        :return:
        """

        def _rp_emukit(x: np.ndarray) -> np.ndarray:
            n, d = x.shape
            res = np.exp(self.gpr.log_sample(phi=np.exp(x))[0] + np.log(self.prior(x)))
            return np.array(res).reshape(n, 1)

        def rp_emukit():
            # Wrap around Emukit interface
            from emukit.core.loop.user_function import UserFunctionWrapper
            return UserFunctionWrapper(_rp_emukit), _rp_emukit
        start = time.time()

        budget = self.options['naive_bq_budget']
        test_x = self.gpr.X_test[:50, :]
        test_y = self.gpr.Y_test[:50]

        q = np.zeros((test_x.shape[0], budget+1))
        var = np.zeros((test_x.shape[0], budget+1))

        # Initial points - note that as per GPML convention, the hyperparameters are expressed in log scale
        # Initialise to the MAP estimate
        map_model, _ = self.maximum_a_posterior(num_restarts=1, max_iters=500, verbose=False)
        self.gpr.reset_params()
        self.gpr.set_params(variance=map_model.rbf.variance, gaussian_noise=map_model.Gaussian_noise.variance)
        log_phi_initial = np.zeros((1, self.dimensions))
        r_initial = np.exp(self.gpr.log_sample(phi=np.exp(log_phi_initial))[0] + np.log(self.prior(log_phi_initial)))
        pred = np.zeros((test_x.shape[0], ))
        var_pred = np.zeros((test_x.shape[0], ))

        # Setting up kernel - Note we only marginalise over the lengthscale terms, other hyperparameters are set to the
        # MAP values.
        kern = GPy.kern.RBF(self.dimensions,
                            variance=2.,
                            lengthscale=2.)

        r_gp = GPy.models.GPRegression(log_phi_initial, r_initial.reshape(1, -1), kern)
        r_model = self._wrap_emukit(r_gp)
        r_loop = VanillaBayesianQuadratureLoop(model=r_model)

        # Firstly, within the given allowance, compute an estimate of the model evidence. Model evidence is the common
        # denominator for all predictive distributions.
        r_loop.run_loop(user_function=rp_emukit()[0], stopping_condition=budget)
        log_phi = r_loop.loop_state.X
        r = r_loop.loop_state.Y.reshape(-1)

        quad_time = time.time()

        r_int = r_model.integrate()[0]  # Model evidence
        print("Estimate of model evidence: ", r_int, )
        print("Model log-evidence ", np.log(r_int))

        for i_x in range(test_x.shape[0]):

            # Note that we do not active sample again for q, we just use the same samples sampled when we compute
            # the log-evidence
            _, q_initial, var_initial = self.gpr.log_sample(phi=np.exp(log_phi_initial), x=test_x[i_x, :])

            # Initialise GPy GP surrogate for and q(\phi)r(\phi)
            # Sample for q values
            q[i_x, 0] = q_initial
            var[i_x, 0] = var_initial
            for i_b in range(1, budget+1):
                log_phi_i = log_phi[i_b, :]
                _, q_i, var_i = self.gpr.log_sample(phi=np.exp(log_phi_i), x=test_x[i_x, :])
                q[i_x, i_b] = q_i
                var[i_x, i_b] = var_i
            # Construct rq vector
            q_x = q[i_x, :]
            var_x = var[i_x, :]

            rq = r * q_x
            rq_gp = GPy.models.GPRegression(log_phi, rq.reshape(-1, 1), kern)
            rq_model = self._wrap_emukit(rq_gp)
            rq_int = rq_model.integrate()[0]

            rvar = r * var_x
            rvar_gp = GPy.models.GPRegression(log_phi, rvar.reshape(-1, 1), kern)
            rvar_model = self._wrap_emukit(rvar_gp)
            rvar_int = rvar_model.integrate()[0]

            # Now estimate the posterior

            pred[i_x] = rq_int / r_int
            var_pred[i_x] = rvar_int / r_int

            logging.info('Progress: '+str(i_x+1)+'/'+str(test_x.shape[0]))

        rmse = self.compute_rmse(pred, test_y)
        ll, cs = self.compute_ll_cs(pred, var_pred, test_y)
        logging.info(pred, test_y)
        print('Root Mean Squared Error:', rmse)
        print('Log-likelihood:', ll)
        print('Calibration Score:', cs)
        end = time.time()
        print("Active Sampling Time: ", quad_time-start)
        print("Total Time elapsed: ", end-start)
        if verbose:
            self.visualise(pred, var_pred, test_y)
        return rmse, ll, quad_time-start

    def wsabi(self, verbose=True):
        # Allocating number of maximum evaluations
        start = time.time()
        budget = self.options['wsabi_budget']
        batch_count = 1
        test_x = self.gpr.X_test[:50, :]
        test_y = self.gpr.Y_test[:50]

        # Allocate memory of the samples and results
        log_phi = np.zeros((budget*batch_count, self.dimensions,))  # The log-hyperparameter sampling points
        log_r = np.zeros((budget*batch_count, ))  # The log-likelihood function
        q = np.zeros((test_x.shape[0], budget*batch_count))  # Prediction
        var = np.zeros((test_x.shape[0], budget*batch_count))  # Posterior variance

        # Initial points - note that as per GPML convention, the hyperparameters are expressed in log scale
        # Initialise to the MAP estimate
        map_model, _ = self.maximum_a_posterior(num_restarts=1, max_iters=500, verbose=False)
        self.gpr.reset_params()
        self.gpr.set_params(variance=map_model.rbf.variance, gaussian_noise=map_model.Gaussian_noise.variance)
        # log_phi_initial = np.log(map_model.rbf.lengthscale).reshape(1, -1)
        log_phi_initial = np.zeros((1, self.dimensions))
        log_r_initial = self.gpr.log_sample(phi=np.exp(log_phi_initial))[0]
        pred = np.zeros((test_x.shape[0], ))
        pred_var = np.zeros((test_x.shape[0], ))

        # Setting up kernel - Note we only marginalise over the lengthscale terms, other hyperparameters are set to the
        # MAP values.
        kern = GPy.kern.RBF(self.dimensions,
                            variance=2.,
                            lengthscale=2.)

        log_r_gp = GPy.models.GPRegression(log_phi_initial, log_r_initial.reshape(1, -1), kern)
        log_r_model = WarpedIntegrandModel(WsabiLGP(log_r_gp), self.prior)

        # Firstly, within the given allowance, compute an estimate of the model evidence. Model evidence is the common
        # denominator for all predictive distributions.
        for i_a in range(budget):
            log_phi_i = np.array(select_batch(log_r_model, batch_count, "Kriging Believer")).reshape(batch_count, -1)
            log_r_i = self.gpr.log_sample(phi=np.exp(log_phi_i))[0]
            log_r[i_a:i_a+batch_count] = log_r_i
            log_phi[i_a:i_a+batch_count, :] = log_phi_i
            log_r_model.update(log_phi_i, log_r_i.reshape(1, -1))

        quad_time = time.time()

        max_log_r = max(log_r)
        r = np.exp(log_r - max_log_r)
        r_gp = GPy.models.GPRegression(log_phi[:1, :], np.sqrt(2 * r[0].reshape(1, 1)), kern)
        r_model = WarpedIntegrandModel(WsabiLGP(r_gp), self.prior)
        r_model.update(log_phi[1:, :], r[1:].reshape(-1, 1))
        r_gp.optimize()
        r_int = np.exp(np.log(r_model.integral_mean()[0]) + max_log_r) # Model evidence
        log_r_int = np.log(r_int) # Model log-evidence

        print("Estimate of model evidence: ", r_int,)
        print("Model log-evidence ", log_r_int)

        # Visualise the model parameter posterior
        # neg_log_post = np.array((budget, )) # Negative log-posterior
        # rp = np.array((budget, ))
        # for i in range(budget):
        #    neg_log_post[i] = (log_r[i] + self.prior.log_eval(log_phi[i, :]) - log_r_int)
        # Then train a GP for the log-posterior surface
        # log_posterior_gp = GPy.models.GPRegression(np.exp(log_phi), np.exp(neg_log_post).reshape(-1, 1), kern)

        # Secondly, compute and marginalise the predictive distribution for each individual points
        for i_x in range(test_x.shape[0]):

            # Note that we do not active sample again for q, we just use the same samples sampled when we compute
            # the log-evidence
            _, q_initial, var_initial = self.gpr.log_sample(phi=np.exp(log_phi_initial), x=test_x[i_x, :])

            # Initialise GPy GP surrogate for and q(\phi)r(\phi)
            # Sample for q values
            for i_b in range(budget*batch_count):
                log_phi_i = log_phi[i_b, :]
                log_r_i, q_i, var_i = self.gpr.log_sample(phi=np.exp(log_phi_i), x=test_x[i_x, :])
                q[i_x, i_b] = q_i
                var[i_x, i_b] = var_i

            # Enforce positivity in q
            q_x = q[i_x, :]
            var_x = var[i_x, :]
            q_min = np.min(q_x)
            if q_min < 0:
                q_x = q_x - q_min
            else:
                q_min = 0

            # Do the same exponentiation and rescaling trick for q
            log_rq_x = log_r + np.log(q_x)
            max_log_rq = np.max(log_rq_x)
            rq = np.exp(log_rq_x - max_log_rq)

            rq_gp = GPy.models.GPRegression(log_phi[:1, :], np.sqrt(2 * rq[0].reshape(1, 1)), kern)
            rq_model = WarpedIntegrandModel(WsabiLGP(rq_gp), self.prior)
            rq_model.update(log_phi[1:, :], rq[1:].reshape(-1, 1))
            rq_gp.optimize()

            # Now estimate the posterior
            rq_int = np.exp(np.log(rq_model.integral_mean()[0]) + max_log_rq) + q_min * r_int

            # Similar for variance
            log_rvar_x = log_r + np.log(var_x)
            max_log_rvar = np.max(log_rvar_x)
            rvar = np.exp(log_rvar_x - max_log_rvar)
            rvar_gp = GPy.models.GPRegression(log_phi[:1, :], np.sqrt(2 * rvar[0].reshape(1, 1)), kern)
            rvar_model = WarpedIntegrandModel(WsabiLGP(rvar_gp), self.prior)
            rvar_model.update(log_phi[1:, :], rvar[1:].reshape(-1, 1))
            rvar_gp.optimize()

            rvar_int = np.exp(np.log(rvar_model.integral_mean()[0]) + max_log_rvar)

            pred[i_x] = rq_int / r_int
            pred_var[i_x] = rvar_int / r_int

            logging.info('Progress: '+str(i_x+1)+'/'+str(test_x.shape[0]))

        rmse = self.compute_rmse(pred, test_y)
        ll, cs = self.compute_ll_cs(pred, pred_var, test_y)
        print('Root Mean Squared Error:', rmse)
        print('Log-likelihood', ll)
        print('Calibration score', cs)
        end = time.time()
        print("Active Sampling Time: ", quad_time-start)
        print("Total Time: ", end-start)
        if verbose:
            logging.info(pred, test_y)
            self.visualise(pred, pred_var, test_y)
        return rmse, ll, quad_time-start

    def wsabi_ratio(self):

        # Implementation of the Bayesian Quadrature for Ratio paper

        def unwarp(warped_y: np.ndarray, alpha: float):
            """
            Unwarp the output of a WSABI GP
            :param warped_y:
            :return:
            """
            return 0.5 * (warped_y ** 2) + alpha

        def get_phi_s(phi_c, method='latin'):
            """
            Construct a Voronoi diagram and identify the locations of phi_s
            (See Bayesian Quadrature for Ratio paper for reasoning)
            Method: 'latin' - latin hypercube sampling; 'voronoi' - voronoi diagram (caution: this method scales very
            poorly in higher dimension!)
            """
            # todo: implement the Latin Hypercube Sampling here!
            from scipy.spatial import Voronoi
            if method == 'voronoi':
                vor = Voronoi(phi_c)
                candidates = vor.vertices

                ub = np.amax(phi_c, axis=0) + 1  # Upper bound
                lb = np.amin(phi_c, axis=0) - 1  # Lower bound
                ub = np.tile(ub, (candidates.shape[0], 1))
                candidates = candidates[~np.any(ub - candidates < 0, axis=1)]
                lb = np.tile(lb, (candidates.shape[0], 1))
                candidates = candidates[~np.any(candidates - lb < 0, axis=1)]
            elif method == 'latin':
                raise NotImplemented()
            else:
                raise NotImplemented()
            logging.info(str(candidates.shape[0])+" candidate phi_s points returned.")
            return candidates

        # 1.1 Allocating number of maximum evaluations
        budget = self.options['wsabi_budget']
        batch_count = 1
        test_x = self.gpr.X_test[:, :]
        test_y = self.gpr.Y_test[:]

        # 1.2 Allocate memory of the samples and results
        log_phi_c = np.zeros((budget * batch_count, self.dimensions,))  # The log-hyperparameter sampling points
        log_r = np.zeros((budget * batch_count,))  # The log-likelihood function
        q = np.zeros((test_x.shape[0], budget * batch_count))  # Prediction

        # 2. Initial points - note that as per GPML convention, the hyperparameters are expressed in log scale
        # 2.1 Initialise to the MAP estimate
        map_model, _ = self.maximum_a_posterior(num_restarts=1, max_iters=500, verbose=False)
        self.gpr.set_params(variance=map_model.rbf.variance, gaussian_noise=map_model.Gaussian_noise.variance)
        log_phi_initial = np.log(map_model.rbf.lengthscale).reshape(1, -1)
        log_r_initial = self.gpr.log_sample(phi=np.exp(log_phi_initial))[0]
        pred = np.zeros((test_x.shape[0],))

        # 2.2 Setting up kernel - Note we only marginalise over the lengthscale terms, other hyperparameters are set
        # to the MAP values.
        kern = GPy.kern.RBF(self.dimensions,
                            variance=2.,
                            lengthscale=2.)

        log_r_gp = GPy.models.GPRegression(log_phi_initial, log_r_initial.reshape(1, -1), kern)
        log_r_model = WarpedIntegrandModel(WsabiLGP(log_r_gp), self.prior)

        # 3.1, within the given allowance, compute an estimate of the model evidence. Model evidence is the common
        # denominator for all predictive distributions.
        for i_a in range(budget):
            log_phi_i = np.array(select_batch(log_r_model, batch_count, )).reshape(batch_count, -1)
            log_r_i = self.gpr.log_sample(phi=np.exp(log_phi_i))[0]

            log_r[i_a:i_a + batch_count] = log_r_i
            log_phi_c[i_a:i_a + batch_count, :] = log_phi_i

            log_r_model.update(log_phi_i, log_r_i.reshape(1, -1))

        # 4. Obtain phi_s locations
        log_phi_s = get_phi_s(log_phi_c)

        # 5. Rescale the original model
        max_log_r = np.max(log_r)
        r = np.exp(log_r - max_log_r)
        r_gp = GPy.models.GPRegression(log_phi_c[:1], np.sqrt(2 * r[0].reshape(-1, 1)), kern)
        warped_r = WsabiLGP(r_gp)
        r_model = WarpedIntegrandModel(warped_r, self.prior)
        r_model.update(log_phi_c[1:], r[1:].reshape(-1, 1))
        r_gp.optimize()
        m_int_r = np.exp(np.log(r_model.integral_mean()[0]) + max_log_r)  # Model evidence
        log_m_int_r = np.log(m_int_r)  # Model log-evidence
        print("Estimate of model evidence: ", m_int_r, )
        print("Model log-evidence ", log_m_int_r)

        # 6. Compute the eta_{rr} vector on log_phi_s
        m_rs = unwarp(r_gp.predict_noiseless(log_phi_s)[0], warped_r._alpha)
        # Posterior mean of the GP over original r at phi_s points
        log_m_rs = np.log(m_rs) + max_log_r  # Log of the posterior mean of GP over original r
        log_m_rs[log_m_rs == -np.inf] = max_log_r
        log_r_gp = GPy.models.GPRegression(log_phi_c, log_r.reshape(-1, 1),)
        log_r_gp.optimize()
        m_log_rs = log_r_gp.predict_noiseless(log_phi_s)[0]  # Posterior mean of the GP over log(r)
        # print(m_log_rs)
        eta_rr = m_rs * (m_log_rs - log_m_rs)
        # Concatenate phi_c and 0 to the above vector
        eta_rr_concat = np.concatenate((eta_rr, np.zeros((log_phi_c.shape[0], 1))))
        log_phi_concat = np.concatenate((log_phi_s, log_phi_c))

        # 6.1 Similar to r, compute the integral mean of eta_{rr}
        eta_rr_gp = GPy.models.GPRegression(log_phi_concat, eta_rr_concat.reshape(-1, 1), kern)
        logging.info("eta_{rr} GP object initialised")
        m_int_eta_rr = self._wrap_emukit(eta_rr_gp)[0]
        logging.info("eta_{rr} estimate "+str(m_int_eta_rr))

        # 7, compute and marginalise the predictive distribution for each individual points
        for i_x in range(test_x.shape[0]):

            # Note that we do not active sample again for q, we just use the same samples sampled when we compute
            # the log-evidence
            _, q_initial = self.gpr.log_sample(phi=np.exp(log_phi_initial), x=test_x[i_x, :])

            # 7.1 Initialise GPy GP surrogate for and q(\phi)r(\phi)
            # Sample for q values
            for i_b in range(budget * batch_count):
                log_phi_i = log_phi_c[i_b, :]
                neg_log_r_i, q_i = self.gpr.log_sample(phi=np.exp(log_phi_i), x=test_x[i_x, :])
                q[i_x, i_b] = q_i

            # 8. Formulate relevant integrals and evaluate the integrals
            q_gp = GPy.models.GPRegression(log_phi_c, q[i_x, :].reshape(-1, 1), kern)
            q_gp.optimize()
            # 8.1 Evaluate rho_{0}

            rq = q[i_x, :] * r

            rq_gp = GPy.models.GPRegression(log_phi_c, rq.reshape(-1, 1), kern)
            m_int_rq = self._wrap_emukit(rq_gp)[0] * np.exp(max_log_r)
            rho_0 = m_int_rq / m_int_r

            # 8.2 Evaluate C_{q} - Adjustment factor 1
            m_qs = q_gp.predict_noiseless(log_phi_s)[0].reshape(-1)
            m_qs = np.concatenate((q[i_x, :], m_qs)).reshape(-1, 1)

            # 8.3 Evaluate C_{r} - Adjustment factor 2
            q_eta_rr = m_qs * eta_rr_concat
            # This quantity is very small; to translation and scaling to prevent numerical issues
            q_eta_rr_min = np.min(q_eta_rr)
            if q_eta_rr_min < 0:
                q_eta_rr -= q_eta_rr_min
            else:
                q_eta_rr_min = 0
            log_q_eta_rr = np.log(q_eta_rr)
            max_log_q_eta_rr = np.max(log_q_eta_rr)
            q_eta_rr = np.exp(log_q_eta_rr - max_log_q_eta_rr)

            q_eta_rr_gp = GPy.models.GPRegression(log_phi_concat, q_eta_rr.reshape(-1, 1),)
            m_int_q_eta_rr = np.exp(np.log(self._wrap_emukit(q_eta_rr_gp)[0]) + max_log_q_eta_rr) + \
                             q_eta_rr_min * m_int_r
            c_r = 1. / m_int_r * (m_int_q_eta_rr - m_int_eta_rr * (m_int_rq / m_int_r))
            print(m_int_r, m_int_q_eta_rr, m_int_eta_rr, m_int_rq)
            exit()
            pred[i_x] = rho_0 + c_r
            print('Progress: ', i_x + 1, '/', test_x.shape[0])

        rmse = self.compute_rmse(pred, test_y)
        print(pred, test_y)
        self.visualise(pred, test_y)
        print('Root Mean Squared Error:', rmse)
        return rmse

    # ----- Evaluation Metric ---- #

    @staticmethod
    def compute_rmse(y_pred: np.ndarray, y_grd: np.ndarray) -> float:
        """
        Compute the root mean squared error between prediction (y_pred) with the ground truth y in the test set
        :param y_pred: a vector with the same length as the test set y
        :return: rmse
        """
        if y_grd is None:
            logger.warning("Ground truth is not provided - unable to produce RMSE result")
            return np.nan
        y_pred = y_pred.reshape(-1)
        y_grd = y_grd.reshape(-1)
        length = y_pred.shape[0]
        assert length == y_grd.shape[0], "the length of the prediction vector does " \
                                         "not match the ground truth vector!"
        rmse = np.sqrt(np.sum((y_pred - y_grd) ** 2) / length)
        return rmse

    @staticmethod
    def compute_ll_cs(y_pred: np.ndarray, y_var: np.ndarray, y_grd: np.ndarray) -> Tuple[float, float]:
        """
        Compute log-likelihood and calibration score using predictive mean, variance and the ground labels
        :param y_pred:
        :param y_var:
        :param y_grd:
        :return:
        """
        from scipy.stats import norm
        if y_grd is None:
            logger.warning("Ground truth is not provided - unable to produce LL and CS results")
            return np.nan, np.nan
        ll = 0.
        y_pred = y_pred.reshape(-1)
        y_var = y_var.reshape(-1)
        y_grd = y_grd.reshape(-1)
        cs = 0
        for i in range(y_pred.shape[0]):
            ll += norm.logpdf(y_grd[i], loc=y_pred[i], scale=np.sqrt(y_var[i]))
            ub = y_pred[i] + 0.67449 * np.sqrt(y_var[i])
            lb = y_pred[i] - 0.67449 * np.sqrt(y_var[i])
            if y_grd[i] >= lb and y_grd[i] <= ub:
                cs += 1
        return ll, cs / y_pred.shape[0]

    def _wrap_emukit(self, gpy_gp: GPy.core.GP):
        """
        Wrap GPy GP around Emukit interface to enable subsequent quadrature
        :param gpy_gp:
        :return:
        """
        # gpy_gp.optimize()
        rbf = RBFGPy(gpy_gp.kern)
        qrbf = QuadratureRBF(rbf, integral_bounds=[(-10.,10.)] * self.dimensions)
        model = BaseGaussianProcessGPy(kern=qrbf, gpy_model=gpy_gp)
        method = VanillaBayesianQuadrature(base_gp=model)
        return method

    def visualise(self, y_pred: np.ndarray, y_var: np.ndarray, y_grd: np.ndarray):
        if self.dimensions > 2:
            # For higher dimension, we only show the comparison between the test data and the ground truth
            xv = list(range(50))
            plt.errorbar(xv, y_pred, yerr=np.sqrt(y_var), fmt='.', ecolor='r', capsize=2, label='Predictions +/- 1SD')
            plt.plot(y_grd, ".", color='red', label='Ground Truth')
            plt.legend()
            plt.xlabel("Sample Number")
            plt.ylabel("Predicted/Actual Value")
        else:
            # For lower dimensions, we can show the things such as the contour plot of the likelihood surface, the
            # posterior and include a plot between independent and dependent variables.
            raise NotImplemented
        plt.show()

    def _unpack_options(self,
                        prior_mean: Union[float, np.ndarray] = 0.,
                        prior_variance: Union[float, np.ndarray] = 2.,
                        smc_budget: int = 1000,
                        naive_bq_budget: int = 300,
                        naive_bq_kern_lengthscale: float = 1.,
                        naive_bq_kern_variance: float = 1.,
                        wsabi_budget: int = 100,
                        posterior_range=(-5., 5.),
                        # The axis ranges between which the posterior samples are drawn
                        posterior_eps: float = 0.02,
                        ):
        if self.dimensions == 1:
            prior_mean = np.array([prior_mean]).reshape(1, 1)
            prior_variance = np.array([[prior_variance]]).reshape(1, 1)
        elif self.dimensions > 1 and isinstance(prior_variance, float) and isinstance(prior_mean, float):
            prior_mean = np.array([prior_mean] * (self.dimensions)).reshape(-1, 1)
            prior_variance *= np.eye(self.dimensions)
        else:
            assert len(prior_mean) == self.dimensions
            assert prior_variance.shape[0] == prior_variance.shape[1]
            assert prior_variance.shape[0] == self.dimensions
        return {
            'prior_mean': prior_mean,
            'prior_variance': prior_variance,
            'smc_budget': smc_budget,
            'naive_bq_budget': naive_bq_budget,
            'naive_bq_kern_lengthscale': naive_bq_kern_lengthscale,
            'naive_bq_kern_variance': naive_bq_kern_variance,
            'wsabi_budget': wsabi_budget,
            'posterior_range': posterior_range,
            'posterior_eps': posterior_eps
        }


# -------- Parameter Posterior -------- #
def sample_from_param_posterior(gpy_gp, num_samples=1000, mc_iters=10,
                                mc_method='smc',
                                save_csv=True, plot_samples=True):
    if mc_method == 'smc':
        mc = GPy.inference.mcmc.Metropolis_Hastings(gpy_gp)
        t = mc.sample(Ntotal=num_samples)
    elif mc_method == 'hmc':
        mc = GPy.inference.mcmc.HMC(gpy_gp, stepsize=2e-2)
        t = mc.sample(num_samples=num_samples, hmc_iters=mc_iters)
    else:
        raise NotImplemented
    if plot_samples:
        _ = plt.plot(t)
        plt.show()
    if save_csv:
        df = pd.DataFrame(t, columns=gpy_gp.parameter_names_flat())
        df.to_csv('PosteriorSampling.csv')
    return t


# ---- Performance evaluation --------- #
def eval_perf(rq: RegressionQuadrature, method: str):
    sample_n = [10, 15, 20, 30, 50, 75, 100, 120, 150, 200, 300]
    res = np.empty((len(sample_n), 3))
    for i in range(len(sample_n)):
        if method == 'wsabi':
            rq.options['wsabi_budget'] = sample_n[i]
            res[i, 0], res[i, 1], res[i, 2] = rq.wsabi(verbose=False)
        elif method == 'bq':
            rq.options['naive_bq_budget'] = sample_n[i]
            res[i, 0], res[i, 1], res[i, 2] = rq.bq(verbose=False)
        elif method == 'mc':
            rq.options['smc_budget'] = sample_n[i]
            res[i, 0], res[i, 1] = rq.mc(verbose=False)
        else:
            raise NotImplemented()
    df = pd.DataFrame(res, columns=['rmse', 'll', 'quad_time'])
    df.to_csv(method+'_perf.csv')
    logging.info(method+' evaluation completed')

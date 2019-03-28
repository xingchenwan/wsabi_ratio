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
import seaborn as sns

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
    def __init__(self, regression_model: Union[RBFGPRegression,PeriodicGPRegression], **kwargs):
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
        X_test = self.gpr.X_test[:, :]
        Y_test = self.gpr.Y_test[:]
        local_gp.set_XY(X, Y)

        # Assign prior to the hyperparameters
        variance_prior = GPy.priors.LogGaussian(mu=0., sigma=4)
        lengthscale_prior = GPy.priors.LogGaussian(mu=0., sigma=4.)
        noise_prior = GPy.priors.LogGaussian(mu=-4., sigma=4.)
        if isinstance(self.gpr, PeriodicGPRegression):
            period_prior = GPy.priors.LogGaussian(mu=5., sigma=4.)

        local_gp.kern.variance.set_prior(variance_prior)
        local_gp.kern.lengthscale.set_prior(lengthscale_prior)
        local_gp.Gaussian_noise.variance.set_prior(noise_prior)
        if isinstance(self.gpr, PeriodicGPRegression):
            local_gp.kern.period.set_prior(period_prior)


        # Optimise under MAP
        local_gp.optimize_restarts(num_restarts=num_restarts, max_iters=max_iters)
        pred, pred_var = local_gp.predict(X_test)
        posteriors = np.squeeze(local_gp.posterior_samples_f(X, size=1), axis=1)
        # plt.plot(X, posteriors, alpha=0.2, color='r', marker='.', linestyle='None', label='MAP GP Posterior')
        #print(pred_var)
        rmse = self.compute_rmse(pred, Y_test)
        ll, cs = self.compute_ll_cs(pred, pred_var, Y_test)
        print('Root Mean Squared Error:', rmse)
        print('Log-likelihood:', ll)
        print('Calibration Score:', cs)
        # print('MAP lengthscale vector:', local_gp.rbf.lengthscale)
        display(local_gp)
        if verbose:
            self.visualise(pred, pred_var, Y_test)
        return local_gp, rmse, ll

    def mc(self, verbose=True):
        """
        Monte carlo marginalisation
        The algorithm first samples from the likelihood surface, then use the different hyperparameter vectors obtained
        to obtain predictions given data. Then we average over the different predictions to give the final result.
        :param verbose: toggle whether to display a graph of the predicted values vs the ground truth
        :return: float: root mean squared error
        """
        from ratio.posterior_mc_inference import PosteriorMCSampler, PyMC3GP

        budget = self.options['smc_budget']
        test_x = self.gpr.X_test
        test_y = self.gpr.Y_test
        test_length = test_x.shape[0]

        # Fetch MAP values of variance and Gaussian noise
        map_model, _, _ = self.maximum_a_posterior(num_restarts=1, verbose=False)
        # variance_map = map_model.rbf.variance
        # gaussian_noise_map = map_model.Gaussian_noise.variance
        # self.gpr.reset_params()
        # self.gpr.set_params(variance=variance_map, gaussian_noise=gaussian_noise_map)

        # Sample the parameter posterior, note that we discard the first column (variance) and last column
        # (gaussian noise) since we are interested in marginalising the length-scale hyperparameters only

        #sampler = PosteriorMCSampler(map_model)
        pygp = PyMC3GP(self.gpr.X_train, self.gpr.Y_train)
        samples = pygp.build()
        #samples = sampler.hmc(num_iters=int(budget*1.25), mode='gpflow')
        # Toggle this li ne on when sampling using GPFlow, for it returns a pandas dataframe rather than a numpy array.
        #samples = samples.values[:, :4]

        # samples = sample_from_param_posterior(self.gpr.model, 1000, 1, 'hmc', False)
        samples = samples[int(budget*0.25):, :]
        logging.info("Parameter posterior sampling completed")

        # Now draw samples from the parameter posterior distribution and do Monte Carlo integration
        y_preds = np.zeros((budget,test_length))
        y_preds_var = np.zeros((budget, test_length))

        for i in range(budget):
            if isinstance(self.gpr, PeriodicGPRegression):
                try:
                    # print(samples)
                    self.gpr.set_params(lengthscale=samples[i, 0], period=samples[i, 1],
                                    variance=samples[i, 2], gaussian_noise=samples[i, 3])
                except np.linalg.LinAlgError:
                    print('LinAlg Error')
                    pass
            else:
                self.gpr.set_params(samples[i, :])
            tmp_res = self.gpr.model.predict(test_x)
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

        return rmse, ll, None, None

    def bmc(self):
        # Allocating number of maximum evaluations
        start = time.time()
        budget = self.options['wsabi_budget']
        batch_count = 1
        test_x = self.gpr.X_test
        test_y = self.gpr.Y_test

        # Allocate memory of the samples and results
        samples = np.zeros((budget * batch_count, self.dimensions,))  # The log-hyperparameter sampling points
        log_r = np.zeros((budget * batch_count,))  # The log-likelihood function

        # Initial points - note that as per GPML convention, the hyperparameters are expressed in log scale
        # Initialise to the MAP estimate
        map_model, _, _ = self.maximum_a_posterior(num_restarts=1, max_iters=1000, verbose=False)
        display(map_model)
        # mean_vals = map_model.param_array
        # self.gpr.reset_params()
        if isinstance(self.gpr, PeriodicGPRegression):
            pass
            # self.gpr.set_params(variance=map_model.std_periodic.variance,
            #                    gaussian_noise=map_model.Gaussian_noise.variance)
        elif isinstance(self.gpr, RBFGPRegression):
            self.gpr.set_params(variance=map_model.rbf.variance, gaussian_noise=map_model.Gaussian_noise.variance)

        # Set prior mean to the MAP value
        # self.prior = Gaussian(mean=mean_vals.reshape(-1), covariance=self.options['prior_variance'])

        log_phi_initial = self.options['prior_mean'].reshape(1, -1)
        log_r_initial = np.sqrt(2 * np.exp(self.gpr.log_sample(
            phi=np.exp(log_phi_initial.reshape(-1))
        )[0]))
        # log_r_initial = self.gpr.log_sample(
        #    phi=np.exp(log_phi_initial.reshape(-1)))[0]

        pred = np.zeros((budget, test_x.shape[0]))
        pred_var = np.zeros((budget, test_x.shape[0],))

        # Setting up kernel - Note we only marginalise over the lengthscale terms, other hyperparameters are set to the
        # MAP values.
        kern = GPy.kern.RBF(self.dimensions,
                            variance=1.,
                            lengthscale=1)

        log_r_gp = GPy.models.GPRegression(log_phi_initial, log_r_initial.reshape(1, -1), kern)
        log_r_model = WarpedIntegrandModel(WsabiLGP(log_r_gp), self.prior)

        # Firstly, within the given allowance, compute an estimate of the model evidence. Model evidence is the common
        # denominator for all predictive distributions.
        for i_a in range(budget - 1):
            log_phi_i = np.array(select_batch(log_r_model, batch_count, "Kriging Believer")).reshape(batch_count, -1)
            try:
                log_r_i = self.gpr.log_sample(phi=np.exp(log_phi_i))[0]
                log_r[i_a:i_a + batch_count] = log_r_i
                samples[i_a:i_a + batch_count, :] = log_phi_i
                log_r_model.update(log_phi_i, np.exp(log_r_i).reshape(1, -1))
                print(np.exp(log_phi_i), log_r_i)
            except np.linalg.linalg.LinAlgError:
                continue

        for i in range(budget):
            if isinstance(self.gpr, PeriodicGPRegression):
                try:
                    # print(samples)
                    self.gpr.set_params(lengthscale=np.exp(samples[i, 0]), period=np.exp(samples[i, 1]),
                                        variance=np.exp(samples[i, 2]), gaussian_noise=np.exp(samples[i, 3]))
                except np.linalg.LinAlgError:
                    print('LinAlg Error')
                    pass
            else:
                self.gpr.set_params(samples[i, :])
            tmp_res = self.gpr.model.predict(test_x)
            pred[i, :] = (tmp_res[0]).reshape(-1)
            pred_var[i, :] = (tmp_res[1]).reshape(-1)
            logging.info("Progress: " + str(i) + " / " + str(budget))

        pred = pred[50:, :]
        pred_var = pred_var[50:, :]
        pred = pred.sum(axis=0) / (budget - 50)
        pred_var = pred_var.sum(axis=0) / (budget - 50)
        rmse = self.compute_rmse(pred, test_y)
        ll, cs = self.compute_ll_cs(pred, pred_var, test_y)
        print("Root Mean Squared Error", rmse)
        print("Log-likelihood", ll)
        print("Calibration Score", cs)
        self.visualise(pred, pred_var, test_y)

        return rmse, ll, None, None

    def bq(self, verbose=True):
        """
        Marginalisation using vanilla Bayesian Quadrature - we use Amazon Emukit interface for this purpose
        :return:
        """

        def _rp_emukit(x: np.ndarray) -> np.ndarray:
            n, d = x.shape
            res = np.exp(self.gpr.log_sample(phi=np.exp(x))[0] + np.log(self.prior(x)))
            logging.info("Query point"+str(x)+" .Log Likelihood: "+str(-np.log(res)))
            return np.array(res).reshape(n, 1)

        def rp_emukit():
            # Wrap around Emukit interface
            from emukit.core.loop.user_function import UserFunctionWrapper
            return UserFunctionWrapper(_rp_emukit), _rp_emukit
        start = time.time()

        budget = self.options['naive_bq_budget']
        test_x = self.gpr.X_test
        test_y = self.gpr.Y_test

        q = np.zeros((test_x.shape[0], budget))
        var = np.zeros((test_x.shape[0], budget))

        # Initial points - note that as per GPML convention, the hyperparameters are expressed in log scale
        # Initialise to the MAP estimate
        map_model, _, _ = self.maximum_a_posterior(num_restarts=1, max_iters=500, verbose=False)
        if isinstance(self.gpr, RBFGPRegression):
            self.gpr.set_params(variance=map_model.rbf.variance, gaussian_noise=map_model.Gaussian_noise.variance)
        elif isinstance(self.gpr, PeriodicGPRegression):
            self.gpr.set_params(variance=map_model.std_periodic.variance,
                                gaussian_noise=map_model.Gaussian_noise.variance)

        log_phi_initial = self.prior.mean.reshape(1, -1)
        #r_initial = np.exp(self.gpr.log_sample(phi=np.exp(log_phi_initial))[0] + np.log(self.prior(log_phi_initial)))
        r = np.empty((budget,))
        samples = np.empty((budget, self.dimensions))
        pred = np.zeros((test_x.shape[0], ))
        pred_var = np.zeros((test_x.shape[0], ))

        log_r_initial = np.sqrt(2 * np.exp(self.gpr.log_sample(
            phi=np.exp(log_phi_initial.reshape(-1))
        )[0]))
        # log_r_initial = self.gpr.log_sample(
        #    phi=np.exp(log_phi_initial.reshape(-1)))[0]

        # Setting up kernel - Note we only marginalise over the lengthscale terms, other hyperparameters are set to the
        # MAP values.
        kern = GPy.kern.RBF(self.dimensions,
                            variance=1.,
                            lengthscale=1)

        log_r_gp = GPy.models.GPRegression(log_phi_initial, log_r_initial.reshape(1, -1), kern)
        log_r_model = WarpedIntegrandModel(WsabiLGP(log_r_gp), self.prior)

        # Firstly, within the given allowance, compute an estimate of the model evidence. Model evidence is the common
        # denominator for all predictive distributions.
        for i_a in range(budget):
            log_phi_i = np.array(select_batch(log_r_model, 1, "Kriging Believer")).reshape(1, -1)
            try:
                log_r_i = self.gpr.log_sample(phi=np.exp(log_phi_i))[0]
                r[i_a] = log_r_i
                samples[i_a, :] = log_phi_i
                log_r_model.update(log_phi_i, np.exp(log_r_i).reshape(1, -1))
                print(np.exp(log_phi_i), log_r_i)
            except np.linalg.linalg.LinAlgError:
                continue

        r = np.exp(r + np.log(self.prior(samples)))
        r_gp = GPy.models.GPRegression(samples, r.reshape(-1, 1), kern)
        r_model = self._wrap_emukit(r_gp)
        #r_loop = VanillaBayesianQuadratureLoop(model=r_model)

        # Firstly, within the given allowance, compute an estimate of the model evidence. Model evidence is the common
        # denominator for all predictive distributions.
        #r_loop.run_loop(user_function=rp_emukit()[0], stopping_condition=budget)
        #log_phi = r_loop.loop_state.X
        #r = r_loop.loop_state.Y.reshape(-1)
        #np.savetxt('bq_sotonmet_samples.csv', r)

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
            for i_b in range(0, budget):
                log_phi_i = samples[i_b, :]
                _, q_i, var_i = self.gpr.log_sample(phi=np.exp(log_phi_i), x=test_x[i_x, :])
                q[i_x, i_b] = q_i
                var[i_x, i_b] = var_i
            # Construct rq vector
            q_x = q[i_x, :]
            var_x = var[i_x, :]

            rq = r * q_x
            rq_gp = GPy.models.GPRegression(samples, rq.reshape(-1, 1), kern)
            rq_model = self._wrap_emukit(rq_gp)
            rq_int = rq_model.integrate()[0]

            rvar = r * var_x
            rvar_gp = GPy.models.GPRegression(samples, rvar.reshape(-1, 1), kern)
            rvar_model = self._wrap_emukit(rvar_gp)
            rvar_int = rvar_model.integrate()[0]

            # Now estimate the posterior

            pred[i_x] = rq_int / r_int
            pred_var[i_x] = rvar_int / r_int

            logging.info('Progress: '+str(i_x+1)+'/'+str(test_x.shape[0]))

        rmse = self.compute_rmse(pred, test_y)
        ll, cs = self.compute_ll_cs(pred, pred_var, test_y)
        logging.info(pred, test_y)
        print('Root Mean Squared Error:', rmse)
        print('Log-likelihood:', ll)
        print('Calibration Score:', cs)
        end = time.time()
        print("Active Sampling Time: ", quad_time-start)
        print("Total Time elapsed: ", end-start)
        if verbose:
            self.visualise(pred, pred_var, test_y)
        return rmse, ll, quad_time-start, np.log(r_int)

    def wsabi(self, verbose=True):
        # Allocating number of maximum evaluations
        start = time.time()
        budget = self.options['wsabi_budget']
        batch_count = 1
        test_x = self.gpr.X_test
        test_y = self.gpr.Y_test

        # Allocate memory of the samples and results
        log_phi = np.zeros((budget*batch_count, self.dimensions,))  # The log-hyperparameter sampling points
        log_r = np.zeros((budget*batch_count, ))  # The log-likelihood function
        q = np.zeros((test_x.shape[0], budget*batch_count))  # Prediction
        var = np.zeros((test_x.shape[0], budget*batch_count))  # Posterior variance

        # Initial points - note that as per GPML convention, the hyperparameters are expressed in log scale
        # Initialise to the MAP estimate
        map_model, _, _ = self.maximum_a_posterior(num_restarts=1, max_iters=1000, verbose=False)
        display(map_model)
        # mean_vals = map_model.param_array
        # self.gpr.reset_params()
        if isinstance(self.gpr, PeriodicGPRegression):
            pass
            #self.gpr.set_params(variance=map_model.std_periodic.variance,
            #                    gaussian_noise=map_model.Gaussian_noise.variance)
        elif isinstance(self.gpr, RBFGPRegression):
            self.gpr.set_params(variance=map_model.rbf.variance, gaussian_noise=map_model.Gaussian_noise.variance)

        # Set prior mean to the MAP value
        # self.prior = Gaussian(mean=mean_vals.reshape(-1), covariance=self.options['prior_variance'])

        log_phi_initial = self.options['prior_mean'].reshape(1, -1)
        log_r_initial = np.sqrt(2 * np.exp(self.gpr.log_sample(
            phi=np.exp(log_phi_initial.reshape(-1))
        )[0]))
        #log_r_initial = self.gpr.log_sample(
        #    phi=np.exp(log_phi_initial.reshape(-1)))[0]

        pred = np.zeros((test_x.shape[0], ))
        pred_var = np.zeros((test_x.shape[0], ))

        # Setting up kernel - Note we only marginalise over the lengthscale terms, other hyperparameters are set to the
        # MAP values.
        kern = GPy.kern.RBF(self.dimensions,
                            variance=1.,
                            lengthscale=1)

        log_r_gp = GPy.models.GPRegression(log_phi_initial, log_r_initial.reshape(1, -1), kern)
        log_r_model = WarpedIntegrandModel(WsabiLGP(log_r_gp), self.prior)
        print(self.prior.mean)

        # Firstly, within the given allowance, compute an estimate of the model evidence. Model evidence is the common
        # denominator for all predictive distributions.
        for i_a in range(budget-1):
            log_phi_i = np.array(select_batch(log_r_model, batch_count, "Kriging Believer")).reshape(batch_count, -1)
            try:
                log_r_i = self.gpr.log_sample(phi=np.exp(log_phi_i))[0]
                log_r[i_a:i_a+batch_count] = log_r_i
                log_phi[i_a:i_a+batch_count, :] = log_phi_i
                log_r_model.update(log_phi_i, np.exp(log_r_i).reshape(1, -1))
                print(np.exp(log_phi_i), log_r_i)
            except np.linalg.linalg.LinAlgError:
                continue
        quad_time = time.time()

        max_log_r = max(log_r)
        # Save the evaluations
        log_r_pd = pd.Series(log_r, name='lml')
        log_r_pd.to_csv('lml_soton.csv')

        r = np.exp(log_r - max_log_r)
        r_gp = GPy.models.GPRegression(log_phi[:1, :], np.sqrt(2 * r[0].reshape(1, 1)), kern)
        r_model = WarpedIntegrandModel(WsabiLGP(r_gp), self.prior)
        r_model.update(log_phi[1:, :], r[1:].reshape(-1, 1))
        r_gp.optimize()
        r_int = np.exp(np.log(r_model.integral_mean()[0]) + max_log_r) # Model evidence
        log_r_int = np.log(r_int)  # Model log-evidence

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
            # rq_int = rq_model.integral_mean()[0] + q_min * r_int
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
            print(pred_var[i_x])

            logging.info('Progress: '+str(i_x+1)+'/'+str(test_x.shape[0]))

        rmse = self.compute_rmse(pred, test_y)
        ll, cs = self.compute_ll_cs(pred, pred_var, test_y)
        print('Root Mean Squared Error:', rmse)
        print('Log-likelihood', ll)
        print('Calibration score', cs)
        end = time.time()
        print("Active Sampling Time: ", quad_time-start)
        print("Total Time: ", end-start)

        print("Estimate of model evidence: ", r_int,)
        print("Model log-evidence ", log_r_int)
        if verbose:
            self.visualise(pred, pred_var, test_y)
        return rmse, ll, quad_time-start, log_r_int

    def wsabi_bqr(self, verbose=True, compute_var=True):
        from bayesquad.quadrature import compute_mean_gp_prod_gpy_2

        # Allocating number of maximum evaluations
        start = time.time()
        budget = self.options['wsabi_budget']
        test_x = self.gpr.X_test
        test_y = self.gpr.Y_test

        # Allocate memory of the samples and results
        log_phi = np.zeros((budget, self.dimensions,))  # The log-hyperparameter sampling points
        log_r = np.zeros((budget, ))  # The log-likelihood function
        q = np.zeros((test_x.shape[0], budget))  # Prediction
        var = np.zeros((test_x.shape[0], budget))  # Posterior variance

        # Initial points - note that as per GPML convention, the hyperparameters are expressed in log scale
        # Initialise to the MAP estimate
        map_model, _, _ = self.maximum_a_posterior(num_restarts=1, max_iters=1000, verbose=False)
        if isinstance(self.gpr, PeriodicGPRegression):
            self.gpr.set_params(variance=map_model.std_periodic.variance,
                                gaussian_noise=map_model.Gaussian_noise.variance)
        elif isinstance(self.gpr, RBFGPRegression):
            self.gpr.set_params(variance=map_model.rbf.variance, gaussian_noise=map_model.Gaussian_noise.variance)

        # Set prior mean to the MAP value
        # self.prior = Gaussian(mean=map_vals.reshape(-1), covariance=self.options['prior_variance'])

        log_phi_initial = self.options['prior_mean'].reshape(1, -1)
        log_r_initial = np.sqrt(2 * np.exp(self.gpr.log_sample(
            phi=np.exp(log_phi_initial.reshape(-1))
        )[0]))

        pred = np.zeros((test_x.shape[0], ))
        pred_var = np.zeros((test_x.shape[0], ))

        # Setting up kernel - Note we only marginalise over the lengthscale terms, other hyperparameters are set to the
        # MAP values.
        kern = GPy.kern.RBF(self.dimensions,
                            variance=1.,
                            lengthscale=1.)

        log_r_gp = GPy.models.GPRegression(log_phi_initial, log_r_initial.reshape(1, -1), kern)
        log_r_model = WarpedIntegrandModel(WsabiLGP(log_r_gp), self.prior)

        # Firstly, within the given allowance, compute an estimate of the model evidence. Model evidence is the common
        # denominator for all predictive distributions.
        for i_a in range(budget):
            log_phi_i = np.array(select_batch(log_r_model, 1, "Kriging Believer")).reshape(1, -1)
            try:
                log_r_i = self.gpr.log_sample(phi=np.exp(log_phi_i))[0]
                log_r[i_a:i_a+1] = log_r_i
                log_phi[i_a:i_a+1, :] = log_phi_i
                log_r_model.update(log_phi_i, np.exp(log_r_i).reshape(1, -1))
                print(np.exp(log_phi_i), log_r_i)
            except np.linalg.linalg.LinAlgError:
                print('Error!')
                continue


        max_log_r = np.max(log_r)
        r = np.exp(log_r - max_log_r)
        # r = np.exp(log_r)
        r_gp = GPy.models.GPRegression(log_phi[:1, :], np.sqrt(2 * r[0].reshape(1, 1)), kern)
        r_model = WarpedIntegrandModel(WsabiLGP(r_gp), self.prior)
        r_model.update(log_phi[1:, :], r[1:].reshape(-1, 1))
        r_gp.optimize()
        r_int_prime = r_model.integral_mean()[0] # Model evidence
        r_int = np.exp(np.log(r_int_prime) + max_log_r)
        # r_int = r_int_prime
        r = np.exp(np.log(log_r) + max_log_r)
        print("Estimate of model evidence: ", r_int,)

        # Secondly, compute and marginalise the predictive distribution for each individual points
        for i_x in range(test_x.shape[0]):

            # Note that we do not active sample again for q, we just use the same samples sampled when we compute
            # the log-evidence
            _, q_initial, var_initial = self.gpr.log_sample(phi=np.exp(log_phi_initial), x=test_x[i_x, :])

            # Initialise GPy GP surrogate for and q(\phi) - note that this is a BQZ approach and we do not model rq
            # as one GP but separate GPs to account for correlation
            # Sample for q values
            for i_b in range(budget):
                log_phi_i = log_phi[i_b, :]
                _, q[i_x, i_b], var[i_x, i_b] = self.gpr.log_sample(phi=np.exp(log_phi_i), x=test_x[i_x, :])

            # Enforce positivity in q
            q_x = q[i_x, :].copy()
            var_x = var[i_x, :].copy()
            #sns.distplot(q_x, bins=20)
            #plt.axvline(test_y[i_x], color='red', label='Ground Truth')
            #plt.xlabel("$\phi$")
            #plt.ylabel("$q(\phi)$")
           # plt.legend()
            #plt.show()

            q_min = np.min(q_x)
            if q_min < 0:
                q_x = q_x - q_min
            else:
                q_min = 0

            # Do the same exponentiation and rescaling trick for q
            q_gp = GPy.models.GPRegression(log_phi[:1, :], np.sqrt(2 * q_x[0].reshape(1, 1)),
                                           GPy.kern.RBF(self.dimensions,
                            variance=2.,
                            lengthscale=2.))
            q_model = WarpedIntegrandModel(WsabiLGP(q_gp), self.prior)
            q_model.update(log_phi[1:, :], q_x[1:].reshape(-1, 1))
            q_gp.optimize()
            #display(q_gp)
            #q_int = q_model.integral_mean()[0]


            # Evaluate numerator
            alpha_q = q_model.gp._alpha
            alpha_r = r_model.gp._alpha
            rq_gp = GPy.models.GPRegression(log_phi, q_model.gp._gp.Y * r_model.gp._gp.Y,
                                            GPy.kern.RBF(self.dimensions,
                                                           variance=1.,
                                                           lengthscale=1.))

            rq_gp.optimize()

            #q_sq_gp = GPy.models.GPRegression(log_phi, (q_x - alpha_q).reshape(-1, 1),
            #                                  GPy.kern.RBF(self.dimensions,
            #                                               variance=2.,
            #                                              lengthscale=1.))

            #q_sq_gp.optimize()

            #r_sq_gp = GPy.models.GPRegression(log_phi, (r-alpha_r).reshape(-1, 1),
            #                                 GPy.kern.RBF(self.dimensions,
            #                                              variance=1.,
             #                                              lengthscale=1.)
            #                                  )
            #r_sq_gp.optimize()

            #display(rq_gp)
            #display(q_sq_gp)
            #print('---------------')
            #display(r_sq_gp)
            #display(q_sq_gp)
            #print( compute_mean_gp_prod_gpy_2(self.prior, q_gp, q_gp))
            #print( q_int )
            #exit()
            n1 = alpha_r * alpha_q
            n2 = 0.5 * alpha_q * compute_mean_gp_prod_gpy_2(self.prior, r_gp, r_gp)
            n3 = 0.5 * alpha_r * compute_mean_gp_prod_gpy_2(self.prior, q_gp, q_gp)
            n4 = 0.25 * compute_mean_gp_prod_gpy_2(self.prior, rq_gp, rq_gp)
            n = (n1 + n2 + n3 + n4)
            res = n / r_int_prime + q_min
            print('res', n1, n2, n3, n4, res)

            pred[i_x] = res
            #print('final', pred[i_x])
            if compute_var:
                var_gp = GPy.models.GPRegression(log_phi[:1, :], np.sqrt(2 * var_x[0].reshape(1, 1)), kern)
                var_model = WarpedIntegrandModel(WsabiLGP(var_gp), self.prior)
                var_model.update(log_phi[1:, :], var_x[1:].reshape(-1, 1))
                var_gp.optimize()

                rvar_gp = GPy.models.GPRegression(log_phi, var_model.gp._gp.Y * r_model.gp._gp.Y, kern)
                rvar_gp.optimize()

                alpha_var = var_model.gp._alpha
                var_num = alpha_r * alpha_var + \
                          0.5 * alpha_r * compute_mean_gp_prod_gpy_2(self.prior, var_model.gp._gp, var_model.gp._gp) + \
                          0.5 * alpha_var * compute_mean_gp_prod_gpy_2(self.prior, r_model.gp._gp, r_model.gp._gp) + \
                          0.25 * (compute_mean_gp_prod_gpy_2(self.prior, rvar_gp, rvar_gp))
                pred_var[i_x] = var_num / r_int
            logging.info('Progress: '+str(i_x+1)+'/'+str(test_x.shape[0]))

        rmse = self.compute_rmse(pred, test_y)
        print('Root Mean Squared Error:', rmse)
        ll, cs = None, None
        if compute_var:
            ll, cs = self.compute_ll_cs(pred, pred_var, test_y)
            print('Log-likelihood', ll)
            print('Calibration score', cs)
        end = time.time()
        print("Total Time: ", end-start)

        print("Estimate of model evidence: ", r_int,)
        print("Model log-evidence ", np.log(r_int))
        if verbose:
            self.visualise(pred, pred_var, test_y)
        return rmse, ll, None, np.log(r_int)
    # ----- Evaluation Metric ---- #

    def reset_prior(self):
        self.prior = Gaussian(mean=self.options['prior_mean'].reshape(-1),
                              covariance=self.options['prior_variance'])
        logging.info("Prior reset at mean, "+str(self.options['prior_mean'])+
                     ' and variance '+str(self.options['prior_variance']))

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
        qrbf = QuadratureRBF(rbf, integral_bounds=[(-4, 8)]*self.dimensions)
        model = BaseGaussianProcessGPy(kern=qrbf, gpy_model=gpy_gp)
        method = VanillaBayesianQuadrature(base_gp=model)
        return method

    def visualise(self, y_pred: np.ndarray, y_var: np.ndarray, y_grd: np.ndarray):
        dim = self.gpr.X_test.shape[1]
        if dim > 2:
            # For higher dimension, we only show the comparison between the test data and the ground truth
            xv = list(range(len(y_pred)))
            plt.figure(1)
            plt.errorbar(xv, y_pred, yerr=np.sqrt(y_var), fmt='.', ecolor='r', capsize=2, label='Predictions +/- 1SD')
            plt.figure(2)
            plt.plot(y_grd, ".", color='red', label='Ground Truth')
            plt.legend()
            plt.xlabel("Sample Number")
            plt.ylabel("Predicted/Actual Value")
        else:
            # For lower dimensions, we can show the things such as the contour plot of the likelihood surface, the
            # posterior and include a plot between independent and dependent variables.

            x_test = self.gpr.X_test.reshape(-1)
            y_test = self.gpr.Y_test.reshape(-1)

            x_train = self.gpr.X_train.reshape(-1)
            y_train = self.gpr.Y_train.reshape(-1)

            plt.figure(1, figsize=(10, 5))
            test = np.vstack((x_test, y_test, y_pred.reshape(-1), y_var.reshape(-1))).T
            train = np.vstack((x_train, y_train)).T
            test = test[test[:, 0].argsort()]
            train = train[train[:, 0].argsort()]
            x_train, y_train = train[:, 0], train[:, 1]
            x_test, y_test, y_pred, y_var = test[:, 0], test[:, 1], test[:, 2], test[:, 3]

            plt.plot(x_train, y_train, color='b', alpha=0.5, label='Training Data')
            plt.plot(x_test, y_test, '.', color='b', markersize=6, label='Ground Truth')
            #plt.errorbar(x_test.reshape(-1), y_pred.reshape(-1), yerr=np.sqrt(y_var),
            #             color='r',
            #             fmt='*', ecolor='lightgrey', capsize=2, markersize=6, label='Prediction +/- 1SD')
            plt.plot(x_test.reshape(-1), y_pred.reshape(-1,), color='r', markersize=6, marker='*', linestyle='None', label='Prediction')
            plt.xlabel('Time Index')
            plt.ylabel('Tide Height (m)')
            plt.legend()
        plt.show()

    def _unpack_options(self,
                        prior_mean: Union[float, np.ndarray] = np.array([0, 5, 0, -4]),
                        prior_variance: Union[float, np.ndarray] = 16.*np.eye(4),
                        smc_budget: int = 1000,
                        naive_bq_budget: int = 75,
                        naive_bq_kern_lengthscale: float = 1.,
                        naive_bq_kern_variance: float = 1.,
                        wsabi_budget: int = 100,
                        posterior_range=(-5., 5.),
                        # The axis ranges between which the posterior samples are drawn
                        posterior_eps: float = 0.02,
                        ):
        if self.gpr.param_dim == 1:
            prior_mean = np.array([prior_mean]).reshape(1, 1)
            prior_variance = np.array([[prior_variance]]).reshape(1, 1)
        elif self.gpr.param_dim > 1 and isinstance(prior_variance, float) and isinstance(prior_mean, float):
            prior_mean = np.array([prior_mean] * (self.gpr.param_dim)).reshape(-1, 1)
            prior_variance *= np.eye(self.gpr.param_dim)
        else:
            assert len(prior_mean) == self.gpr.param_dim
            assert prior_variance.shape[0] == prior_variance.shape[1]
            assert prior_variance.shape[0] == self.gpr.param_dim
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
def sample_from_param_posterior(gpy_gp: GPy.core.gp, num_samples=1000, mc_iters=1,
                                mc_method='smc',
                                save_csv=True, plot_samples=True):
    if mc_method == 'smc':
        mc = GPy.inference.mcmc.Metropolis_Hastings(gpy_gp)
        t = mc.sample(Ntotal=num_samples)
    elif mc_method == 'hmc':
        mc = GPy.inference.mcmc.hmc.HMC_shortcut(gpy_gp)
        t = mc.sample(m_iters=num_samples, hmc_iters=mc_iters)
    else:
        raise NotImplemented
    if plot_samples:
        print(t)
        _ = plt.plot(t)
        plt.show()
    if save_csv:
        df = pd.DataFrame(t, columns=gpy_gp.parameter_names_flat())
        df.to_csv('PosteriorSampling.csv')
    return t


# ---- Performance evaluation --------- #
def eval_perf(rq: RegressionQuadrature, method: str):
    sample_n = [120, 150, 200, 250]
    res = np.zeros((len(sample_n), 4))
    for i in range(len(sample_n)):
            print('**NUMBER OF SAMPLES**:', sample_n[i])
            if method == 'wsabi':
                rq.options['wsabi_budget'] = sample_n[i]
                res[i, 0], res[i, 1], res[i, 2], res[i, 3] = rq.wsabi(verbose=False)
            elif method == 'bqz':
                rq.options['wsabi_budget'] = sample_n[i]
                res[i, 0], res[i, 1], res[i, 2], res[i, 3] = rq.wsabi_bqr(verbose=False)
            elif method == 'bq':
                rq.options['naive_bq_budget'] = sample_n[i]
                res[i, 0], res[i, 1], res[i, 2], res[i, 3] = rq.bq(verbose=False)
            elif method == 'mc':
                rq.options['smc_budget'] = sample_n[i]
                res[i, 0], res[i, 1], res[i, 2], res[i, 3] = rq.mc(verbose=False)
            elif method == 'map':
                map_res = rq.maximum_a_posterior(verbose=False, max_iters=sample_n[i], num_restarts=2)
                res[i, 0], res[i, 1] = map_res[1], map_res[2]
                res[i, 2], res[i, 3] = np.nan, np.nan
            else:
                raise NotImplemented()


    df = pd.DataFrame(res, columns=['rmse', 'll', 'quad_time', 'evidence'])
    df.index = sample_n
    df.to_csv(method+'_perf.csv')
    logging.info(method+' evaluation completed')

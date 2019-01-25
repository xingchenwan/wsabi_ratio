# Yacht Regression Experiment
# Xingchen Wan | xingchen.wan@st-annes.ox.ac.uk | Jan 2019

from ratio.functions import GPRegressionFromFile
from bayesquad.priors import Gaussian
import numpy as np
from typing import Union, Tuple
from scipy.stats import moment
import pymc3 as pm

import matplotlib
matplotlib.use("TkAgg")
# This is to prevent a macOS bug with matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import theano.tensor as tt
import pandas as pd

import GPy
from bayesquad.quadrature import WarpedIntegrandModel, WsabiLGP, WarpedGP, GP
from bayesquad.batch_selection import select_batch
from IPython.display import display

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RegressionQuadrature:
    def __init__(self, regression_model: GPRegressionFromFile, **kwargs):
        self.gpr = regression_model
        self.options = self._unpack_options(**kwargs)
        self.dimensions = self.gpr.dimensions

        # Parameter prior - note that the prior is in *log-space*
        self.prior = Gaussian(mean=self.options['prior_mean'].reshape(-1),
                              covariance=self.options['prior_variance'])
        self.gp = self.gpr.model

    def maximum_a_posterior(self, num_restarts=10, max_iters=1000):
        """
        Using a point estimate of the log-likelihood function
        :return: an array of predictions, y_pred
        """
        # Create a local version of the model
        local_gp = self.gp.copy()

        # Fetch and train data from the GPy GP object
        X = self.gpr.X_train
        Y = self.gpr.Y_train
        X_test = self.gpr.X_test
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
        rmse = self.compute_rmse(pred, self.gpr.Y_test)
        print('Root Mean Squared Error:', rmse)
        print('MAP lengthscale vector:', local_gp.rbf.lengthscale)
        display(local_gp)
        self.visualise(pred, self.gpr.Y_test)
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
        test_x = self.gpr.X_test[:, :]
        test_y = self.gpr.Y_test[:]
        test_length = test_x.shape[0]

        # Fetch MAP values of variance and Gaussian noise
        map_model, _ = self.maximum_a_posterior(num_restarts=1)
        variance_map = map_model.rbf.variance
        gaussian_noise_map = map_model.Gaussian_noise.variance
        self.gpr.set_params(variance=variance_map,gaussian_noise=gaussian_noise_map)

        # Sample the parameter posterior, note that we discard the first column (variance) and last column
        # (gaussian noise) since we are interested in marginalising the lengthscale hyperparameters only
        samples = sample_from_param_posterior(self.gpr.model, budget, 10, False, False)
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
        rmse = self.compute_rmse(pred, self.gpr.Y_test)
        if verbose:
            self.visualise(pred, test_y)
        print("Root Mean Squared Error", rmse)
        return rmse

    def wsabi(self, verbose=True):
        # Allocating number of maximum evaluations
        budget = self.options['wsabi_budget']
        batch_count = 1
        test_x = self.gpr.X_test[:, :]
        test_y = self.gpr.Y_test[:]

        # Allocate memory of the samples and results
        log_phi = np.zeros((budget*batch_count, self.gpr.dimensions,)) # The log-hyperparameter sampling points
        log_r = np.zeros((budget*batch_count, )) # The log-likelihood function
        q = np.zeros((test_x.shape[0], budget*batch_count)) # Prediction

        # Initial points - note that as per GPML convention, the hyperparameters are expressed in log scale
        # Initialise to the MAP estimate
        map_model, _ = self.maximum_a_posterior(num_restarts=1, max_iters=500)
        self.gpr.set_params(variance=map_model.rbf.variance, gaussian_noise=map_model.Gaussian_noise.variance)
        log_phi_initial = np.log(map_model.rbf.lengthscale).reshape(1, -1)
        log_r_initial = self.gpr.log_sample(phi=np.exp(log_phi_initial))[0]
        pred = np.zeros((test_x.shape[0], ))

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
            log_phi_i = np.array(select_batch(log_r_model, batch_count, )).reshape(batch_count, -1)
            log_r_i = self.gpr.log_sample(phi=np.exp(log_phi_i))[0]

            log_r[i_a:i_a+batch_count] = log_r_i
            log_phi[i_a:i_a+batch_count, :] = log_phi_i

            log_r_model.update(log_phi_i, log_r_i.reshape(1, -1))
        max_log_r = max(log_r)
        r = np.exp(log_r - max_log_r)
        r_gp = GPy.models.GPRegression(log_phi, r.reshape(-1, 1), kern)
        r_model = WarpedIntegrandModel(WsabiLGP(r_gp), self.prior)
        r_gp.optimize()
        r_int = np.exp(np.log(r_model.integral_mean()[0]) + max_log_r) # Model evidence
        log_r_int = np.log(r_int) # Model log-evidence
        if verbose:
            print("Estimate of model evidence: ", r_int,)
            print("Model log-evidence ", log_r_int)

        # Visualise the model parameter posterior
        # neg_log_post = np.array((budget, )) # Negative log-posterior
        # rp = np.array((budget, ))
        # for i in range(budget):
        #    neg_log_post[i] = (log_r[i] + self.prior.log_eval(log_phi[i, :]) - log_r_int)
        # Then train a GP for the log-posterior surface
        #log_posterior_gp = GPy.models.GPRegression(np.exp(log_phi), np.exp(neg_log_post).reshape(-1, 1), kern)
        #sample_from_param_posterior(log_posterior_gp)

        # Secondly, compute and marginalise the predictive distribution for each individual points
        for i_x in range(test_x.shape[0]):

            # Note that we do not active sample again for q, we just use the same samples sampled when we compute
            # the log-evidence
            _, q_initial = self.gpr.log_sample(phi=np.exp(log_phi_initial), x=test_x[i_x, :])

            # Initialise GPy GP surrogate for and q(\phi)r(\phi)
            # Sample for q values
            for i_b in range(budget*batch_count):
                log_phi_i = log_phi[i_b, :]
                log_r_i, q_i = self.gpr.log_sample(phi=np.exp(log_phi_i), x=test_x[i_x, :])
                q[i_x, i_b] = q_i

            # Enforce positivity in q
            q_x = q[i_x, :]
            q_min = np.min(q_x)
            if q_min < 0:
                q_x = q_x - q_min
            else:
                q_min = 0
            log_phi_x = log_phi
            log_r_x = log_r

            # Do the same exponentiation and rescaling trick for q
            log_rq_x = log_r_x + np.log(q_x)
            max_log_rq = np.max(log_rq_x)
            rq = np.exp(log_rq_x - max_log_rq)

            rq_gp = GPy.models.GPRegression(log_phi_x, rq.reshape(-1, 1), kern)
            rq_gp.optimize()
            rq_model = WarpedIntegrandModel(WsabiLGP(rq_gp), self.prior)

            # Now estimate the posterior
            rq_int = np.exp(np.log(rq_model.integral_mean()[0]) + max_log_rq) + q_min * r_int
            pred[i_x] = rq_int / r_int
            logging.info('Progress: '+str(i_x+1)+'/'+str(test_x.shape[0]))

        rmse = self.compute_rmse(pred, test_y)
        if verbose:
            logging.info(pred, test_y)
            self.visualise(pred, test_y)
        print('Root Mean Squared Error:', rmse)
        return rmse

    def wsabi_ratio(self):

        # Implementation of the Bayesian Quadrature for Ratio paper

        # 1.1 Allocating number of maximum evaluations
        budget = self.options['wsabi_budget']
        batch_count = 1
        test_x = self.gpr.X_test[:, :]
        test_y = self.gpr.Y_test[:]

        # 1.2 Allocate memory of the samples and results
        log_phi = np.zeros((budget * batch_count, self.gpr.dimensions,))  # The log-hyperparameter sampling points
        log_r = np.zeros((budget * batch_count,))  # The log-likelihood function
        q = np.zeros((test_x.shape[0], budget * batch_count))  # Prediction

        # 2. Initial points - note that as per GPML convention, the hyperparameters are expressed in log scale
        # 2.1 Initialise to the MAP estimate
        map_model, _ = self.maximum_a_posterior(num_restarts=1, max_iters=500)
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
            log_phi[i_a:i_a + batch_count, :] = log_phi_i

            log_r_model.update(log_phi_i, log_r_i.reshape(1, -1))

        # 4. Obtain phi_s locations
        log_phi_s = self.get_phi_s(log_phi)

        # 5. Rescale the original model
        max_log_r = max(log_r)
        r = np.exp(log_r - max_log_r)
        r_gp = GPy.models.GPRegression(log_phi, r.reshape(-1, 1), kern)
        r_model = WarpedIntegrandModel(WsabiLGP(r_gp), self.prior)
        r_gp.optimize()
        m_int_r = np.exp(np.log(r_model.integral_mean()[0]) + max_log_r)  # Model evidence
        log_m_int_r = np.log(m_int_r)  # Model log-evidence
        print("Estimate of model evidence: ", m_int_r, )
        print("Model log-evidence ", log_m_int_r)

        # 6. Compute the eta_{rr} vector on log_phi_s
        m_rs = r_gp.predict_noiseless(log_phi_s) # Posterior mean of the GP over original r at phi_s points
        log_m_rs = np.log(m_rs) + max_log_r  # Log of the posterior mean of GP over original r
        m_log_rs = log_r_gp.predict_noiseless(log_phi_s) # Posterior mean of the GP over log(r)
        eta_rr = m_rs * (m_log_rs - log_m_rs)

        # 6.1 Similar to r, compute the integral mean of eta_{rr}
        eta_rr_gp = GPy.models.GPRegression(log_phi_s, eta_rr, kern)
        eta_rr_gp.optimize()
        eta_rr_model = WarpedIntegrandModel(WsabiLGP(eta_rr_gp), self.prior)
        m_int_eta_rr = eta_rr_model.integral_mean()[0]

        # 7, compute and marginalise the predictive distribution for each individual points
        for i_x in range(test_x.shape[0]):

            # Note that we do not active sample again for q, we just use the same samples sampled when we compute
            # the log-evidence
            _, q_initial = self.gpr.log_sample(phi=np.exp(log_phi_initial), x=test_x[i_x, :])

            # 7.1 Initialise GPy GP surrogate for and q(\phi)r(\phi)
            # Sample for q values
            for i_b in range(budget * batch_count):
                log_phi_i = log_phi[i_b, :]
                log_r_i, q_i = self.gpr.log_sample(phi=np.exp(log_phi_i), x=test_x[i_x, :])
                q[i_x, i_b] = q_i

            # 7.2 Discard hyperparameter samples that lead to negative prediction - there should not be neg predictions!
            q_x = q[i_x, :]
            nonneg_idx = q_x >= 0
            q_x = q_x[nonneg_idx]
            log_r_x = log_r[nonneg_idx]
            log_phi_x = log_phi[nonneg_idx, :]

            # 7.3 Compute the eta_{rq} vector on log_phi_s
            q_gp = GPy.models.GPRegression(log_phi_x, q_x, kern)
            q_gp.optimize()
            log_q_gp = GPy.models.GPRegression(log_phi_x, np.log(q_x), kern)
            log_q_gp.optimize()
            m_qs = q_gp.predict_noiseless(log_phi_s)
            log_m_qs = np.log(m_qs)
            m_log_qs = log_q_gp.predict_noiseless(log_phi_s)
            eta_rq = m_rs * (m_log_qs - log_m_qs)

            # 8. Formulate relevant integrals and evaluate the integrals

            # 8.1 Evaluate rho_{0}
            log_rq_x = log_r_x + np.log(q_x)
            max_log_rq = np.max(log_rq_x)
            rq = np.exp(log_rq_x - max_log_rq)
            rq_gp = GPy.models.GPRegression(log_phi_x, rq.reshape(-1, 1), kern)
            rq_gp.optimize()
            rq_model = WarpedIntegrandModel(WsabiLGP(rq_gp), self.prior)
            m_int_rq = np.exp(np.log(rq_model.integral_mean()[0]) + max_log_rq)
            rho_0 = m_int_rq / m_int_r

            # 8.2 Evaluate C_{q} - Adjustment factor 1
            q_eta_rq = m_qs * eta_rq
            q_eta_rq_gp = GPy.models.GPRegression(log_phi_x, q_eta_rq.reshape(-1, 1), kern)
            q_eta_rq_gp.optimize()
            q_eta_rq_model = WarpedIntegrandModel(WsabiLGP(q_eta_rq_gp), self.prior)
            m_int_q_eta_rq = q_eta_rq_model.integral_mean()[0]
            c_q = m_int_q_eta_rq / m_int_r

            # 8.3 Evaluate C_{r} - Adjustment factor 2
            q_eta_rr = m_qs * eta_rr
            q_eta_rr_gp = GPy.models.GPRegression(log_phi_x, q_eta_rr.reshape(-1, 1), kern)
            q_eta_rr_gp.optimize()
            q_eta_rr_model = WarpedIntegrandModel(WsabiLGP(q_eta_rq_gp), self.prior)
            m_int_q_eta_rr = q_eta_rr_model.integral_mean()[0]
            c_r = 1. / m_int_r * (m_int_q_eta_rr - m_int_eta_rr * (m_int_rq / m_int_r))

            pred[i_x] = rho_0 + c_q + c_r
            print('Progress: ', i_x + 1, '/', test_x.shape[0])

        rmse = self.compute_rmse(pred, test_y)
        print(pred, test_y)
        self.visualise(pred, test_y)
        print('Root Mean Squared Error:', rmse)
        return rmse

    # ----- Evaluation Metric ---- #

    def compute_rmse(self, y_pred: np.ndarray, y_grd: np.ndarray) -> float:
        """
        Compute the root mean squared error between prediction (y_pred) with the ground truth y in the test set
        :param y_pred: a vector with the same length as the test set y
        :return: rmse
        """
        y_pred = y_pred.reshape(-1)
        y_grd = y_grd.reshape(-1)
        length = y_pred.shape[0]
        assert length == y_grd.shape[0], "the length of the prediction vector does " \
                                               "not match the ground truth vector!"
        rmse = np.sqrt(np.sum((y_pred - y_grd) ** 2) / length)
        return rmse

    def get_phi_s(self, phi_c):
        """
        Construct a Voronoi diagram and identify the locations of phi_s
        (See Bayesian Quadrature for Ratio paper for reasoning)
        """
        from scipy.spatial import Voronoi
        vor = Voronoi(phi_c)
        return np.concatenate((vor.vertices, phi_c), axis=0)

    @staticmethod
    def visualise(y_pred: np.ndarray, y_grd: np.ndarray):
        plt.plot(y_pred, ".", color='red')
        plt.plot(y_grd, ".", color='blue')
        plt.show()

    def _unpack_options(self,
                        prior_mean: Union[float, np.ndarray] = 0.,
                        prior_variance: Union[float, np.ndarray] = 2.,
                        smc_budget: int = 300,
                        naive_bq_budget: int = 1000,
                        naive_bq_kern_lengthscale: float = 1.,
                        naive_bq_kern_variance: float = 1.,
                        wsabi_budget: int = 100,
                        posterior_range=(-5., 5.),
                        # The axis ranges between which the posterior samples are drawn
                        posterior_eps: float = 0.02,
                        ):
        if self.gpr.dimensions > 1 and isinstance(prior_variance, float) and isinstance(prior_mean, float):
            prior_mean = np.array([prior_mean] * (self.gpr.dimensions)).reshape(-1, 1)
            prior_variance *= np.eye(self.gpr.dimensions)
        else:
            assert len(prior_mean) == self.gpr.dimensions
            assert prior_variance.shape[0] == prior_variance.shape[1]
            assert prior_variance.shape[0] == self.gpr.dimensions
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


# Code lifted from PyMC3 Tutorial site

# define a theano Op for our likelihood function
class LogLike(tt.Op):

    """
    Specify what type of object will be passed and returned to the Op when it is
    called. In our case we will be passing it a vector of values (the parameters
    that define our model) and returning a single "scalar" value (the
    log-likelihood)
    """
    itypes = [tt.dvector] # expects a vector of parameter values when called
    otypes = [tt.dscalar] # outputs a single scalar value (the log likelihood)

    def __init__(self, loglike,):# x):
        """
        Initialise the Op with various things that our log-likelihood function
        requires. Below are the things that are needed in this particular
        example.

        Parameters
        ----------
        loglike:
            The log-likelihood (or whatever) function we've defined
        data:
            The "observed" data that our log-likelihood function takes in
        x:
            The dependent variable (aka 'x') that our model requires
        sigma:
            The noise standard deviation that our function requires.
        """

        # add inputs as class attributes
        self.likelihood = loglike
        #self.x = x

    def perform(self, node, inputs, outputs):
        # the method that is used when calling the Op
        data = inputs

        # call the log-likelihood function
        logl = self.likelihood(data)

        outputs[0][0] = np.array(logl) # output the log-likelihood


# -------- Parameter Posterior -------- #
def sample_from_param_posterior(gpy_gp, num_samples=1000, hmc_iters=10,
                                save_csv=True, plot_samples=True):
    hmc = GPy.inference.mcmc.HMC(gpy_gp, stepsize=2e-2)
    t = hmc.sample(num_samples=num_samples, hmc_iters=hmc_iters)
    if plot_samples:
        _ = plt.plot(t)
        plt.show()
    if save_csv:
        df = pd.DataFrame(t, columns=gpy_gp.parameter_names_flat())
        df.to_csv('PosteriorSampling.csv')
    return t


# ---- Performance evaluation --------- #
def eval_wsabi_perf(rq: RegressionQuadrature):
    max_iters = 200
    rmse = np.empty((max_iters, ))
    for i in range(10, max_iters):
        rq.options['wsabi_budget'] = i
        rmse[i] = rq.wsabi(verbose=False)
        print('Progress:'+str(i-10)+' /'+str(max_iters-10))
    df = pd.Series(rmse, name='wsabi_rmse_v_iterations')
    df.to_csv('WSABI_perf.csv')
    print('WSABI evaluation completed')
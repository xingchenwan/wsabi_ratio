from ratio.svm import SVMClassification
from bayesquad.priors import Gaussian
import numpy as np
from typing import Union, Tuple

import matplotlib
matplotlib.use("TkAgg")
# This is to prevent a macOS bug with matplotlib
import matplotlib.pyplot as plt
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
from scipy.optimize import minimize, Bounds
from scipy.interpolate import griddata
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV


class ClassificationQuadrature:
    def __init__(self, classification_model: SVMClassification, **kwargs):
        self.model = classification_model
        self.options = self._unpack_options(**kwargs)
        self.dimensions = self.model.param_dim
        self.prior = Gaussian(mean=self.options['prior_mean'].reshape(-1),
                              covariance=self.options['prior_variance'])

    def maximum_likelihood(self):
        """
        The more Bayesian way of model selection - compute the maximum likelihood estimate
        :return:
        """

        def neg_log_lik(log_phi: np.ndarray):
            return -self.model.log_sample(np.exp(log_phi))[0]

        self.plot_log_lik()

        x_test, y_test = self.model.X_test, self.model.Y_test
        log_phi0 = np.array([0.]*self.dimensions)
        bounds = Bounds([1e-10, 1e-10], [np.inf, np.inf]) # Enforce non-negativity in gamma and C optimisation

        phi_mle = minimize(neg_log_lik, log_phi0, bounds=bounds).x
        logging.info("MLE Hyperparameter: "+str(phi_mle))
        y_pred = self.model.predict(c=np.exp(phi_mle[0]), gamma=np.exp(phi_mle[1]), x_test=x_test)
        labels = y_pred.copy()
        labels[labels < 0.5] = 0
        labels[labels >= 0.5] = 1
        accuracy, precision, recall, f1 = self.model.score(labels, y_test.reshape(-1))
        logging.info("Accuracy: "+str(accuracy))
        logging.info("Precision: "+str(precision))
        logging.info("Recall: "+str(recall))
        logging.info("F1 score: "+str(f1))

    def grid_search(self):
        """
        The frequentist approach to SVM parameter search - grid search the hyperparameters...
        :return:
        """
        pass

    def wsabi(self, verbose=True):
        # Allocating number of maximum evaluations
        start = time.time()
        budget = self.options['wsabi_budget']
        batch_count = 1
        test_x = self.model.X_test
        test_y = np.squeeze(self.model.Y_test)

        # Allocate memory of the samples and results
        log_phi = np.zeros((budget * batch_count, self.dimensions,))  # The log-hyperparameter sampling points
        log_r = np.zeros((budget * batch_count,))  # The log-likelihood function
        q = np.zeros((test_x.shape[0], budget * batch_count))  # Prediction

        # Set prior mean to the MAP value

        log_phi_initial = np.zeros(self.dimensions).reshape(1, -1)
        # log_phi_initial = self.options['prior_mean'].reshape(1, -1)
        log_r_initial = np.sqrt(2 * np.exp(self.model.log_sample(phi=np.exp(log_phi_initial).reshape(-1))[0]))
        # print(log_r_initial)
        pred = np.zeros((test_x.shape[0],))

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
            log_phi_i = np.array(select_batch(log_r_model, batch_count, "Kriging Believer")).reshape(batch_count, -1)
            log_r_i = self.model.log_sample(phi=np.exp(log_phi_i))[0]
            if verbose:
                logging.info('phi: '+str(log_phi_i)+' log_lik: '+str(log_r_i))
            log_r[i_a:i_a + batch_count] = log_r_i
            log_phi[i_a:i_a + batch_count, :] = log_phi_i
            log_r_model.update(log_phi_i, np.exp(log_r_i).reshape(1, -1))
        quad_time = time.time()

        max_log_r = max(log_r)
        r = np.exp(log_r - max_log_r)
        r_gp = GPy.models.GPRegression(log_phi[:1, :], np.sqrt(2 * r[0].reshape(1, 1)), kern)
        r_model = WarpedIntegrandModel(WsabiLGP(r_gp), self.prior)
        r_model.update(log_phi[1:, :], r[1:].reshape(-1, 1))
        r_gp.optimize()
        r_int = np.exp(np.log(r_model.integral_mean()[0]) + max_log_r)  # Model evidence
        log_r_int = np.log(r_int)  # Model log-evidence

        print("Estimate of model evidence: ", r_int, )
        print("Model log-evidence ", log_r_int)

        # Secondly, compute and marginalise the predictive distribution for each individual points
        for i_x in range(test_x.shape[0]):

            # Note that we do not active sample again for q, we just use the same samples sampled when we compute
            # the log-evidence
            _, q_initial = self.model.log_sample(phi=np.exp(log_phi_initial), x=test_x[i_x, :])

            # Initialise GPy GP surrogate for and q(\phi)r(\phi)
            # Sample for q values
            for i_b in range(budget * batch_count):
                log_phi_i = log_phi[i_b, :]
                log_r_i, q_i = self.model.log_sample(phi=np.exp(log_phi_i), x=test_x[i_x, :])
                q[i_x, i_b] = q_i

            # Enforce positivity in q
            q_x = q[i_x, :]
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
            pred[i_x] = rq_int / r_int
            logging.info('Progress: ' + str(i_x + 1) + '/' + str(test_x.shape[0]))

        labels = pred.copy()
        labels[labels < 0.5] = 0
        labels[labels >= 0.5] = 1
        labels = np.squeeze(labels)
        accuracy, precision, recall, f1 = self.model.score(np.squeeze(test_y), labels)
        non_zero = np.count_nonzero(np.squeeze(test_y) - np.squeeze(labels))
        print("------ WSABI Summary -----------")
        print("Number of mismatch: "+str(non_zero))
        print("Accuracy: " + str(accuracy))
        print("Precision: " + str(precision))
        print("Recall: " + str(recall))
        print("F1 score: " + str(f1))
        if verbose:
            print("Ground truth labels: "+str(test_y))
            print("Predictions: "+str(labels))
            print('Predictive Probabilities: '+str(pred))

        end = time.time()
        print("Active Sampling Time: ", quad_time - start)
        print("Total Time: ", end - start)
        return accuracy, precision, recall, f1

    def plot_log_lik(self):
        if self.dimensions > 2:
            raise ValueError("Visualistion is not available for higher dimension data!")
        x = np.linspace(-5, 5, 25)
        y = np.linspace(-7, 0, 25)
        xv = np.stack(np.meshgrid(x, y, indexing='ij'), axis=-1).reshape(-1, 2)
        z = np.empty(xv.shape[0])
        for i in range(xv.shape[0]):
            z[i] = self.model.log_sample(np.exp(xv[i, :]))[0]
        X, Y = np.meshgrid(x, y)
        Z = griddata(xv, z, (X, Y), method='cubic')
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        _ = ax.plot_wireframe(X, Y, Z)
        ax.set_xlabel('log(C)')
        ax.set_ylabel('log(gamma)')
        ax.set_zlabel('Log-likelihood')
        plt.show()

    def _unpack_options(self,
                        prior_mean: Union[float, np.ndarray] = 0.,
                        prior_variance: Union[float, np.ndarray] = 100.,
                        smc_budget: int = 100,
                        naive_bq_budget: int = 200,
                        naive_bq_kern_lengthscale: float = 1.,
                        naive_bq_kern_variance: float = 1.,
                        wsabi_budget: int = 100,
                        posterior_range=(-5., 5.),
                        # The axis ranges between which the posterior samples are drawn
                        posterior_eps: float = 0.02,
                        ):
        if self.model.param_dim == 1:
            prior_mean = np.array([prior_mean]).reshape(1, 1)
            prior_variance = np.array([[prior_variance]]).reshape(1, 1)
        elif self.model.param_dim > 1 and isinstance(prior_variance, float) and isinstance(prior_mean, float):
            prior_mean = np.array([prior_mean] * (self.model.param_dim)).reshape(-1, 1)
            prior_variance *= np.eye(self.model.param_dim)
        else:
            assert len(prior_mean) == self.model.param_dim
            assert prior_variance.shape[0] == prior_variance.shape[1]
            assert prior_variance.shape[0] == self.model.param_dim
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



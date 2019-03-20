# Xingchen Wan | 2018
# Implementation of the naive Bayesian quadrature for ratios using WSABI/original BQ/Monte Carlo Techniques on 1D

from bayesquad.quadrature import WarpedIntegrandModel, OriginalIntegrandModel
from bayesquad.batch_selection import select_batch
from bayesquad.gps import WsabiLGP, GP
from bayesquad.priors import Gaussian
from ratio.functions import Functions
import numpy as np
import GPy
import matplotlib.pyplot as plt
import seaborn as sns
from abc import ABC, abstractmethod
from emukit.model_wrappers.gpy_quadrature_wrappers import BaseGaussianProcessGPy, RBFGPy
from emukit.quadrature.kernels import QuadratureRBF, IntegralBounds
from emukit.quadrature.methods import VanillaBayesianQuadrature
from emukit.quadrature.loop import VanillaBayesianQuadratureLoop
import pandas as pd


class Quadrature1D(ABC):
    def __init__(self, r: Functions, q: Functions, p: Gaussian,
                 true_prediction_integral: float=None, true_evidence_integral: float=None):
        assert r.dimensions == q.dimensions, \
            "The dimensions of the numerator and denominator do not match!"
        self.dim = r.dimensions
        self.r = r
        self.q = q
        self.p = p

        self.results = None
        self.options = {}

        # Ground truth integral values
        self.true_prediction_integral = true_prediction_integral
        self.true_evidence_integral = true_evidence_integral
        if self.true_evidence_integral is None or self.true_prediction_integral is None:
            self.true_ratio = None
        else:
            self.true_ratio = self.true_prediction_integral / self.true_evidence_integral

    def plot_result(self,):
        approx_only = False
        if self.true_ratio is None:
            print("Ground truth values are not supplied - plotting the quadrature approximations only.")
            approx_only = True
        res = np.array(self.results)
        res = np.squeeze(res,)

        xi = np.arange(0, self.options['budget'], 1)
        plt.subplot(211)

        if approx_only is False:
            rmse = np.sqrt((res - self.true_ratio) ** 2)
            plt.loglog(xi, rmse, ".")
            plt.xlabel("Number of batches")
            plt.ylabel("RMSE")

        plt.subplot(212)
        plt.semilogx(xi, res, ".")
        if approx_only is False:
            plt.axhline(self.true_ratio)
        plt.xlabel("Number of batches")
        plt.ylabel("Result")
        plt.ylim(0, 1)

        # Save as a pandas dataframe
        true_ratios = np.array([self.true_ratio] * res.shape[0]).reshape(-1)
        res = np.stack([res.reshape(-1), true_ratios, rmse.reshape(-1)], axis=1)
        save = pd.DataFrame(res, columns=['Results', 'GrnTruth', 'RMSE'])
        print(save)
        save.to_csv('output/1D_results/quad_1d.csv')


    def plot_true_integrands(self, plot_range=(-5, 5, 0.1), numerator=False):
        """
        Generate plots for the ground truth integrands
        :param plot_range:
        :param numerator:
        :return:
        """
        x_i = np.arange(*plot_range).reshape(-1, 1)
        y_i = self.r.sample(x_i) * self.p(x_i)
        if numerator is False:
            plt.plot(x_i, y_i)
        else:
            y_ii = y_i * self.q.sample(x_i)
            plt.plot(x_i, y_ii)

    def plot_parameter_posterior(self, plot_range=(-5, 5, 0.1)):
        """
        Generate plots for the ground truth paramter posterior
        :param plot_range:
        :return:
        """
        if self.true_evidence_integral is None:
            print("True evidence integral is not supplied - plotting is not possible")
            return
        x_i = np.arange(*plot_range).reshape(-1, 1)
        y_i = self.r.sample(x_i) * self.p(x_i) / self.true_evidence_integral
        plt.plot(x_i, y_i)


class WsabiNBQ(Quadrature1D):
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

    def __init__(self, r: Functions, q: Functions, p: Gaussian,
                 true_prediction_integral: float = None, true_evidence_integral: float = None,
                 **options):
        super(WsabiNBQ, self).__init__(r, q, p, true_prediction_integral, true_evidence_integral)

        # Initialise the GPy GP instances and the WSABI-L model for the numerator and denominator integrands

        self.options = self._unpack_options(**options)
        self.results = [np.nan] * self.options["budget"]

    def quadrature(self):

        budget = self.options['budget']
        phis = np.empty((budget, 1))
        rs = np.empty((budget, ))
        rqs = np.empty((budget, ))

        phi_init = self.p.mean.reshape(1, 1)
        r_init = self.r.sample(phi_init)

        kern = GPy.kern.RBF(input_dim=1, variance=1., lengthscale=1.)

        rq_init = r_init * self.q.sample(phi_init)
        r_gp = GPy.models.GPRegression(phi_init, np.sqrt(2 * r_init).reshape(1, 1), kern)
        rq_gp = GPy.models.GPRegression(phi_init, np.sqrt(2 * rq_init).reshape(1, 1), kern)
        r_model = WarpedIntegrandModel(WsabiLGP(r_gp), self.p)
        rq_model = WarpedIntegrandModel(WsabiLGP(rq_gp), self.p)

        phis[0, :] = phi_init
        rs[0] = r_init
        rqs[0] = rq_init

        for i in range(1, budget):
            phi = (select_batch(r_model, 1)[0]).reshape(1, 1) # phi is 1 dimensional!
            phis[i, :] = phi
            rs[i] = self.r.sample(phi)
            rqs[i] = rs[i] * self.q.sample(phi)

            r_model.update(phi, rs[i].reshape(1, 1))
            rq_model.update(phi, rqs[i].reshape(1, 1))
            r_gp.optimize()
            rq_gp.optimize()

            rq_int_mean = rq_model.integral_mean()[0]
            r_int_mean = r_model.integral_mean()[0]
            self.results[i] = rq_int_mean / r_int_mean
            if i % 10 == 1:
                print('Samples', phi, "Numerator: ", rq_int_mean, "Denominator", r_int_mean)
                if self.options['plot_iterations']:
                    self.draw_samples(i, phis, rs, rqs, r_model, rq_model)
                    plt.show()
        return self.results[-1]

    def _unpack_options(self, kernel: GPy.kern.Kern = None,
                        budget: int = 100,
                        plot_iterations: bool = False,
                        display_step: int = 10,
                        plot_range: tuple = (-5, 5, 0.1)) -> dict:
        if kernel is None:
            kernel = GPy.kern.RBF(self.dim, variance=2, lengthscale=2)
        assert len(plot_range) == 3, "Supply a plot range in the format of (start, end, step)"
        return {
            "kernel": kernel,
            'budget': budget,
            'plot_iterations': plot_iterations,
            'display_step': display_step,
            'plot_range': plot_range,
        }

    def draw_samples(self,
                     i,
                     phi, r, rq,
                     r_model, rq_model,
                     sample_count=10,):

        x = np.linspace(-5, 5, 200).reshape(-1, 1)
        alpha_rq = rq_model.gp._alpha
        alpha_r = r_model.gp._alpha
        posterior_den = 0.5 * np.squeeze(r_model.gp._gp.posterior_samples_f(x, size=sample_count), axis=1) ** 2
        posterior_num = alpha_rq + 0.5 * np.squeeze(rq_model.gp._gp.posterior_samples_f(x, size=sample_count), axis=1) ** 2
        selected_pts = phi[:i+1]
        evaluated_den_points = (r[:i + 1])
        evaluated_num_points = (rq[:i + 1])

        #plt.figure(1, figsize=(8, 4))
        #plt.plot(test_locations, posterior_den, color='grey')
        #plt.plot(selected_pts[:-1], evaluated_den_points[:-1], "x", color='black')
        #plt.plot(selected_pts[-1], evaluated_den_points[-1], "x", color='red')
        #plt.xlabel("$\phi$")
        #plt.ylabel("$f(\phi)$")
        r_truth = np.empty((x.shape[0],))
        for i in range(x.shape[0]):
            r_truth[i] = self.r.sample(x[i]) * self.q.sample(x[i])
        r_pred = alpha_rq + 0.5 * (rq_model.gp._gp.predict_noiseless(x.reshape(-1, 1))[0]) ** 2

        plt.figure(2, figsize=(5, 5))
        plt.plot(x, posterior_num[:, 0], color='orange', alpha=0.2, label='Random GP Draws')
        plt.plot(x, posterior_num[:, 1:], color='orange', alpha=0.2,)
        plt.plot(x, r_pred, color='red', label='GP Posterior Mean')
        plt.plot(x, r_truth, color='gray', label='Ground Truth')
        plt.plot(selected_pts[:-1], evaluated_num_points[:-1], "x", color='black')
        plt.plot(selected_pts[-1], evaluated_num_points[-1], "x", color='red')
        plt.xlabel("$\phi$", labelpad=-5)
        plt.ylabel("$f(\phi)$")
        plt.yticks([])
        plt.legend()


class WsabiBQZ(Quadrature1D):

    def __init__(self, r: Functions, q: Functions, p: Gaussian,
                 true_prediction_integral: float = None, true_evidence_integral: float = None,
                 **options):
        super(WsabiBQZ, self).__init__(r, q, p, true_prediction_integral, true_evidence_integral)

        # Initialise the GPy GP instances and the WSABI-L model for the numerator and denominator integrands

        self.options = self._unpack_options(**options)
        self.results = [np.nan] * self.options["budget"]

    def quadrature(self):
        from bayesquad.quadrature import compute_mean_gp_prod_gpy_2

        budget = self.options['budget']
        phis = np.empty((budget, 1))
        rs = np.empty((budget,))
        qs = np.empty((budget,))

        phi_init = self.p.mean.reshape(1, 1)
        r_init = np.array(self.r.sample(phi_init))
        q_init = np.array(self.q.sample(phi_init))

        kern = GPy.kern.RBF(input_dim=1, variance=1., lengthscale=1.)

        r_gp = GPy.models.GPRegression(phi_init, np.sqrt(2 * r_init).reshape(1, 1), kern)
        q_gp = GPy.models.GPRegression(phi_init, np.sqrt(2 * q_init).reshape(1, 1), kern)
        r_model = WarpedIntegrandModel(WsabiLGP(r_gp), self.p)
        q_model = WarpedIntegrandModel(WsabiLGP(q_gp), self.p)
        phis[0, :] = phi_init
        rs[0] = r_init
        qs[0] = q_init

        for i in range(1, budget):
            phi = select_batch(r_model, 1)[0].reshape(1, 1)
            r = self.r.sample(phi)
            q = self.q.sample(phi)
            r_model.update(phi, r.reshape(1, 1))
            q_model.update(phi, q.reshape(1, 1))
            r_gp.optimize()
            q_gp.optimize()

            rs[i] = r
            qs[i] = q
            phis[i, :] = phi

            rq_gp = GPy.models.GPRegression(phis[:i+1, :], q_model.gp._gp.Y * r_model.gp._gp.Y, kern)
            rq_gp.optimize()

            alpha_q = q_model.gp._alpha
            alpha_r = r_model.gp._alpha

            r_int_mean = r_model.integral_mean()[0]
            n1 = alpha_r * alpha_q

            n2 = 0.5 * alpha_r * compute_mean_gp_prod_gpy_2(self.p, q_model.gp._gp, q_model.gp._gp)
            n3 = 0.5 * alpha_q * compute_mean_gp_prod_gpy_2(self.p, r_model.gp._gp, r_model.gp._gp)
            n4 = 0.25 * (compute_mean_gp_prod_gpy_2(self.p, rq_gp, rq_gp))
            rq_int_mean = n1+n2+n3+n4
            self.results[i] = rq_int_mean / r_int_mean
            if i % self.options['display_step'] == 0:
                print('Samples', phi, "Numerator: ", rq_int_mean, "Denominator", r_int_mean)
        return self.results[-1]

    def _unpack_options(self,
                        budget: int = 100,
                        display_step: int = 10,
                        plot_range: tuple = (-5, 5, 0.1),
                        histogram_sample_count: int = 50,
                        plot_iterations: bool = False) -> dict:
        assert len(plot_range) == 3, "Supply a plot range in the format of (start, end, step)"
        return {
            'budget': budget,
            'plot_iterations': plot_iterations,
            'display_step': display_step,
            'plot_range': plot_range,
            'histogram_sample_count': histogram_sample_count
        }


class NBQ(Quadrature1D):
    """
    Direct implementation of the Bayesiqn Quadrature method applied independently to both the numerator and denominator
    integrals without warping the output space as in WSABI methods.
    """
    def __init__(self, r: Functions, q: Functions, p: Gaussian,
                 true_prediction_integral: float = None, true_evidence_integral: float = None,
                 **options):
        super(NBQ, self).__init__(r, q, p, true_prediction_integral, true_evidence_integral)
        self.options = self._unpack_options(**options)
        self.results = [np.nan] * self.options["budget"]

    def quadrature(self):

        def _rp_emukit(x: np.ndarray) -> np.ndarray:
            n, d = x.shape
            res = self.r.sample(x)[0] * self.p(x)
            return np.array(res).reshape(n, 1)

        def rp_emukit():
            # Wrap around Emukit interface
            from emukit.core.loop.user_function import UserFunctionWrapper
            return UserFunctionWrapper(_rp_emukit), _rp_emukit

        budget = self.options['budget']
        phis = np.empty((budget, 1))
        rs = np.empty((budget, ))
        rqs = np.empty((budget, ))

        phi_init = self.p.mean.reshape(1, 1)
        r_init = np.array(self.r.sample(phi_init))

        kern = GPy.kern.RBF(input_dim=1, variance=0.1, lengthscale=0.5)

        rq_init = r_init * self.q.sample(phi_init)
        r_gp = GPy.models.GPRegression(phi_init, r_init.reshape(1, 1), kern)
        rq_gp = GPy.models.GPRegression(phi_init, rq_init.reshape(1, 1), kern)
        r_model = _wrap_emukit(r_gp)

        for i in range(1, budget):
            r_loop = VanillaBayesianQuadratureLoop(model=r_model)
            r_loop.run_loop(rp_emukit()[0], 1)
            phi = r_loop.loop_state.X[-1, :]
            r = r_loop.loop_state.Y[-1]
            rq = r * self.q.sample(phi)

            phis[i, :] = phi
            rs[i] = r
            rqs[i] = rq

            r_gp.set_XY(phis[1:i+1, :], rs[1:i+1].reshape(-1, 1))
            rq_gp.set_XY(phis[1:i+1, :], rqs[1:i+1].reshape(-1, 1))
            r_model = _wrap_emukit(r_gp)
            rq_model = _wrap_emukit(rq_gp)
            r_int = r_model.integrate()[0]
            q_int = rq_model.integrate()[0]
            self.results[i] = q_int / r_int
            if i % self.options['display_step'] == 1:
                print('Samples', phi, "Numerator: ", r_int, "Denominator", q_int)
                if self.options['plot_iterations']:
                    self.draw_samples(i, phis, rs, rqs, r_gp, rq_gp)
                    print('Iteration', i)
                    plt.show()
        return self.results[-1]

    def _unpack_options(self, kernel: GPy.kern.Kern = None,
                        budget: int = 100,
                        display_step: int = 10,
                        plot_range: tuple = (-5, 5, 0.1),
                        histogram_sample_count: int = 50,
                        plot_iterations: bool = False) -> dict:
        if kernel is None:
            kernel = GPy.kern.RBF(self.dim, variance=2, lengthscale=2)
        assert len(plot_range) == 3, "Supply a plot range in the format of (start, end, step)"
        return {
            "kernel": kernel,
            'budget': budget,
            'plot_iterations': plot_iterations,
            'display_step': display_step,
            'plot_range': plot_range,
            'histogram_sample_count': histogram_sample_count
        }

    def draw_samples(self, i, phis, rs, rqs, r_model, rq_model):
        x = np.arange(*self.options['plot_range']).reshape(-1, 1)

        rq_posterior = np.squeeze(rq_model.posterior_samples_f(x, size=5), axis=1)
        r_posterior = np.squeeze(r_model.posterior_samples_f(x, size=5), axis=1)
        selected_pts = phis[1:i+1]
        r_samples = rs[1:i+1]
        rq_samples = rqs[1:i+1]
        r_truth = np.empty((x.shape[0], ))
        for i in range(x.shape[0]):
            r_truth[i] = self.r.sample(x[i]) * self.p(x[i].reshape(1, 1)) * self.q.sample(x[i])
        r_pred = rq_model.predict_noiseless(x.reshape(-1, 1))[0]

        plt.figure(2, figsize=(5, 5))
        plt.plot(x, r_truth, color='grey', label='Ground Truth')
        plt.plot(x, rq_posterior[:, 0], color='orange', label='Random GP Draws', alpha=0.2)
        plt.plot(x, r_pred, color='red', label='GP Posterior Mean')
        plt.plot(x, rq_posterior[:, 1:], color='orange', alpha=0.2)
        plt.plot(selected_pts[:-1], rq_samples[:-1], "x", color='black')
        plt.plot(selected_pts[-1], rq_samples[-1], "x", color='red')
        plt.xlabel("$\phi$", labelpad=-5)
        plt.ylabel("$f(\phi)$")
        plt.legend()

def _wrap_emukit(gpy_gp: GPy.core.GP):

    """
    Wrap GPy GP around Emukit interface to enable subsequent quadrature
    :param gpy_gp:
    :return:
    """
    # gpy_gp.optimize()
    dimensions = gpy_gp.input_dim
    rbf = RBFGPy(gpy_gp.kern)
    qrbf = QuadratureRBF(rbf, integral_bounds=[(-5.,5.)] * dimensions)
    model = BaseGaussianProcessGPy(kern=qrbf, gpy_model=gpy_gp)
    method = VanillaBayesianQuadrature(base_gp=model)
    return method


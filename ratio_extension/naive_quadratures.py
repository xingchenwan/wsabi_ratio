# Xingchen Wan | 2018
# Implementation of the naive Bayesian quadrature for ratios using WSABI/original BQ/Monte Carlo Techniques

from bayesquad.quadrature import WarpedIntegrandModel, OriginalIntegrandModel
from bayesquad.batch_selection import select_batch
from bayesquad.gps import WsabiLGP, GP
from bayesquad.priors import Prior
from ratio_extension.test_functions import TrueFunctions
import numpy as np
import GPy
import matplotlib.pyplot as plt
import seaborn as sns
from abc import ABC, abstractmethod


class NaiveMethods(ABC):
    def __init__(self, r: TrueFunctions, q: TrueFunctions, p: Prior,
                 true_prediction_integral: float=None, true_evidence_integral: float=None):
        assert r.dimensions == q.dimensions, \
            "The dimensions of the numerator and denominator do not match!"
        self.dim = r.dimensions
        self.r = r
        self.q = q
        self.p = p

        self.results = None
        self.options = {}


        self.gpy_gp_den = None
        self.gpy_gp_num = None
        self.step_count = 0

        self.selected_points = None
        self.evaluated_den_points = None
        self.evaluated_num_points = None

        # Ground truth integral values
        self.true_prediction_integral = true_prediction_integral
        self.true_evidence_integral = true_evidence_integral
        self.true_ratio = self.true_prediction_integral / self.true_evidence_integral

    def quadrature(self):
        for i in range(self.options['num_batches']):
            res = self._batch_iterate()
            # print(self.results[i])
            if i % self.options['display_step'] == 0:
                print("Step: "+str(i)+": "+str(res))
            self.results[i] = res
        return self.results[-1]

    def plot_result(self,):
        approx_only = False
        if self.true_ratio is None:
            print("Ground truth values are not supplied - plotting the quadrature approximations only.")
            approx_only = True
        if np.nan in self.results:
            raise ValueError("Quadrature has not been run!")
        res = np.array(self.results)
        res = np.squeeze(res,)

        xi = np.arange(0, self.options['num_batches'], 1)
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

    @abstractmethod
    def initialise_gp(self): pass

    @abstractmethod
    def _batch_iterate(self,): pass

    def plot_samples(self,):
        if len(self.selected_points) == 0:
            raise ValueError('Quadrature has not been run yet!')
        plt.plot(self.selected_points, self.evaluated_den_points, 'x', color='b', label='Evaluated $r(\phi)$')
        plt.plot(self.selected_points, self.evaluated_num_points, 'x', color='r', label='Evaluated $r(\phi)q(\phi)$')
        plt.legend()

    @abstractmethod
    def draw_samples(self): pass

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


class NaiveWSABI(NaiveMethods):
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

    def __init__(self, r: TrueFunctions, q: TrueFunctions, p: Prior,
                 true_prediction_integral: float = None, true_evidence_integral: float = None,
                 **options):
        super(NaiveWSABI, self).__init__(r, q, p, true_prediction_integral, true_evidence_integral)

        # Initialise the GPy GP instances and the WSABI-L model for the numerator and denominator integrands
        self.gpy_gp_den = None
        self.gpy_gp_num = None
        self.model_den = None
        self.model_num = None

        self.options = self._unpack_options(**options)
        self.results = [np.nan] * self.options["num_batches"]
        self.initialise_gp()

        self.selected_points = np.zeros((self.options['num_batches'], self.dim))
        self.evaluated_den_points = np.zeros(self.options['num_batches'])
        self.evaluated_num_points = self.evaluated_den_points.copy()

    def _batch_iterate(self,):
        # Active sampling by minimising the variance of the *integrand*, and then update the corresponding Gaussian
        # Process
        batch_phi = select_batch(self.model_den, self.options['batch_size'], "Kriging Believer")
        self.selected_points[self.step_count, :] = batch_phi

        r_sample = self.r.sample(batch_phi)
        # p_sample = self.p(np.array(batch_phi))
        q_sample = self.q.sample(batch_phi)

        batch_y_den = r_sample
        self.evaluated_den_points[self.step_count] = batch_y_den
        # batch_y_den = np.sqrt(r_sample)
        self.model_den.update(batch_phi, batch_y_den)
        self.gpy_gp_den.optimize()
        # batch_y_num = np.sqrt(r_sample * self.q.sample(batch_phi))
        batch_y_num = batch_y_den * q_sample
        self.evaluated_num_points[self.step_count] = batch_y_num
        self.model_num.update(batch_phi, batch_y_num)
        self.gpy_gp_num.optimize()

        num_integral_mean, _, _ = self.model_num.integral_mean()
        den_integral_mean, _, _ = self.model_den.integral_mean()
        if self.step_count % self.options['display_step'] == 0:
            print(batch_phi, "Numerator: ", num_integral_mean, "Denominator", den_integral_mean)
            if self.options['plot_iterations']:
                self.draw_samples()
                plt.show()
        self.step_count += 1
        return num_integral_mean / den_integral_mean

    def initialise_gp(self):
        """
        Initialise the Gaussian process approximations to both the numerator and denominator
        """
        init_x = np.zeros((self.dim, ))
        r_sample = self.r.sample(init_x)
        if init_x.ndim <= 1:
            init_x = init_x.reshape(1, init_x.shape[0])
            r_sample = r_sample.reshape(1, r_sample.shape[0])
        init_y_den = np.sqrt(r_sample)
        init_y_num = np.sqrt(r_sample * self.q.sample(init_x))
        # Note the square-root warping of the y values in the denominator

        self.gpy_gp_den = GPy.core.GP(init_x, init_y_den,
                                      kernel=self.options['kernel'], likelihood=self.options['likelihood'])

        warped_gp = WsabiLGP(self.gpy_gp_den)
        self.model_den = WarpedIntegrandModel(warped_gp, self.p)
        self.gpy_gp_num = GPy.core.GP(init_x, init_y_num,
                                      kernel=self.options['kernel'], likelihood=self.options['likelihood'])
        self.model_num = WarpedIntegrandModel(WsabiLGP(self.gpy_gp_num), self.p)

    def _unpack_options(self, kernel: GPy.kern.Kern = None,
                        likelihood: GPy.likelihoods = GPy.likelihoods.Gaussian(variance=1e-10),
                        batch_size: int = 1,
                        num_batches: int = 100,
                        plot_iterations: bool = False,
                        display_step: int = 10,
                        plot_range: tuple = (-5, 5, 0.1)) -> dict:
        if kernel is None:
            kernel = GPy.kern.RBF(self.dim, variance=2, lengthscale=2)
        assert len(plot_range) == 3, "Supply a plot range in the format of (start, end, step)"
        return {
            "kernel": kernel,
            "likelihood": likelihood,
            'batch_size': batch_size,
            'num_batches': num_batches,
            'plot_iterations': plot_iterations,
            'display_step': display_step,
            'plot_range': plot_range,
        }

    def draw_samples(self,
                     sample_count=5, ):
        if self.gpy_gp_den is None or self.gpy_gp_num is None:
            raise ValueError("The GPy.GP instances need to be instantiated first!")
        test_locations = np.linspace(-5, 5, 200).reshape(-1, 1)
        posterior_den = self.gpy_gp_den.posterior_samples_f(test_locations, size=sample_count)
        posterior_num = self.gpy_gp_num.posterior_samples_f(test_locations, size=sample_count)
        selected_pts = self.selected_points[:self.step_count]
        evaluated_den_points = self.evaluated_den_points[:self.step_count + 1]
        evaluated_num_points = self.evaluated_num_points[:self.step_count + 1]

        plt.subplot(211)
        plt.plot(test_locations, posterior_den)
        plt.plot(selected_pts[:-1], evaluated_den_points[:-1], "x", color='grey')
        plt.plot(selected_pts[-1], evaluated_den_points[-1], "x", color='red')
        plt.title("Draws from Denominator Posterior")
        plt.subplot(212)
        plt.plot(test_locations, posterior_num)
        plt.plot(selected_pts[:-1], evaluated_num_points[:-1], "x", color='grey')
        plt.plot(selected_pts[-1], evaluated_num_points[-1], "x", color='red')
        plt.title("Draws from Numerator Posterior")


class NaiveBQ(NaiveMethods):
    """
    Direct implementation of the Bayesiqn Quadrature method applied independently to both the numerator and denominator
    integrals without warping the output space as in WSABI methods.
    """
    def __init__(self, r: TrueFunctions, q: TrueFunctions, p: Prior,
                 true_prediction_integral: float = None, true_evidence_integral: float = None,
                 **options):
        super(NaiveBQ, self).__init__(r, q, p, true_prediction_integral, true_evidence_integral)
        self.gpy_gp_den = None
        self.gpy_gp_num = None
        self.model_den = None
        self.model_num = None
        self.options = self._unpack_options(**options)
        self.initialise_gp()
        self.results = [np.nan] * self.options["num_batches"]

        self.selected_points = np.zeros((self.options['num_batches'], self.dim))
        self.evaluated_den_points = np.zeros(self.options['num_batches'])
        self.evaluated_num_points = self.evaluated_den_points.copy()

    def initialise_gp(self):
        init_x = np.zeros((self.dim,))
        init_y_den = self.r.sample(init_x)
        if init_x.ndim <= 1:
            init_x = init_x.reshape(1, init_x.shape[0])
            init_y_den = init_y_den.reshape(1, init_y_den.shape[0])
        init_y_num = init_y_den * self.q.sample(init_x)

        self.gpy_gp_den = GPy.core.GP(init_x, init_y_den, kernel=self.options['kernel'],
                                      likelihood=self.options['likelihood'])
        self.gpy_gp_num = GPy.core.GP(init_x, init_y_num, kernel=self.options['kernel'],
                                      likelihood=self.options['likelihood'])
        self.model_den = OriginalIntegrandModel(GP(self.gpy_gp_den), self.p)
        self.model_num = OriginalIntegrandModel(GP(self.gpy_gp_num), self.p)

    def _unpack_options(self, kernel: GPy.kern.Kern = None,
                        likelihood: GPy.likelihoods = GPy.likelihoods.Gaussian(variance=1e-10),
                        batch_size: int = 1,
                        num_batches: int = 100,
                        display_step: int = 10,
                        plot_range: tuple = (-5, 5, 0.1),
                        histogram_sample_count: int = 50,
                        plot_iterations: bool = False) -> dict:
        if kernel is None:
            kernel = GPy.kern.RBF(self.dim, variance=2, lengthscale=2)
        assert len(plot_range) == 3, "Supply a plot range in the format of (start, end, step)"
        return {
            "kernel": kernel,
            "likelihood": likelihood,
            'batch_size': batch_size,
            'num_batches': num_batches,
            'plot_iterations': plot_iterations,
            'display_step': display_step,
            'plot_range': plot_range,
            'histogram_sample_count': histogram_sample_count
        }

    def _batch_iterate(self,):
        batch_phi = select_batch(self.model_den, self.options['batch_size'], 'Kriging Believer')
        self.selected_points[self.step_count, :] = batch_phi
        batch_y_den = self.r.sample(batch_phi)
        batch_y_num = batch_y_den * self.q.sample(batch_phi)
        self.model_den.update(batch_phi, batch_y_den)
        self.model_num.update(batch_phi, batch_y_num)
        self.gpy_gp_num.optimize()
        self.gpy_gp_den.optimize()
        self.evaluated_den_points[self.step_count] = batch_y_den
        self.evaluated_num_points[self.step_count] = batch_y_num
        num_integral_mean, _, _ = self.model_num.integral_mean()
        den_integral_mean, _, _ = self.model_den.integral_mean()
        if self.step_count % self.options['display_step'] == 1:
            print("Numerator: ", num_integral_mean, "Denominator: ",den_integral_mean)
            if self.options['plot_iterations']:
                self.draw_samples()
        self.step_count += 1
        return num_integral_mean / den_integral_mean

    def draw_samples(self,):
        if self.gpy_gp_den is None or self.gpy_gp_num is None:
            raise ValueError("The GPy.GP instances need to be instantiated first!")
        x = np.arange(*self.options['plot_range']).reshape(-1, 1)

        numerator_integrals, numerator_samples = self.model_num.sample_histogram(x, sample_count=self.options['histogram_sample_count'])
        denominator_integrals, denominator_samples = self.model_den.sample_histogram(x, sample_count=self.options['histogram_sample_count'])
        numerator_samples = np.squeeze(numerator_samples)
        denominator_samples = np.squeeze(denominator_samples)

        selected_pts = self.selected_points[:self.step_count]
        evaluated_den_points = self.evaluated_den_points[:self.step_count]
        evaluated_num_points = self.evaluated_num_points[:self.step_count]

        #print(selected_pts)
        plt.subplot(211)
        plt.plot(x, numerator_samples)
        plt.plot(selected_pts[:-1], evaluated_num_points[:-1], "x", color='grey')
        plt.plot(selected_pts[-1], evaluated_num_points[-1], "x", color='red')
        plt.xlabel("$\phi$")
        plt.ylabel("$q(\phi)r(\phi)$")

        plt.subplot(212)
        plt.plot(x, denominator_samples)
        plt.plot(selected_pts[:-1], evaluated_den_points[:-1], "x", color='grey')
        plt.plot(selected_pts[-1], evaluated_den_points[-1], "x", color='red')
        plt.xlabel("$\phi$")
        plt.ylabel("$r(\phi)$")

        plt.show()

        plt.subplot(211)
        #print(numerator_integrals)
        sns.distplot(numerator_integrals, norm_hist=True)
        num_integral_mean = np.mean(numerator_integrals)
        num_integral_std = np.std(numerator_integrals)
        plt.axvline(num_integral_mean, color='r')
        #plt.axvline(self.true_prediction_integral, color='b')
        plt.axvline(num_integral_mean+num_integral_std, color='gray')
        plt.axvline(num_integral_mean-num_integral_std, color='gray')
        plt.xlabel("$E(\int q(\phi)r(\phi)p(\phi) d\phi)$")
        plt.yticks([])

        plt.subplot(212)
        sns.distplot(denominator_integrals, norm_hist=True)
        den_integral_mean = np.mean(denominator_integrals)
        den_integral_std = np.std(denominator_integrals)
        plt.axvline(den_integral_mean, color='r')
        #plt.axvline(self.true_evidence_integral, color='b')
        plt.axvline(den_integral_mean + den_integral_std, color='gray')
        plt.axvline(den_integral_mean - den_integral_std, color='gray')
        plt.xlabel("$E(\int r(\phi)p(\phi) d\phi)$")
        plt.yticks([])

        plt.show()

        ratio = numerator_integrals / denominator_integrals
        ratio_mean = np.mean(ratio)
        ratio_std = np.std(ratio)
        sns.distplot(ratio, norm_hist=True)
        plt.axvline(ratio_mean, color='r', label='Estimated Mean')
        plt.axvline(ratio_mean-ratio_std, color='gray')
        plt.axvline(ratio_mean+ratio_std, color='gray', label='+/- 1SD')
        plt.axvline(self.true_ratio, color='b', label='True Mean')
        plt.xlabel(r"$E(\frac{\int q(\phi)r(\phi)p(\phi) d\phi}{\int r(\phi)p(\phi) d\phi)})$")
        plt.legend()
        plt.show()
# Xingchen Wan | xingchen.wan@st-annes.ox.ac.uk 2018
# An implementation of Monte Carlo Quadrature (Slice Sampling)

import numpy as np
from ratio.naive_quadratures import NaiveMethods
from ratio.functions import Functions
from bayesquad.priors import Prior
from typing import Union
import matplotlib.pyplot as plt
import seaborn as sns


class MonteCarlo(NaiveMethods):
    """
    An implementation of the multidimensional Slice Sampling Monte Carlo method - proposed by Neal 2003
    """
    def __init__(self, r: Functions, q: Functions, p: Prior,
                 true_prediction_integral: float = None, true_evidence_integral: float = None,
                 **options):
        super(MonteCarlo, self).__init__(r, q, p, true_prediction_integral, true_evidence_integral)
        self.options = self._unpack_options(**options)
        self.sample_count = self.options['num_batches']
        self.widths = self.options['width']
        self.iterations = 0
        self.results = np.zeros(self.options['num_batches'])
        # Set up a container for all the samples
        self.selected_points = np.zeros((self.sample_count, self.dim))
        self.evaluated_points = []

    def _batch_iterate(self, x: np.ndarray = None,):
        """
        Use slice sampling to draw a sample from the parameter posterior distribution
        $
        p(\phi|z_d) = \frac{r(\phi)p(\phi)}{\int r(\phi)p(\phi)d\phi}
        $
        :param x: query point
        :return:
        """
        perm = range(self.dim)
        np.random.shuffle(perm)
        log_prob = self._eval_fns_log(x, self.p, self.r)
        x_l = x.copy()
        x_r = x.copy()
        x_prime = x.copy()
        for dd in perm:
            log_prob_perturbed = log_prob + np.log(np.random.rand())
            # Perturb by log probability of the last likelihood by a random number in the range of (0, 1)
            rdm = np.random.rand()
            x_l[dd] = x[dd] - rdm * self.widths[dd]
            x_r[dd] = x[dd] + (1-rdm) * self.widths[dd]
            if self.options['step_out']:
                while self._eval_fns_log(x_l, self.p, self.r) > log_prob_perturbed:
                    x_l[dd] -= self.widths[dd]
                while self._eval_fns_log(x_r, self.p, self.r) > log_prob_perturbed:
                    x_r[dd] += self.widths[dd]
            while True:
                x_prime[dd] = np.random.rand() * (x_r[dd] - x_l[dd]) + x_l[dd]
                log_prob_x_prime = self._eval_fns_log(x_prime, self.p, self.r)
                if log_prob_x_prime > log_prob_perturbed:
                    break
                else:
                    if x_prime[dd] > x[dd]:
                        x_r[dd] = x_prime[dd]
                    elif x_prime[dd] < x[dd]:
                        x_l[dd] = x_prime[dd]
                    else:
                        raise RuntimeError("Invalid shrinkage!")
            x[dd] = x_prime[dd]
            x_l[dd] = x_prime[dd]
            x_r[dd] = x_prime[dd]

        return x

    @staticmethod
    def _eval_fns_log(x: np.ndarray, prior: Prior, *funcs: Functions) -> float:
        """
        Evaluate the log(g(\phi)) where g is:
        $
        g = \prod f(\phi) p(\phi)
        $
        at x.
        :param x: query point for function evaluation
        :param prior: p(\phi)
        :param funcs: one or more functions in term of x
        :return: result
        """
        res = 0.
        for each_func in funcs:
            res += each_func.log_sample(x)
        res += np.log(prior(x))
        return res

    def quadrature(self) -> float:
        """
        The quadrature process models the numerator and denominator separately, and hence there are two sample acquisi-
        tion processes and so the random numbers generated in the numerator and denominator in each step are different
        from each other. At each step, the volume by evaluating the maximum and minimum value of the samples acquired
        across all dimensions is also computed which is used to approximate the integral value.
        :return: float - the final evaluated integral ratio at the last evaluation step
        """
        x = self.options['initial_point']
        for i in range(self.options['num_batches']):
            # Draw a sample from the parameter posterior
            x_prime = self._batch_iterate(x)
            self.selected_points[i, :] = x_prime.copy()

            # Evaluate q(\phi) at the drawn point and add to the bag of evaluated points
            y = self.q.sample(x_prime)
            self.evaluated_points.append(y)

            x = x_prime
            # vol = self._find_volume()
            integral_mean = np.sum(self.evaluated_points) / i
            if i >= self.options['burn_in']:
                self.results[i] = integral_mean
                if i % self.options['display_step'] == 1:
                    print("Iteration " + str(i) + ": " + str(self.results[i]))
                    print('Integral Mean: '+ str(integral_mean))
                    if self.options['plot_iterations']:
                        self.draw_samples(i)
                        plt.show()
            else:
                self.results[i] = np.nan

        plt.show()
        return self.results[-1]

    def _unpack_options(self,
                        num_batches: int = 1000,
                        width: Union[float, np.ndarray] = 0.5,
                        step_out: bool = True,
                        initial_point: np.ndarray = None,
                        plot_iterations: bool = False,
                        display_step: int = 10,
                        burn_in: int = None,
                        ) -> dict:
        """
        Unpack optional keyword arguments supplied
        :param num_batches: Number of samples in the Monte Carlo method
        :param width: the size of "jump" between successive steps of MCMC
        :param step_out: as per Neal's paper (2003) on improving slice sampling
        :param initial_point: Initial sampling point of the method. Default value is the origin in the d-dimensional
        space where d is the dimensionality of the input space
        :param plot_iterations: whether to enable the visualisation of the sample acquisition process
        :param burn_in: number of initial samples to be discarded
        :return: dictionary for use of the object
        """
        if initial_point is not None:
            assert initial_point.ndim == 1
            assert initial_point.shape[0] == self.dim
        else:
            initial_point = np.array([[0.]*self.dim])
        if isinstance(width, float):
            width = np.array([width])
        if burn_in is None:
            burn_in = min(50, int(num_batches * 0.1))
        return {
            'num_batches': num_batches,
            'width': width,
            'step_out': step_out,
            'initial_point': initial_point,
            'plot_iterations': plot_iterations,
            'display_step': display_step,
            'burn_in': burn_in
        }

    def initialise_gp(self):
        raise TypeError("Invalid method for Monte Carlo quadrature!")

    def draw_samples(self, i=0):
        """Visualise the sample acquisition process in the Monte Carlo Sampler"""
        if i <= 2:
            return
        selected_pts = self.selected_points[:i+1, :]
        plt.subplot(311)
        plt.title("Parameter posterior")
        self.plot_parameter_posterior()
        sns.distplot(self.selected_points[:i+1, :], kde=True)
        plt.subplot(312)
        self.q.plot_lik((-5, 0.1, 5))
        plt.plot(selected_pts[:-self.options['display_step'], 0], self.evaluated_points[:-self.options['display_step']],
                 'x', color='grey',)
        plt.plot(selected_pts[-self.options['display_step']:, 0], self.evaluated_points[-self.options['display_step']:],
                 'x', color='red')
        plt.title("Draws from Posterior")
        plt.subplot(313)
        plt.plot(selected_pts[:i+1, 0], "x--")
        plt.title('Selected Samples')
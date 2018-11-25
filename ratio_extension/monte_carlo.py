# Xingchen Wan | xingchen.wan@st-annes.ox.ac.uk 2018
# An implementation of Monte Carlo Quadrature (Slice Sampling)

import numpy as np
from ratio_extension.naive_quadratures import NaiveMethods
from ratio_extension.test_functions import TrueFunctions
from bayesquad.priors import Prior
from typing import Union
import matplotlib.pyplot as plt


class MonteCarlo(NaiveMethods):
    """
    An implementation of the multidimensional Slice Sampling Monte Carlo method - proposed by Neal 2003
    """
    def __init__(self, r: TrueFunctions, q: TrueFunctions, p: Prior,
                 **options):
        super(MonteCarlo, self).__init__(r, q, p)
        self.options = self._unpack_options(**options)
        self.sample_count = self.options['num_batches']
        self.widths = self.options['width']
        self.iterations = 0
        self.results = np.zeros(self.options['num_batches'])
        # Set up a container for all the samples
        self.selected_points = np.zeros((self.sample_count, 2, self.dim))

    def _batch_iterate(self, x: np.ndarray = None,
                       prior: Prior = None, *fns: TrueFunctions, prev_log_prob: float = None,
                       display_step: int = 5):
        self.iterations += 1
        perm = range(self.dim)
        np.random.shuffle(perm)
        log_prob = prev_log_prob if prev_log_prob is not None else self._eval_fns_log(x, self.p, *fns)
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
                while self._eval_fns_log(x_l, self.p, *fns) > log_prob_perturbed:
                    x_l[dd] -= self.widths[dd]
                while self._eval_fns_log(x_r, self.p, *fns) > log_prob_perturbed:
                    x_r[dd] += self.widths[dd]
            while True:
                x_prime[dd] = np.random.rand() * (x_r[dd] - x_l[dd]) + x_l[dd]
                log_prob_x_prime = self._eval_fns_log(x_prime, self.p, *fns)
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
        y = self._eval_fns_log(x, self.p, *fns)
        return x, y

    @staticmethod
    def _eval_fns_log(x: np.ndarray, prior: Prior, *funcs: TrueFunctions):
        """
        Evaluate the function:
        $
        g = \prod f(\phi) p(\phi)
        $
        at x
        :param x: query point for function evaluation
        :param prior: p(\phi)
        :param funcs: one or more functions in term of x
        :return:
        """
        res = 0.
        for each_func in funcs:
            res += each_func.log_sample(x)
        res += np.log(prior(x))
        return res

    def quadrature(self,):
        """
        The quadrature process models the numerator and denominator separately, and hence there are two sample acquisi-
        tion processes and so the random numbers generated in the numerator and denominator in each step are different
        from each other. At each step, the volume by evaluating the maximum and minimum value of the samples acquired
        across all dimensions is also computed which is used to approximate the integral value.
        :return: float - the final evaluated integral ratio at the last evaluation step
        """
        x_num = self.options['initial_point']
        x_den = x_num.copy()
        y_num = None
        y_den = None
        for i in range(self.options['num_batches']):
            x_num_prime, y_num_prime = self._batch_iterate(x_num, self.p, self.r, self.q, prev_log_prob=y_num)
            x_den_prime, y_den_prime = self._batch_iterate(x_den, self.p, self.r, prev_log_prob=y_den)
            self.selected_points[i, 0, :] = x_num_prime.copy()
            self.selected_points[i, 1, :] = x_den_prime.copy()
            self.evaluated_num_points.append(np.exp(y_num_prime))
            self.evaluated_den_points.append(np.exp(y_den_prime))

            x_num = x_num_prime
            x_den = x_den_prime
            y_num = y_num_prime
            y_den = y_den_prime
            vol_num, vol_den = self._find_volume()
            num_integral_mean = np.sum(self.evaluated_num_points) * vol_num
            den_integral_mean = np.sum(self.evaluated_den_points) * vol_den
            if i >= self.options['burn_in']:
                self.results[i] = num_integral_mean / den_integral_mean
                if i % self.options['display_step'] == 1:
                    print("Iteration " + str(i) + ": " + str(self.results[i]))
                    print('Numerator Integral Mean: '+ str(num_integral_mean/i))
                    print('Denominator Integral Mean:' + str(den_integral_mean/i))
                    if self.options['plot_iterations']:
                        self.draw_samples(i)
                        self.plot_true_integrands()
                        plt.show()
            else:
                self.results[i] = np.nan
        return self.results[-1]

    def _find_volume(self,):
        """
        Compute the volume (or the higher dimensional equivalent for volume) for the Monte Carlo integration
        :return: tuple of two floats for the denominator and numerator
        """
        vol_num = 1
        vol_den = 1
        if len(self.selected_points) == 0:
            return 0, 0
        for i in range(self.dim):
            num_dim_slice = self.selected_points[:, 0, i]
            den_dim_slice = self.selected_points[:, 1, i]
            vol_num *= max(num_dim_slice) - min(num_dim_slice)
            vol_den *= max(den_dim_slice) - min(den_dim_slice)
        # print(vol_num, vol_den)
        return vol_num, vol_den

    def _unpack_options(self,
                        num_batches: int = 1000,
                        width: Union[float, np.ndarray] = 1.,
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
        selected_pts = self.selected_points[:i+1, :, :]
        plt.subplot(211)
        plt.plot(selected_pts[:-self.options['display_step'], 1, 0],
                 self.evaluated_den_points[:-self.options['display_step']], "x", color='grey')
        plt.plot(selected_pts[-self.options['display_step']:, 1, 0],
                 self.evaluated_den_points[-self.options['display_step']:], "x", color='red')
        plt.title("Draws from Denominator Posterior")
        plt.subplot(212)
        plt.plot(selected_pts[:-self.options['display_step'], 0, 0],
                 self.evaluated_num_points[:-self.options['display_step']], "x", color='grey')
        plt.plot(selected_pts[-self.options['display_step']:, 0, 0],
                 self.evaluated_num_points[-self.options['display_step']:], "x", color='red')
        plt.title("Draws from Numerator Posterior")
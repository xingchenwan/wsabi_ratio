# Xingchen Wan | xingchen.wan@st-annes.ox.ac.uk 2018
# An implementation of Monte Carlo Quadrature (Slice Sampling)

import numpy as np
from ratio_extension.naive_quadratures import NaiveMethods
from ratio_extension.test_functions import TrueFunctions
from bayesquad.priors import Prior
from typing import Union

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
        # Set up a container for all the samples
        self.samples = np.zeros((self.dim, self.sample_count))
        self.iterations = 0

    def _batch_iterate(self, display_step: int = 5, x: np.ndarray = None):
        self.iterations += 1
        perm = range(self.dim)
        np.random.shuffle(perm)
        print(x)
        last_log_prob = self._log_probability(self.q, x)
        x_l = x.copy()
        x_r = x.copy()
        x_prime = x.copy()
        for dd in perm:
            last_log_prob_perturbed = last_log_prob + np.log(np.random.rand())
            # Perturb by log probability of the last likelihood by a random number in the range of (0, 1)
            rdm = np.random.rand(1)
            x_l[dd] -= rdm * self.widths[dd]
            x_r[dd] -= (1-rdm) * self.widths[dd]
            if self.options['step_out']:
                while self._log_probability(self.q, x_l) > last_log_prob_perturbed:
                    x_l[dd] -= self.widths[dd]
                while self._log_probability(self.q, x_r) > last_log_prob_perturbed:
                    x_r[dd] -= self.widths[dd]
            while True:
                x_prime[dd] = np.random.rand() * (x_r[dd] - x_l[dd]) + x_l[dd]
                log_prob_x_prime = self._log_probability(self.q, x_prime)
                if log_prob_x_prime > last_log_prob_perturbed:
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
        if self.iterations % display_step == 0:
            print("Iteration "+str(self.iterations)+": "+str(x))
        return x

    def quadrature(self, display_step=1):
        samples = []
        x = self.options['initial_point']
        for i in range(self.options['num_batches']):
            new_x = self._batch_iterate(x=x)
            samples.append(new_x)
            x = new_x
            print(x)
        return x.sum() / (self.options['num_batches'] + .0)

    def _log_probability(self, func: TrueFunctions, x: np.ndarray) -> np.ndarray:
        if x.ndim == 2:
            assert x.shape[1] == self.dim
        else:
            assert self.dim == 1
        res = np.zeros(x.shape)
        for i in range(x.shape[0]):
            res[i, :] = np.log(func.sample(x[i]))
        return res

    def _unpack_options(self,
                        num_batches: int = 100,
                        width: Union[float, np.ndarray] = 1.,
                        step_out: bool = True,
                        initial_point: np.ndarray = None) -> dict:
        """
        Unpack optional keyword arguments supplied
        :param num_batches: Number of samples in the Monte Carlo method
        :param width:
        :param step_out:
        :param initial_point: Initial sampling point of the method. Default value is the origin in the d-dimensional
        space where d is the dimensionality of the input space
        :return: dictionary for use of the object
        """
        if initial_point is not None:
            assert initial_point.ndim == 1
            assert initial_point.shape[0] == self.dim
        else:
            initial_point = np.array([[0.]*self.dim])
        if isinstance(width, float):
            width = np.array([width])
        return {
            'num_batches': num_batches,
            'width': width,
            'step_out': step_out,
            'initial_point': initial_point
        }

    def initialise_gp(self):
        raise TypeError("Invalid method for Monte Carlo quadrature!")

    def draw_samples(self):
        raise TypeError("Invalid method for Monte Carlo quadrature!")
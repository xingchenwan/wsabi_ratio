# Xingchen Wan | xingchen.wan@st-annes.ox.ac.uk | Jan 2019

import GPy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Union, Tuple
from abc import ABC
from bayesquad.priors import Gaussian, Prior
from scipy import integrate


class Functions(ABC):
    """
    Abstract class specifying the necessary interface for valid likelihood functions
    """
    def __init__(self,):
        self.dimensions = None
        self.model = None
        self.grd_evidence = None
        self.grd_log_evidence = None
        self.grd_posterior_dist = None

    def log_sample(self, x: Union[np.ndarray, float, list]) -> float:
        pass

    def sample(self, x: Union[np.ndarray, float, list]) -> float:
        """
        Convert the log-likelihood back to likelihood
        :param x: List/array of parameters
        :return: the likelihood of the model evaluated
        """
        return np.exp(self.log_sample(x))

    def _collect_params(self) -> np.ndarray:
        pass


class Rosenbrook2D(Functions):
    """
    A sample 2D Rosenbrook likelihood function.
    Input is a 2 dimensional vector and output is the result of the Rosenbrook function queried on that point.
    """
    def __init__(self, prior: Prior = None,
                 x_range=(-10, 10), y_range=(-10, 10), eps=0.01):
        super(Rosenbrook2D, self).__init__()
        self.prior = prior
        self.x_range = x_range
        self.y_range = y_range
        self.eps = eps
        self.dimensions = 2
        self.last_x = None
        self.grd_evidence, self.grd_log_evidence = self.grd_marginal_lik()
        self.grd_posterior_dist = self.grd_posterior()

    def log_sample(self, x: Union[np.ndarray, float, list]):
        # assert len(x) == 2
        x = np.asarray(x.reshape(-1))
        self.last_x = x
        return -1./100*(x[0] - 1) ** 2 - (x[0] ** 2 - x[1]) ** 2

    # ----- Ground Truth Computations ---- #
    # These functions should not be accessed by the trial integration methods

    def _log_sample(self, x, y):
        return -1./100*(x-1)**2 - (x**2-y)**2

    def _sample(self, x, y):
        return np.exp(self._log_sample(x, y))

    def grd_marginal_lik(self,) -> Tuple[float, float]:
        # Compute the ground truth marginal likelihood
        # Implicit integration range in 2D space:
        xmin, xmax = (-10, 10)
        ymin, ymax = (-10, 10)

        _sample = self._sample
        _prior = self.prior

        def f(x: float, y: float):
            # Integrand function for subsequent integration
            xv = np.array([[x, y]])
            return _sample(x, y) * _prior(xv)

        margin_lik = integrate.dblquad(f, xmin, xmax, lambda x: ymin, lambda x: ymax)[0]
        log_margin_lik = np.log(margin_lik)
        return margin_lik, log_margin_lik

    def grd_posterior(self,) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # Compute the ground truth posterior *distribution* over the x/y ranges specified
        evidence, _ = self.grd_marginal_lik()
        x1 = np.arange(self.x_range[0], self.x_range[1], self.eps)
        x2 = np.arange(self.y_range[0], self.y_range[1], self.eps)
        xv, yv = np.meshgrid(x1, x2)
        z = self._sample(xv, yv) / evidence
        return xv, yv, z

    # ----- Ground Truth Visualisation----- #
    def plot_grd_posterior(self, grd_posterior_res: Tuple[np.ndarray, np.ndarray, np.ndarray] = None):
        if grd_posterior_res is None:
            grd_posterior_res = self.grd_posterior()
        xv, yv, z = grd_posterior_res
        _ = plt.contour(xv, yv, z)
        plt.show()

    def _collect_params(self) -> np.ndarray:
        return self.last_x


class GPRegressionFromFile(Functions):
    """
    Construct a GP regression object from file, then this object acts as a black box that outputs a likelihood function,
    from which we can build another GP surrogate to compute the marginal likelihood and the posterior distribution.
    The input should be a 8 dimensional vector (or array of vectors) comprising the 6 lengthscale hyperparameter of the
    GP regression (the original GP regression), 1 variance hyperparameter and 1 Gaussian noise hyperparameter.
    """
    def __init__(self,
                 kernel='rbf',
                 file_path="data/yacht_hydrodynamics.data.txt",
                 col_headers: tuple =
                 ('Longitudinal position of buoyancy centre',
                  'Prismatic Coefficient',
                  'Length-displacement Ratio',
                  'Beam-draught Ratio',
                  'Length-beam Ratio',
                  'Froude Number',
                  'Residuary Resistance Per Unit Weight of Displacement')
                 ):
        super(GPRegressionFromFile, self).__init__()
        self.X, self.Y = self.load_data()
        self.dimensions = self.X.shape[1] + 2
        self.kernel_option = kernel

        self.model = self.init_gp_model()
        self.file_path = file_path
        self.col_headers = col_headers

    def load_data(self, plot_graph=False):
        raw_data = pd.read_csv(filepath_or_buffer=self.file_path, header=None, sep='\s+')
        if plot_graph:
            num_cols = len(raw_data.columns)
            for i in range(num_cols):
                plt.subplot(4, 2, i+1)
                plt.plot(raw_data.index, raw_data.loc[:, i])
                plt.title(self.col_headers[i])
            plt.show()
        data_X = raw_data.iloc[:, :-1].values
        data_Y = raw_data.iloc[:, -1].values

        # Refactor to 2d array
        if data_Y.ndim == 1:
            data_Y = data_Y.reshape(-1, 1)
        if data_X.ndim == 1:
            data_X = np.array([data_X])
        assert data_X.shape[0] == data_Y.shape[0]
        return data_X, data_Y

    def init_gp_model(self):

        init_len_scale = np.array([0.2]*self.X.shape[1])
        init_var = 0.2

        if self.kernel_option == 'rbf':
            ker = GPy.kern.RBF(input_dim=self.X.shape[1],
                               lengthscale=init_len_scale,
                               variance=init_var,
                               ARD=True,)
        else:
            raise NotImplementedError()
        m = GPy.models.GPRegression(self.X, self.Y, ker)
        return m

    # ------------ Compute the marginal log-likelihood of the model -------- #
    def log_sample(self, x: Union[np.ndarray, float, list]) -> float:
        """
        Compute the log-likelihood of the given parameter array x.
        :param x: List/array of parameters. The first parameter corresponds to the model variance. The second - penul-
        timate items correspond to the model lengthscale in each dimension. The last item corresponds to the Gaussian
        noise parameter
        The length of the parameter array must be exactly 2 more than the dimensionality of the data
        :return: the log-likelihood of the model evaluated.
        """

        # Transform to exponentiated space
        x = np.asarray(np.exp(x)).reshape(-1)
        # display(self.model)
        assert len(x) == self.dimensions + 2
        # 2 extra dimensions to accommodate the Gaussian noise and model variance parameter of the RBF kernel
        self.model.rbf.variance = x[0]
        self.model.rbf.lengthscale = x[1:-1]
        self.model.Gaussian_noise.variance = x[-1]
        return self.model.log_likelihood()

    def _collect_params(self) -> np.ndarray:
        """
        Collect the parameter values into a numpy n-d array
        :return: condensed numpy array of params
        """
        res = np.array([0.]*(self.dimensions+2))
        res[0] = self.model.rbf.variance
        res[1:-1] = self.model.rbf.lengthscale
        res[-1] = self.model.Gaussian_noise.variance
        # This is the parameter we are interested in for the Diego paper
        return res

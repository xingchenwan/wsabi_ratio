# Xingchen Wan | xingchen.wan@st-annes.ox.ac.uk | Jan 2019

import GPy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Union, Tuple
from abc import ABC
from bayesquad.priors import Gaussian, Prior
from scipy import integrate
from scipy.stats import norm, multivariate_normal
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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

    def _sample_test(self, x: Union[np.ndarray, float, list]):
        x = np.asarray(x)
        if x.ndim == 0 or x.ndim == 1:
            assert self.dimensions == 1, "Scalar input is only permitted if the function is of dimensions!"
        else:
            # If plotting a list of points, the ndim of the supplied x np array must be 2
            assert x.ndim == 2
            assert x.shape[1] == self.dimensions, "Dimension of function is of" + str(self.dimensions) + \
                                                  " ,but the dimensions of input is " + str(x.shape[1])
        return x


class Unity(Functions):
    """
    A function that always return 1
    """
    def __init__(self, dim):
        super(Unity, self).__init__()
        self.dimensions = dim

    def log_sample(self, x: Union[np.ndarray, float, list]):
        x = self._sample_test(x)
        return 0

    def sample(self, x: Union[np.ndarray, float, list]):
        x = self._sample_test(x)
        return 1


class Rosenbrock2D(Functions):
    """
    A sample 2D Rosenbrook likelihood function.
    Input is a 2 dimensional vector and output is the result of the Rosenbrook function queried on that point.
    """
    def __init__(self, prior: Prior = None,
                 x_range=(-3, 3), y_range=(-3, 3), eps=0.01,
                 integration_range=(-10, 10)):
        super(Rosenbrock2D, self).__init__()
        self.prior = prior
        self.x_range = x_range
        self.y_range = y_range
        self.integration_range = integration_range
        self.eps = eps
        self.dimensions = 2
        self.grd_evidence, self.grd_log_evidence = self.grd_marginal_lik()
        self.grd_posterior_dist = self.grd_lik_posterior()

    def log_sample(self, x: Union[np.ndarray, float, list]):
        # assert len(x) == 2
        if x.ndim == 2:
            x = x.squeeze()
        res = -1./100*(x[0] - 1.) ** 2 - (x[0] ** 2 - x[1]) ** 2
        return res

    # ----- Ground Truth Computations ---- #
    # These functions should not be accessed by the trial integration methods

    def _log_sample(self, x, y):
        return -1./100*(x-1)**2 - (x**2-y)**2

    def _sample(self, x, y):
        return np.exp(self._log_sample(x, y))

    def grd_marginal_lik(self,) -> Tuple[float, float]:
        # Compute the ground truth marginal likelihood
        # Implicit integration range in 2D space:
        xmin, xmax = self.integration_range
        ymin, ymax = self.integration_range

        _sample = self._sample
        _prior = self.prior

        def f(x: float, y: float):
            # Integrand function for subsequent integration
            return _sample(x, y) * _prior.eval(x, y)

        margin_lik = integrate.dblquad(f, xmin, xmax, lambda x: ymin, lambda x: ymax)[0]
        log_margin_lik = np.log(margin_lik)
        return margin_lik, log_margin_lik

    def grd_lik_posterior(self, ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        # Compute the ground truth posterior *distribution* over the x/y ranges specified

        x1 = np.arange(self.x_range[0], self.x_range[1], self.eps)
        x2 = np.arange(self.y_range[0], self.y_range[1], self.eps)
        x_len = len(x1)
        xv, yv = np.meshgrid(x1, x2)
        lik = self._sample(xv, yv)
        posterior = self._sample(xv, yv) * self.prior.eval(xv, yv).reshape(x_len, -1) / self.grd_evidence
        return xv, yv, lik, posterior

    def grd_posterior_gaussian(self, ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Construct a multivariate Gaussian distribution of the ground posterior distribution with the first and second
        moments matched to the true posterior.
        :return: Tuple[nd.array, nd.array]: the mean and covariance of the Gaussian representation
        """
        xmin, xmax = self.x_range
        ymin, ymax = self.y_range

        mu = np.array([0, 0])
        sigma = np.zeros((2, 2))

        _sample = self._sample
        _prior = self.prior

        def mean_x(x: float, y: float):
            return x * _sample(x, y) * _prior.eval(x, y)

        def mean_y(x: float, y: float):
            return y * _sample(x, y) * _prior.eval(x, y)

        def var_x(x: float, y: float):
            return x * mean_x(x, y)

        def var_y(x: float, y: float):
            return y * mean_y(x, y)

        # def var_xy(x: float, y: float):
        #    return x * mean_y(x, y)

        # First moment
        (mu[0], mu[1]) = (integrate.dblquad(mean_x, xmin, xmax, lambda x: ymin, lambda x: ymax)[0],
                        integrate.dblquad(mean_y, xmin, xmax, lambda x: ymin, lambda x: ymax)[0])
        (sigma[0, 0], sigma[1, 1]) = \
            (integrate.dblquad(var_x, xmin, xmax, lambda x: ymin, lambda x: ymax)[0],
             integrate.dblquad(var_y, xmin, xmax, lambda x: ymin, lambda x: ymax)[0],)
             #integrate.dblquad(var_xy, xmin, xmax, lambda x: ymin, lambda x: ymax)[0],)
        return mu, sigma

    # ----- Ground Truth Visualisation----- #
    def plot_grd_posterior(self, grd_posterior_res: Tuple[np.ndarray, np.ndarray, np.ndarray] = None,):
        if grd_posterior_res is None:
            grd_posterior_res = self.grd_lik_posterior()
        xv, yv, lik, posterior = grd_posterior_res
        gauss_mu, gauss_sigma = self.grd_posterior_gaussian()

        # Plot the Gaussian distribution moment matched to the posterior distribution
        pos = np.empty(xv.shape + (2, ))
        pos[:, :, 0] = xv
        pos[:, :, 1] = yv
        z1 = multivariate_normal(gauss_mu, gauss_sigma).pdf(pos)

        # Plotting stuff
        levels = np.arange(0, 1, 0.05)

        plt.subplot(221)
        plt.title("Ground Truth Likelihood")
        CS1 = plt.contour(xv, yv, lik, levels)

        plt.subplot(222)
        plt.title("Ground Truth Parameter Posterior")
        CS2 = plt.contour(xv, yv, posterior, levels)

        plt.subplot(223)
        plt.title("Gaussian Surrogate")
        CS3 = plt.contour(xv, yv, z1, levels)
        plt.colorbar(CS3)
        plt.show()


class GPRegressionFromFile(Functions):
    """
    Construct a GP regression object from file, then this object acts as a black box that outputs a likelihood function,
    from which we can build another GP surrogate to compute the marginal likelihood and the posterior distribution.
    The input should be a 8 dimensional vector (or array of vectors) comprising the 6 lengthscale hyperparameter of the
    GP regression (the original GP regression), 1 variance hyperparameter and 1 Gaussian noise hyperparameter.
    """
    def __init__(self,
                 test_set_ratio=0.3,
                 file_path="/Users/xingchenwan/Dropbox/4YP/Codes/wsabi_ratio/data/yacht_hydrodynamics.data.txt",
                 col_headers: tuple =
                 ('Longitudinal position of buoyancy centre',
                  'Prismatic Coefficient',
                  'Length-displacement Ratio',
                  'Beam-draught Ratio',
                  'Length-beam Ratio',
                  'Froude Number',
                  'Residuary Resistance Per Unit Weight of Displacement'),
                 y_offset = 0.,
                 ):
        super(GPRegressionFromFile, self).__init__()
        self.file_path = file_path
        self.y_offset = y_offset
        self.X, self.Y = self.load_data()
        self.dimensions = self.X.shape[1]
        self.n = self.X.shape[0]

        # Split the data into test set and training sets:
        n_training = int(self.n * (1 - test_set_ratio))
        self.X_train = self.X[:n_training, :]
        self.Y_train = self.Y[:n_training]
        self.X_test = self.X[n_training:, :]
        self.Y_test = self.Y[n_training:]

        self.kernel_option = 'rbf'
        self.model = self.init_gp_model()
        self.col_headers = col_headers

    def load_data(self, plot_graph=False):
        raw_data = pd.read_csv(filepath_or_buffer=self.file_path, header=None, sep='\s+').values
        if plot_graph:
            num_cols = len(raw_data.columns)
            for i in range(num_cols):
                plt.subplot(4, 2, i+1)
                plt.plot(raw_data.index, raw_data.loc[:, i])
                plt.title(self.col_headers[i])
            plt.show()

        np.random.seed(4)
        np.random.shuffle(raw_data)

        data_X = raw_data[:, :-1]
        data_Y = raw_data[:, -1] + self.y_offset

        # Refactor to 2d array
        if data_Y.ndim == 1:
            data_Y = data_Y.reshape(-1, 1)
        if data_X.ndim == 1:
            data_X = np.array([data_X])
        assert data_X.shape[0] == data_Y.shape[0]
        return data_X, data_Y

    def init_gp_model(self):
        """
        Initialise a GP regression model
        :return: A GPy GP regression model
        """
        init_len_scale = np.array([0.2]*self.X.shape[1])
        init_var = 0.2

        if self.kernel_option == 'rbf':
            ker = GPy.kern.RBF(input_dim=self.X.shape[1],
                               lengthscale=init_len_scale,
                               variance=init_var,
                               ARD=True,)
        else:
            raise NotImplementedError()
        m = GPy.models.GPRegression(self.X_train, self.Y_train, ker)
        return m

    def set_params(self, lengthscale:np.ndarray=None, variance:np.float=None, gaussian_noise:np.float=None):
        if lengthscale is not None:
            self.model.rbf.lengthscale = lengthscale
            logger.info("Lengthscale parameter set")
        if variance is not None:
            self.model.rbf.variance = variance
            logger.info("Variance parameter set")
        if gaussian_noise is not None:
            self.model.Gaussian_noise.variance = gaussian_noise
            logger.info("Gaussian Noise parameter set")

    def log_sample(self, phi: np.ndarray, x: np.ndarray = None):
        """
        Sample on the log-likelihood surface - this is the evaluation of an "expensive function".
        If x (query points) are supplied, this function also returns the prediction value -- another "expensive function"
        :param phi: List/array of parameters. The first parameter corresponds to the model variance. The second - penul-
        timate items correspond to the model lengthscale in each dimensions. The last item corresponds to the Gaussian
        noise parameter
        The length of the parameter array must be exactly 2 more than the dimensionality of the data
        :param x: List/array of query points. There can be multiple query points supplied at the same time.
        :return: the log-likelihood of the model evaluated and the gradient
        """

        # Sanity checks on phi
        if phi.ndim == 1:
            phi = np.asarray(phi).reshape(1, -1)
        assert phi.shape[1] == self.dimensions, 'The length of the parameter vector does not match the model dimensions!, ' \
                                            'Model dimension:'+str(self.dimensions)+" but given data is of "+str(len(phi)) \
                                            + str(phi)
        assert phi.ndim == 2, "Phi needs to be a 2-dimensional array!"
        if np.any(phi < 0.):
            raise ValueError("Negative values of hyperparameter value encountered! Hyperparameter requested", phi)

        n, d = phi.shape
        log_lik = np.empty((n, ))

        # Sanity checks on x (if any)
        if x is not None:
            if x.ndim == 1:
                x = x.reshape(1, -1)
            assert x.shape[1] == self.dimensions, 'The length of the data matrix does not match the model dimensions'

        pred = np.empty((n, ))

        for i in range(n):
            # Change the parameters of the model
            self.model.rbf.lengthscale = phi[i, :]
            # Compute the log likelihood
            log_lik[i] = self.model.log_likelihood()
            # if x is supplied, now compute the prediction from the model as well
            if x is not None:
                pred[i] = self.model.predict(x)[0]

        if x is not None:
            return log_lik, pred
        return log_lik, None

    def sample(self, phi: np.ndarray, x: np.ndarray):
        return np.exp((self.log_sample(phi, x)))

    # ----- Functions for the convenience of PyMC3 invokations ----- #
    def log_lik_pymc(self, phi):
        self.model.rbf.lengthscale = phi
        return self.model.log_likelihood()

    def log_lik_grad_pymc(self, phi):
        return (self.log_sample(phi))[1]

    # ------- Utility Functions --------

    def _collect_params(self) -> np.ndarray:
        """
        Collect the parameter values into a numpy n-d array
        :return: condensed numpy array of params
        """
        res = np.array([0.]*(self.dimensions))
        res[0] = self.model.rbf.variance
        res[1:-1] = self.model.rbf.lengthscale
        res[-1] = self.model.Gaussian_noise.variance
        # This is the parameter we are interested in for the Diego paper
        return res


class GaussMixture(Functions):
    """
    A test function consists of a mixture (summation) of Gaussians (so used because it allows the evaluation of
    the integration exactly as a benchmark for other quadrature methods.
    """
    def __init__(self, means: Union[np.ndarray, float, list], covariances: Union[np.ndarray, float, list],
                 weights: Union[np.ndarray, list, float]=None):
        super(GaussMixture, self).__init__()

        self.means = np.asarray(means)
        self.covs = np.asarray(covariances)
        self.mixture_count = len(self.means)

        if self.means.ndim == 1:
            self.dimensions = 1
        else:
            self.dimensions = self.means.shape[1]
        if weights is None:
            # For unspecified weights, each individual Gaussian distribution within the mixture will receive
            # an equal weight
            weights = np.array([1./self.mixture_count]*self.mixture_count)
        self.weights = np.asarray(weights)
        assert self.means.shape[0] == self.covs.shape[0], "Mean and Covariance List mismatch!"
        assert self.means.shape[0] == self.weights.shape[0]
        assert self.weights.ndim <= 1, "Weight vector must be a 1D array!"

    def sample(self, x: Union[np.ndarray, float, list], ):
        """
        Sample from the true function either with one query point or a list of points
        :param x: the coordinate(s) of the query point(s)
        :return: the value of the true function evaluated at the query point(s)
        """
        x = self._sample_test(x)
        if x.ndim <= 1:
            y = 0
            for i in range(self.mixture_count):
                if self.dimensions == 1:
                    y += self.weights[i] * self.one_d_normal(x, self.means[i], self.covs[i])
                else:
                    y += self.weights[i] * self.multi_d_gauss(x, self.means[i], self.covs[i])
        else:
            x = np.squeeze(x, axis=1)
            y = np.zeros((x.shape[0], ))
            for i in range(self.mixture_count):
                if self.dimensions == 1:
                    y += self.weights[i] * self.one_d_normal(x, self.means[i], self.covs[i])
                else:
                    y += self.weights[i] * self.multi_d_gauss(x, self.means[i], self.covs[i])
        return y

    def log_sample(self, x: Union[np.ndarray, float, list]):
        return np.log(self.sample(x))

    @staticmethod
    def one_d_normal(x: np.ndarray, mean, var) -> np.ndarray:
        assert x.ndim == 1
        return np.array([norm.pdf(x[i], mean, var) for i in range(x.shape[0])])

    @staticmethod
    def multi_d_gauss(x: np.ndarray, mean, cov) -> np.ndarray:
        assert x.ndim == 2
        return np.array([multivariate_normal.pdf(x[i], mean=mean, cov=cov) for i in range(x.shape[0])])

    def add_gaussian(self, means: Union[np.ndarray, float], var: Union[np.ndarray, float], weight: Union[np.ndarray, float]):
        assert means.shape == self.means.shape[1:]
        assert var.shape == self.covs.shape[1:]
        self.means = np.append(self.means, means)
        self.covs = np.append(self.covs, var)
        self.weights = np.append(self.weights, weight)

    def _rebase_weight(self):
        self.weights = self.weights / np.sum(self.weights)


class ProductOfGaussianMixture(Functions):
    """
    A test function that is product of n Gaussian mixtures (defined below)
    """
    def __init__(self, *gauss_mixtures: Functions):
        super(ProductOfGaussianMixture, self).__init__()
        gauss_mixtures_dims = []
        for each_mixture in gauss_mixtures:
            assert isinstance(each_mixture, GaussMixture), "Invalid Type: GaussMixture object(s) expected"
            gauss_mixtures_dims.append(each_mixture.dimensions)
        assert len(set(gauss_mixtures_dims)) == 1, "There are different dimensions in the GaussMixture objects!"
        self.dimensions = gauss_mixtures_dims[0]
        self.gauss_mixtures = gauss_mixtures
        self.gauss_mixtures_count = len(gauss_mixtures)

    def sample(self, x: Union[np.ndarray, float, list]):
        x = self._sample_test(x)
        if x.ndim <= 1:
            y_s = np.array([each_mixture.sample(x) for each_mixture in self.gauss_mixtures])
            print(y_s)
            return np.prod(y_s)
        else:
            y_s = []
            for j in range(x.shape[0]):
                y_s.append([each_mixture.sample(x[j]) for each_mixture in self.gauss_mixtures])
            y_s = np.asarray(y_s)
            return np.prod(y_s, axis=1)

    def log_sample(self, x: Union[np.ndarray, float, list]):
        return np.log(self.sample(x))



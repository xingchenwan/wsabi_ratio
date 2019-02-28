import numpy as np
import scipy.stats
import GPy
import matplotlib
matplotlib.use("TkAgg")
# This is to prevent a macOS bug with matplotlib
import matplotlib.pyplot as plt
from functools import partial
import matplotlib.cm as cm
from abc import ABC, abstractmethod


def BOCD(data, hazard_func, model):
    maxes = np.zeros(len(data) + 1)

    R = np.zeros((len(data) + 1, len(data) + 1))
    R[0, 0] = 1

    for t, x in enumerate(data):
        # Evaluate the predictive distribution for the new datum under each of
        # the parameters.  This is the standard thing from Bayesian inference.
        predprobs = model.pdf(x)

        # Evaluate the hazard function for this interval
        H = hazard_func(np.array(range(t + 1)))

        # Evaluate the growth probabilities - shift the probabilities down and to
        # the right, scaled by the hazard function and the predictive
        # probabilities.
        R[1:t + 2, t + 1] = R[0:t + 1, t] * predprobs * (1 - H)

        # Evaluate the probability that there *was* a changepoint and we're
        # accumulating the mass back down at r = 0.
        R[0, t + 1] = np.sum(R[0:t + 1, t] * predprobs * H)

        # Renormalize the run length probabilities for improved numerical
        # stability.
        R[:, t + 1] = R[:, t + 1] / np.sum(R[:, t + 1])
        # print(R)
        # Update the parameter sets for each possible run length.
        model.update(x)

        maxes[t] = R[:, t].argmax()
    return R, maxes


def constant_hazard(lam, r):
    return 1 / lam * np.ones(r.shape)


class Model(ABC):
    def __init__(self): pass

    @abstractmethod
    def pdf(self, data): pass

    @abstractmethod
    def update(self, data): pass


class StudentT(Model):
    """Student's t predictive posterior.
    """
    def __init__(self, alpha, beta, kappa, mu):
        super(StudentT, self).__init__()
        self.alpha0 = self.alpha = np.array([alpha])
        self.beta0 = self.beta = np.array([beta])
        self.kappa0 = self.kappa = np.array([kappa])
        self.mu0 = self.mu = np.array([mu])

    def pdf(self, data):
        """PDF of the predictive posterior.
        """
        return scipy.stats.t.pdf(x=data,
                                            df=2*self.alpha,
                                            loc=self.mu,
                                            scale=np.sqrt(self.beta * (self.kappa+1) /
                                                  (self.alpha * self.kappa)))

    def update(self, data):
        """Bayesian update.
        """
        muT0 = np.concatenate((self.mu0, (self.kappa * self.mu + data) /
                                            (self.kappa + 1)))
        kappaT0 = np.concatenate((self.kappa0, self.kappa + 1.))
        alphaT0 = np.concatenate((self.alpha0, self.alpha + 0.5))
        betaT0 = np.concatenate((self.beta0,
                                    self.beta +
                                    (self.kappa * (data - self.mu)**2) /
                                    (2. * (self.kappa + 1.))))

        self.mu = muT0
        self.kappa = kappaT0
        self.alpha = alphaT0
        self.beta = betaT0

    def prune(self, t):
        """Prunes memory before t.
        """
        self.mu = self.mu[:t + 1]
        self.kappa = self.kappa[:t + 1]
        self.alpha = self.alpha[:t + 1]
        self.beta = self.beta[:t + 1]


class GaussianProcess(Model):
    """ GP predictive posterior"""
    def __init__(self, window_len=25,
                 init_log_parameters=np.array([1., 1., 1e-3]),
                 mode='MLE'):
        super(GaussianProcess, self).__init__()
        self.X = None
        self.Y = None
        self.model = None
        self.init_log_parameters = init_log_parameters
        self.window_len = window_len
        self.mode = mode

    def pdf(self, data):
        if self.mode == 'MLE':
            prob = self._pdf_point_est(data)
        elif self.mode == 'WSABI':
            raise NotImplementedError
        else:
            raise ValueError("Unrecognised Keyword!")
        return prob

    def _pdf_point_est(self, data):
        if self.model is None:
            return .5 # The model is not initialised, and we cannot do prediction here
        # Do a prediction here from the GP
        self.model.optimize()
        X_pred = np.array(self.X[-1, :] + 1)
        print(X_pred)
        pred_mean, pred_var = self.model.predict(Xnew=X_pred.reshape(1, 1))
        prob = scipy.stats.norm.pdf(data, loc=pred_mean, scale=np.sqrt(pred_var))
        return prob

    def _pdf_wsabi(self, data):
        pass

    def _init_gp(self, data: np.ndarray):
        init_Y = data.reshape(-1, 1)
        init_X = np.array([_ for _ in range(init_Y.shape[0])]).reshape(-1, 1)
        init_kern = GPy.kern.RBF(input_dim=1,
                                 lengthscale=np.exp(self.init_log_parameters[0]),
                                 variance=np.exp(self.init_log_parameters[1]))
        self.model = GPy.models.GPRegression(init_X, init_Y, kernel=init_kern)
        self.model.Gaussian_noise.variance = np.exp(self.init_log_parameters[2])
        self.X = init_X
        self.Y = init_Y

    def update(self, data: np.ndarray):
        if self.model is None:
            self._init_gp(data)
        else:
            self.Y = np.concatenate([self.Y, data.reshape(1, 1)], axis=0)
            # Concatenate the new data unto the observation array
            self.X = np.concatenate([self.X, np.array(self.X[-1]+1).reshape(1, 1)], axis=0)
            # Concatenate the index array
            if self.X.shape[0] > self.window_len: # Concatenate the array
                self.X = self.X[-self.window_len:, :]
                self.Y = self.Y[-self.window_len:, :]
            self.model.set_XY(self.X, self.Y)
            #self.model.plot()
            #plt.show()


def demo():

    def generate_normal_time_series(num, minl=50, maxl=1000):
        data = np.array([], dtype=np.float64)
        partition = np.random.randint(minl, maxl, num)
        for p in partition:
            mean = np.random.randn() * 10
            var = np.random.randn() * 1
            if var < 0:
                var = var * -1
            tdata = np.random.normal(mean, var, p)
            data = np.concatenate((data, tdata))
        return data

    data = generate_normal_time_series(4, 50, 52)
    # R, maxes = BOCD(data, partial(constant_hazard, 250), StudentT(0.1, .01, 1, 0))
    R, maxes = BOCD(data, partial(constant_hazard, 250), GaussianProcess())

    fig, ax = plt.subplots(figsize=[18, 16])
    ax = fig.add_subplot(3, 1, 1)
    ax.plot(data)
    ax = fig.add_subplot(3, 1, 2, sharex=ax)
    sparsity = 5  # only plot every fifth data for faster display
    ax.pcolor(np.array(range(0, len(R[:, 0]), sparsity)),
              np.array(range(0, len(R[:, 0]), sparsity)),
              -np.log(R[0:-1:sparsity, 0:-1:sparsity]),
              cmap=cm.Greys, vmin=0, vmax=30)
    ax = fig.add_subplot(3, 1, 3, sharex=ax)
    Nw = 10
    ax.plot(R[Nw, Nw:-1])
    plt.show()
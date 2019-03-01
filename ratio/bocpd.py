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
    R, maxes = BOCD(data, partial(constant_hazard, 250), StudentT(0.1, .01, 1, 0))
    # R, maxes = BOCD(data, partial(constant_hazard, 250), GaussianProcess())

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
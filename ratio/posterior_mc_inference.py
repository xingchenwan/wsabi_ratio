import pandas as pd
import GPy, logging
import numpy as np
import gpflow
from typing import Union
import matplotlib.pyplot as plt
from pylab import *
import pymc3 as pm
import theano

theano.config.floatX = 'float32'
theano.config.compute_test_value = 'raise'
theano.config.exception_verbosity= 'high'



class PosteriorMCSampler:
    def __init__(self, gpy_gp: GPy.core.gp):
        self.gpy_gp = gpy_gp
        self.gpflow_gp = None

    def hmc(self, num_iters: int = 10000, num_chains: int = 4, mode='gpy'):
        if mode == 'gpy':
            samples = self._hmc_gpy(num_iters, num_chains)
            if samples is None:
                samples = self._hmc_gpflow(num_iters, num_chains)
        elif mode == 'gpflow':
            samples = self._hmc_gpflow(num_iters, num_chains)
        else:
            raise ValueError("Unknown mode keyword", mode)
        # self.save_as_csv(samples)
        return samples

    def _hmc_gpy(self, num_iters, num_chains):
        #self.gpy_gp.kern.variance.set_prior(GPy.priors.Gamma.from_EV(25.,150.))
        #self.gpy_gp.kern.lengthscale.set_prior(GPy.priors.Gamma.from_EV(120., 2000.))
        self.gpy_gp.kern.lengthscale.fix()
        self.gpy_gp.kern.variance.fix()
        self.gpy_gp.Gaussian_noise.variance.fix()
        print(self.gpy_gp)
        mc = GPy.inference.mcmc.hmc.HMC(self.gpy_gp, stepsize=1e-1)
        try:
            t = mc.sample(num_samples=num_iters, hmc_iters=num_chains)
            plot(t)
            plt.show()
        except np.linalg.LinAlgError:
            t = None
            logging.warning("GPy HMC Sampling failed - switching to GPFlow")
            exit()
        return t

    def _hmc_gpflow(self, num_iters, num_chains):
        if self.gpflow_gp is None:
            self.gpflow_gp = self._translate()
        self.gpflow_gp.clear()
        # Assign prior - up to this point the project uses LogGaussian prior (0, 2) for all hyperparameters, so I will
        # leave this hard-coded for now todo: change the hard-coding!
        self.gpflow_gp.kern.period.prior = gpflow.priors.LogNormal(5, 16)
        self.gpflow_gp.kern.lengthscales.prior = gpflow.priors.LogNormal(0, 16)
        self.gpflow_gp.kern.variance.prior = gpflow.priors.LogNormal(0, 16)
        self.gpflow_gp.likelihood.variance.prior = gpflow.priors.LogNormal(-4, 16)
        self.gpflow_gp.compile()
        o = gpflow.train.AdamOptimizer(0.01)
        o.minimize(self.gpflow_gp, maxiter=15)  # start near MAP
        mc = gpflow.train.HMC()
        t = mc.sample(self.gpflow_gp, num_samples=num_iters, epsilon=0.2, lmax=30, lmin=5,logprobs=True)
        # self.plot_fn_posterior(t)
        self.save_as_csv(t)
        return t

    def _translate(self):
        """
        Translate a GPy object into a GPFlow object
        :return:
        """
        obj = self.gpy_gp
        X = obj.X   # Fetch X and y values from the GPy object
        y = obj.Y
        if isinstance(obj.kern, GPy.kern.RBF):
            # For RBF kernel, we have 3 hyperparameters: rbf.variance, rbf.lengthscale and Gaussian_noise.variance
            params = obj.param_array
            variance = params[0]
            gaussian_noise = params[-1]
            lengthscale = params[1:-1]
            # Build GPFlow object
            k = gpflow.kernels.RBF(input_dim=obj.kern.input_dim, variance=variance, lengthscales=lengthscale,
                                   ARD=obj.kern.ARD)
            m = gpflow.models.GPR(X, y, kern=k)
            m.likelihood.variance = gaussian_noise
        elif isinstance(obj.kern, GPy.kern.StdPeriodic):
            # For Periodic kernel, we have 4 hyperparameters: std_periodic.variance, std_periodic.lengthscale,
            # std_periodic.period and Gaussian_noise_variance

            # ARD1 = True: each dimension will be assigned different period parameters, ARD2 = True: each dimension
            # will be assigned different lengthscale parameters.
            if obj.kern.ARD1 or obj.kern.ARD2:
                raise NotImplementedError("GPFlow does not currently support ARD for periodic kernel")
            params = obj.param_array
            input_dim = obj.kern.input_dim
            variance, period, lengthscale, gaussian_noise = params[0], params[1], params[2], params[3]
            # Build GPFlow object
            k = gpflow.kernels.Periodic(input_dim=input_dim, variance=variance, lengthscales=lengthscale, period=period)
            m = gpflow.models.GPR(X, y, kern=k)
            m.likelihood.variance = gaussian_noise
        else:
            raise NotImplementedError("Currently only RBF and periodic kernel translation is supported!")

        gpflow.train.ScipyOptimizer().minimize(m)
        self.gpflow_gp = m
        return m

    def plot_gpflow_model(self, plot_range=(0, 1000, 1000)):
        if self.gpflow_gp is None:
            self._translate()
        xx = np.linspace(*plot_range).reshape(-1, 1)
        X = self.gpy_gp.X
        Y = self.gpy_gp.Y
        mean, var = self.gpflow_gp.predict_y(xx)
        plt.figure(figsize=(12, 6))
        plt.plot(X, Y, 'kx', mew=2)
        plt.plot(xx, mean, 'C0', lw=2)
        plt.fill_between(xx[:, 0],
                         mean[:, 0] - 2 * np.sqrt(var[:, 0]),
                         mean[:, 0] + 2 * np.sqrt(var[:, 0]),
                         color='C0', alpha=0.2)
        plt.xlim(plot_range[0], plot_range[2])
        plt.show()

    def plot_fn_posterior(self, t: Union[pd.DataFrame, pd.Series], num_samples=80):
        if self.gpflow_gp is None:
            self._translate()
        X = self.gpy_gp.X
        Y = self.gpy_gp.Y
        xx = np.linspace(0, np.max(X), 1000).reshape(-1, 1)
        plt.figure(figsize=(12, 6))

        for i, s in t.iloc[400::num_samples].iterrows():
            f = self.gpflow_gp.predict_f_samples(xx, 1, initialize=False, feed_dict=self.gpflow_gp.sample_feed_dict(s))
            plt.plot(xx, f[0, :, :], 'C0', lw=2, alpha=0.1)

        plt.plot(X, Y, ".", mew=2)
        _ = plt.xlim(xx.min(), xx.max())
        _ = plt.ylim(0, np.max(Y))
        plt.show()

    def save_as_csv(self, t: Union[pd.DataFrame, pd.Series]):
        t.to_csv('posterior_samples.csv')


class PyMC3GP:
    def __init__(self, X_train, Y_train):
        self.X_train = X_train
        self.Y_train = Y_train.reshape(-1)

    def build(self):
        with pm.Model() as model:
            w = pm.Lognormal('lengthscale', 0, 4)
            h2 = pm.Lognormal('variance', 0, 4)
            # sigma = pm.Lognormal('sigma', 0, 4)
            p = pm.Lognormal('p', 5, 4)

            f_cov = h2 * pm.gp.cov.Periodic(1, period=p, ls=w)
            gp = pm.gp.Latent(cov_func=f_cov)
            f = gp.prior('f', X=self.X_train)
            s2 = pm.Lognormal('Gaussian_noise', -4, 4)
            y_ = pm.StudentT('y', mu=f, nu=s2, observed=self.Y_train)
            #start = pm.find_MAP()
            step = pm.Metropolis()
            db = pm.backends.Text('trace')
            trace = pm.sample(2000, step, chains=1, njobs=1)# start=start)

        pm.traceplot(trace, varnames=['lengthscale', 'variance', 'Gaussian_noise'])
        plt.show()
        return trace
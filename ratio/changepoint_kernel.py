# This file should be placed on GPy.kern.src and, __init__ file in that folder also needs to be modified to include this
# file

# An implementation of the changepoint RBF kernel proposed by Garnett et al 2009
# Note that since the gradient with respect to the hyperparameters has not been implemented, optimisation and MCMC
# regimes requiring a gradient observation cannot be used for now. Note that gradient is not required for Bayesian
# quadrature algorithms (.. although they can be incorporated, but not the case for this project...)

# todo: implement a gradient observation.
# Xingchen Wan | xingchen.wan@st-annes.ox.ac.uk | Feb 2019

from GPy.kern import Kern
from GPy.core.parameterization.param import Param
from GPy.kern import RBF
import numpy as np


class ChangepointRBF(Kern):
    def __init__(self, input_dim, 
                 variance1=1., variance2=1., lengthscale1=1., lengthscale2=1., xc=1, 
                 active_dims=None):
        super(ChangepointRBF, self).__init__(input_dim, active_dims, 'chngpt')
        assert input_dim == 1, "For this kernel we assume input_dim = 1"
        self.variance1 = Param('variance1', variance1)
        self.variance2 = Param('variance2', variance2)
        self.lengthscale1 = Param('lengthscale1', lengthscale1)
        self.lengthscale2 = Param('lengthscale2', lengthscale2)
        self.rbf = RBF(input_dim=input_dim, lengthscale=1., variance=1.)
        self.xc = Param('xc', xc)
        self.add_parameters(self.variance1, self.variance2, self.lengthscale1, self.lengthscale2, self.xc)

    def parameters_changed(self):
        pass

    def K(self, X, X2):
        """Covariance matrix"""
        u1 = self.u(X)
        a1 = self.a(X)
        if X2 is None:
            u2 = u1
            a2 = a1
        else:
            u2 = self.u(X2)
            a2 = self.a(X2)
        return a1 * a2 * self.rbf.K(X=u1, X2=u2)

    def Kdiag(self, X):
        """Diagonal of covariance matrix"""
        u = self.u(X)
        a = self.a(X)
        return a * self.rbf.Kdiag(u)

    def u(self, X: np.ndarray):
        """u operation in the paper"""
        u = np.empty(X.shape)
        for i in X.shape[0]:
            if X[i] < self.xc: u[i] = X[i] / self.variance1
            else: u[i] = self.xc/self.variance1 + (X[i] - self.xc)/self.variance2
        return u

    def a(self, X: np.ndarray):
        """a operation in the paper"""
        a = np.empty(X.shape)
        for i in X.shape[0]:
            if X[i] < self.xc: a[i] = self.lengthscale1
            else: a[i] = self.lengthscale2
        return a


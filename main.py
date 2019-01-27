import matplotlib
matplotlib.use("TkAgg")
# This is to prevent a macOS bug with matplotlib
from ratio.test_1d import *
import matplotlib.pyplot as plt
from ratio.naive_quadratures import NaiveWSABI, NaiveBQ
from bayesquad.priors import Gaussian
from ratio.functions import Rosenbrock2D, GPRegressionFromFile
from ratio.monte_carlo import MonteCarlo
from ratio.regression_quadrature import *
from ratio.posterior import ParamPosterior


def one_d_example():
    r = GaussMixture(means=[-1, 2], covariances=[0.7, 2], weights=[0.1, 0.2])

    q = GaussMixture([0.5, 1.5, -1.5, -0.3, 0.2], [100, 1, 0.4, 0.2, 1], weights=[3, 0.5, 0.5, 0.2, -0.1])
    prior = Gaussian(mean=np.array([[0]]), covariance=np.array([[1]]))

    prediction = predictive_integral(r, q, prior_mean=0, prior_var=1)
    evidence = evidence_integral(r, prior_mean=0, prior_var=1)
    print(prediction, evidence, prediction/evidence)
    num, den, ratio = approx_integrals(prior, q, r)

    naive_wsabi = NaiveWSABI(r, q, prior, num_batches=200, display_step=100, plot_iterations=True,
                             true_prediction_integral=num, true_evidence_integral=den)
    naive_wsabi.quadrature()
    naive_wsabi.plot_result()
    plt.show()

    mcmc = MonteCarlo(r, q, prior, true_prediction_integral=num, true_evidence_integral=den,
                      num_batches=203, plot_iterations=True, display_step=50, )
    mcmc.quadrature()
    mcmc.plot_result()
    plt.show()


def two_d_example():
    pr = Gaussian(mean=np.array([0, 0]), covariance=np.array([[2, 0],[0, 2]]))
    rb = Rosenbrock2D(prior=pr)
    post = ParamPosterior(rb)
    post.wsabi()


def multi_d_example():
    regression_model = GPRegressionFromFile()
    rq = RegressionQuadrature(regression_model)
    #model, _ = rq.maximum_a_posterior(num_restarts=1, max_iters=1000)
    #eval_perf(rq, 'mc')
    #rq.wsabi()
    # ample_from_param_posterior(model)
    rq.bq()
    #eval_wsabi_perf(rq)


if __name__ == "__main__":
    np.set_printoptions(threshold=1000)
    multi_d_example()
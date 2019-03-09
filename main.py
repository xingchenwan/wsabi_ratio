import matplotlib
matplotlib.use("TkAgg")
# This is to prevent a macOS bug with matplotlib
from ratio.test_1d import *
import matplotlib.pyplot as plt
from ratio.naive_quadratures import NaiveWSABI, NaiveBQ
from bayesquad.priors import Gaussian
from ratio.functions import *
from ratio.monte_carlo import MonteCarlo
from ratio.regression_quadrature import *
from ratio.posterior import ParamPosterior
from ratio.posterior_mc_inference import PosteriorMCSampler


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


def yacht():
    regression_model = RBFGPRegression()
    rq = RegressionQuadrature(regression_model)
    # rq.maximum_a_posterior(num_restarts=1, max_iters=1000)

    #eval_perf(rq, 'wsabi')
    #exit()
    rq.bq()
    #rq.wsabi()
    #rq.bq()
    #rq.bq()
    #eval_wsabi_perf(rq)


def sotonmet():
    regression_model = PeriodicGPRegression(selected_cols=['Tide height (m)', 'True tide height (m)'],
                                            n_test=15, train_ratio=0.2)
    rq = RegressionQuadrature(regression_model)
    # t = PosteriorMCSampler(rq.gpr.model).hmc(num_iters=1000, mode='gpflow')
    # t.plot()
    # plt.show()
    # optimised_model, _, _ = rq.maximum_a_posterior(num_restarts=1, max_iters=1000, verbose=True)
    # rq.options['prior_mean'] = np.array(np.log(optimised_model.param_array)).reshape(-1)
    # rq.reset_prior()
    # rq.wsabi()
    eval_perf(rq, 'bq')


def svm():
    from ratio.svm import SVMClassification
    from ratio.classification_quadrature import ClassificationQuadrature
    svm = SVMClassification(n_train=100, n_test=500)
    cq = ClassificationQuadrature(svm,  wsabi_budget=200)
    cq.grid_search()
    #
    #cq.maximum_likelihood()
    #cq.wsabi(verbose=True,)


def changepoint():
    from ratio.changepoint import do_experiment
    from ratio.bocpd_gp import demo
    # from ratio.bocpd import demo
    demo()

def plot_kernel():
    import GPy
    x = np.array([[0.]])
    y = np.array([[0.]])
    kern = GPy.kern.StdPeriodic(input_dim=1, lengthscale=0.5, variance=0.5, period=1)
    gp = GPy.models.GPRegression(x, y, kern)
    samples = np.arange(0., 5., 0.01)
    res = gp.posterior_samples_f(samples.reshape(-1, 1), 5)
    plt.figure(1, figsize=(5,5))
    kern.plot(ax=plt.gca())
    plt.figure(2,figsize=(5,5))
    plt.plot(samples, np.squeeze(res))
    plt.show()

if __name__ == "__main__":
    np.set_printoptions(threshold=1000)
    plot_kernel()
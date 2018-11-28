from ratio_extension.test_functions import *
import matplotlib.pyplot as plt
from ratio_extension.naive_quadratures import NaiveWSABI, NaiveBQ
from bayesquad.priors import Gaussian
from ratio_extension.prior_1d import Gaussian1D
from ratio_extension.monte_carlo import MonteCarlo

def plot_gauss_mix(r: GaussMixture, q: GaussMixture):
    r.plot(label='$r(\phi) = p(z_d|\phi)$')
    q.plot(label='$q(\phi) = p(y_*|z_d, \phi)$')

    plt.xlabel("$\phi$")
    plt.legend()


if __name__ == "__main__":
    r = GaussMixture(means=[-1, 2], covariances=[0.7, 2], weights=[0.1, 0.2])

    q = GaussMixture([0.5, 1.5, -1.5, -0.3, 0.2], [100, 1, 0.4, 0.2, 1], weights=[3, 0.5, 0.5, 0.2, -0.1])
    prior = Gaussian(mean=np.array([[0]]), covariance=np.array([[1]]))

    prediction = predictive_integral(r, q, prior_mean=0, prior_var=1)
    evidence = evidence_integral(r, prior_mean=0, prior_var=1)
    print(prediction, evidence, prediction/evidence)
    num, den, ratio = approx_integrals(prior, q, r)

    #plot_gauss_mix(r, q)
    #plt.show()

    naive_bq = NaiveBQ(r, q, prior, num_batches=203, batch_size=1, plot_iterations=True,
                       display_step=50,
                       true_prediction_integral=num, true_evidence_integral=den)
    naive_bq.quadrature()
    naive_bq.plot_result()
    plt.show()

    #naive_wsabi = NaiveWSABI(r, q, prior, num_batches=200, display_step=100, plot_iterations=True,
    #                         true_prediction_integral=num, true_evidence_integral=den)
    #naive_wsabi.quadrature()
    #naive_wsabi.plot_result()
    #plt.show()

    mcmc = MonteCarlo(r, q, prior, true_prediction_integral=num, true_evidence_integral=den,
                      num_batches=203, plot_iterations=True, display_step=50, )
    mcmc.quadrature()
    mcmc.plot_result()
    plt.show()
    #naive_wsabi.plot_samples()
    #plt.show()



    #plot_gauss_mix(r, q)
    #naive_bq.plot_samples()
    #plt.show()
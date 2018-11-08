from ratio_extension.test_functions import *
import matplotlib.pyplot as plt

if __name__ == "__main__":
    a = GaussMixture([-5, 2], [4, 2])
    a.plot([-10, 0.01, 10])
    plt.show()

    b = GaussMixture([2, 3], [1, 2])

    posterior = predictive_integral(a, b, prior_mean=0, prior_var=1) / \
                evidence_integral(a, prior_mean=0, prior_var=1)
    print(posterior)
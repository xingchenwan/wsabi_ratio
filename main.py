from ratio_extension.test_functions import *
import matplotlib.pyplot as plt

if __name__ == "__main__":
    means = np.array([[-5], [2], [5]])
    variances = np.array([[1], [2], [1.5]])
    a = GaussMixture(means, variances)
    a.plot([-10, 0.01, 10])
    plt.show()
    print(a.evidence_integral(np.array([0]), np.array([1])))
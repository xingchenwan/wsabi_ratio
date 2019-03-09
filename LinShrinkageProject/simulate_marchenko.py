import numpy as np#
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import pandas as pd
from scipy.integrate import quad

seed = 1
np.random.seed(seed)


# Definition of the Marchenko-Pastur density
def marchenko_pastur_pdf(x, Q, sigma=1):
    y = 1 / Q
    b = np.power(sigma * (1 + np.sqrt(1 / Q)), 2)  # Largest eigenvalue
    a = np.power(sigma * (1 - np.sqrt(1 / Q)), 2)  # Smallest eigenvalue
    return (1 / (2 * np.pi * sigma * sigma * x * y)) * np.sqrt((b - x) * (x - a)) * (0 if (x > b or x < a) else 1)


def mean_marchenko(Q, sigma=1):

    def func(x, Q, sigma):
        return x * (1 / (2 * np.pi * sigma * sigma * x * 1 / Q)) * np.sqrt((b - x) * (x - a))

    b = np.power(sigma * (1 + np.sqrt(1 / Q)), 2)  # Largest eigenvalue
    a = np.power(sigma * (1 - np.sqrt(1 / Q)), 2)  # Smallest eigenvalue
    I = quad(func, a, b, args=(Q, sigma))
    return I[0]

def plot_beta(beta):
    import seaborn as sns
    import seaborn.timeseries

    def _plot_range_band(*args, central_data=None, ci=None, data=None, **kwargs):
        upper = data.max(axis=0)
        lower = data.min(axis=0)
        # import pdb; pdb.set_trace()
        ci = np.asarray((lower, upper))
        kwargs.update({"central_data": central_data, "ci": ci, "data": data})
        seaborn.timeseries._plot_ci_band(*args, **kwargs)

    beta = beta.transpose()
    beta = np.log(beta)
    beta_mean = np.mean(beta, axis=1)
    beta_var = np.std(beta, axis=1)
    print(beta_mean)
    print(beta_var)
    x = np.arange(0, beta.shape[0]*40, 40)

    fig, ax = plt.subplots()
    clrs = sns.color_palette("husl", 5)

    with sns.axes_style("darkgrid"):
        ax.plot(x, beta_mean,  color=clrs[0])
        ax.fill_between(x, beta_mean - beta_var, beta_mean + beta_var,alpha=0.3, facecolor=clrs[0])
        ax.legend()


def compare_eigenvalue_distribution(correlation_matrix, bm, Q, sigma=1, set_autoscale=False, show_top=False):
    e, _ = np.linalg.eig(correlation_matrix)  # Correlation matrix is Hermitian, so this is faster
    # than other variants of eig

    x_min = .0001 if np.power(sigma * (1 - np.sqrt(1 / Q)), 2) < .0001 else np.power(sigma * (1 - np.sqrt(1 / Q)), 2)
    x_max = np.power(sigma * (1 + np.sqrt(1 / Q)), 2)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    bins = 100
    if not show_top:
        # Clear top eigenvalue from plot
        e = e[e <= x_max + 1]
    ax.hist(e, density=True, bins=bins, align='left', label='Histogram of eiganvalues')  # Histogram the eigenvalues
    ax.axvline(bm, linestyle='--', color='orange', label='Bulk Median')
    mean = mean_marchenko(Q, sigma)
    ax.axvline(mean, linestyle='-', color='orange', label='Mean of Marchenko-Pastur')
    ax.set_autoscale_on(set_autoscale)
    print(bm, mean)

    # Plot the theoretical density
    f = np.vectorize(lambda x: marchenko_pastur_pdf(x, Q, sigma=sigma))

    x_min = .0001 if np.power(sigma * (1 - np.sqrt(1 / Q)), 2) < .0001 else np.power(sigma * (1 - np.sqrt(1 / Q)), 2)
    x_max = np.power(sigma * (1 + np.sqrt(1 / Q)), 2)

    x = np.linspace(x_min, x_max, 5000)
    ax.plot(x, f(x), linewidth=1, color='r', label='Marchenko-Pastur Distribution ($q = 1$)')
    plt.legend()


def load_matrix(path):
    raw_data = pd.read_csv(path, header=None, sep=' ').values
    assert raw_data.ndim == 2
    assert raw_data.shape[0] == raw_data.shape[1]
    mat = raw_data
    # sigma = raw_data.std()
    sigma = np.trace(mat)
    return mat, sigma


def load_bulk(path):
    bulk = pd.read_csv(path, header=None, sep=' ').values
    return bulk


#K, sigma = load_matrix('/Users/xingchenwan/Dropbox/4YP/Codes/wsabi_ratio/output/7Mar/covariance_matrix_0_50.txt')
#bm = load_bulk('/Users/xingchenwan/Dropbox/4YP/Codes/wsabi_ratio/output/7Mar/median_bulk_0_50.txt')
betas = load_bulk('/Users/xingchenwan/Dropbox/4YP/Codes/wsabi_ratio/output/7Mar/beta_.txt')
plot_beta(betas)

plt.show()
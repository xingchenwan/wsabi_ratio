import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
matplotlib.rcParams.update({'font.size': 12})
from scipy.stats import norm

def plot_1d_examples(max_iter=120, **paths):
    plt.figure(1, figsize=(8, 4))
    color_plate = ['b', 'orange', 'r', 'y', 'black']
    i = 0
    for name, path in paths.items():
        color = color_plate[i % len(color_plate)]
        data = pd.read_csv(path)
        data = data.iloc[:max_iter, :]
        assert 'Results' in data.columns
        assert 'GrnTruth' in data.columns
        assert 'RMSE' in data.columns
        x = np.arange(0, data.shape[0])
        rmse = np.log(data['RMSE'])
        roll_mean = rmse.rolling(window=10, ).mean()
        roll_std = rmse.rolling(window=10, ).std()
        plt.plot(x, roll_mean, linewidth=2, label=name, color=color,)
        plt.fill_between(x, (roll_mean - roll_std),
                             (roll_mean + roll_std), alpha=.1 , color=color, label='_')
        plt.xlim([10, np.max(x)])
        plt.xscale('log')
        plt.xlabel('Number of Samples', labelpad=-10)
        plt.ylabel('logRMSE')
        plt.legend()
        i += 1


def plot_gauss_mixture(**mixtures):
    plt.figure(1, figsize=(8, 4))
    style_plate = [None, '--']
    i = 0
    x = np.arange(-5, 5, 0.01)
    n_x = x.shape[0]
    n_mixtures = len(mixtures)
    y = np.empty((n_x, n_mixtures))
    for name, m in mixtures.items():
        style = style_plate[i % len(style_plate)]
        for j in range(n_x):
            y[j, i] = m.sample(x[j].reshape(1))
        plt.plot(x, y[:, i], label=name, color='gray', linestyle=style)
        i += 1
    plt.xlabel("$\phi$", labelpad=-5)
    #plt.ylabel("$f(\phi)$")
    plt.legend()


if __name__ == "__main__":
    from ratio.functions import GaussMixture
    #r = GaussMixture(means=[-1, 2], covariances=[0.7, 2], weights=[0.1, 0.2])

    #q = GaussMixture([0.5, 1.5, -1.5, -0.3, 0.2], [100, 1, 0.4, 0.2, 1], weights=[3, 0.5, 0.5, 0.2, -0.1])

    #mixtures = {'$r(\phi) = p(\mathbf{x_d}|\phi)$': r, '$q(\phi) = p(y^*|\mathbf{x_d}, \phi)$': q}
    #plot_gauss_mixture(**mixtures)
    #plt.show()

    paths = {'HMC': '/Users/xingchenwan/Dropbox/4YP/Codes/wsabi_ratio/output/1D_results/quad_1d_mc.csv',
             'WsabiBQZ': '/Users/xingchenwan/Dropbox/4YP/Codes/wsabi_ratio/output/1D_results/quad_1d_bqz.csv',
             'WsabiNBQ': '/Users/xingchenwan/Dropbox/4YP/Codes/wsabi_ratio/output/1D_results/quad_1d_nbq.csv',
             'NBQ': '/Users/xingchenwan/Dropbox/4YP/Codes/wsabi_ratio/output/1D_results/quad_1d_bq.csv'}
    plot_1d_examples(**paths)
    plt.show()

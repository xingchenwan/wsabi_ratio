# SVM classification
# Xingchen Wan | xingchen.wan@st-annes.ox.ac.uk | Feb 2019

from sklearn.svm import SVC
from ratio.functions import Functions
from typing import Union, Tuple
import pandas as pd, numpy as np
import random


class SVMClassification(Functions):
    def __init__(self,
                 file_path='/Users/xingchenwan/Dropbox/4YP/Codes/wsabi_ratio/data/wisconsin_cancer.txt',
                 train_ratio=0.5,
                 n_test=50):
        super(SVMClassification, self).__init__()
        self.file_path = file_path
        self.train_ratio = train_ratio
        self.n_test = n_test
        self.dimensions = 2  # 2 or 4
        self.X_test, self.Y_test, self.X_train, self.Y_train = [None]*4
        self.load_data()

    def log_sample(self, phi:np.ndarray, x:np.ndarray=None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # Sample on the log-likelihood hyperparameter surface.
        assert len(x) == self.dimensions
        if self.dimensions == 4:
            raise NotImplementedError
        else:
            svm = self.fit_svm(c=phi[0], gamma=phi[1])
            # Compute the log-likelihood of the data


    def load_data(self):
        # This load_data function is currently tailored to the Wisconsin Cancer dataset, to use it for another dataset,
        # there is a need to modify this function!
        raw_data = pd.read_csv(filepath_or_buffer=self.file_path, header=None,)
        drop_n = int(1. / self.train_ratio)
        raw_data = raw_data.iloc[::drop_n]
        raw_data.reset_index(inplace=True)
        X = raw_data.iloc[:, 1:-1]  # From second column to the penultimate column is the feature vector
        Y = raw_data.iloc[:, -1]  # The last column is the label column
        Y = Y.replace([4, 2], [1, 0])
        X = X.values
        Y = Y.values # Convert pandas dataframe to numpy array
        # 4 in the original dataset means malignancy - replace with 1 for positive cases; 2 in the original dataset for
        # benign - replace with 0 for negative cases

        # Split into train and test data
        random.seed(4)
        test_pt = np.minimum(self.n_test, Y.shape[0])
        test_idx = np.array(random.sample(range(Y.shape[0]), test_pt))
        self.X_test = X[test_idx]
        self.Y_test = Y[test_idx].reshape(-1, 1)
        self.X_train = X[~test_idx]
        self.Y_train = Y[~test_idx].reshape(-1, 1)

    def fit_svm(self, c, gamma=None, ):
        if gamma is None:
            gamma = 'scaled'
        clf = SVC(C=c, gamma=gamma, probability=True)
        clf.fit(self.X_train, self.Y_train)
        return clf

    def reset_params(self):
        pass

    def set_params(self):
        pass

    def _compute_log_likelihood(self):
        pass







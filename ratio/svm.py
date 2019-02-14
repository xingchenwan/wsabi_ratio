# SVM classification
# Xingchen Wan | xingchen.wan@st-annes.ox.ac.uk | Feb 2019

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from ratio.functions import Functions
from typing import Union, Tuple
import pandas as pd, numpy as np
import logging
import random
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SVMClassification(Functions):
    """
    Bayesian Support Vector Machine Classification
    """
    def __init__(self,
                 file_path='/Users/xingchenwan/Dropbox/4YP/Codes/wsabi_ratio/data/breast-cancer-wisconsin.data.txt',
                 train_ratio=0.2,
                 n_test=30):
        super(SVMClassification, self).__init__()
        self.file_path = file_path
        self.train_ratio = train_ratio
        self.n_test = n_test
        self.n_partion = 10  # number of partition in the splitting train data part
        self.param_dim = 2  # 2 or 4
        self.X_test, self.Y_test, self.X_train, self.Y_train = [None]*4
        self.load_data()

    def log_sample(self, phi:np.ndarray, x:np.ndarray=None) -> Tuple[np.ndarray, np.ndarray]:
        # Sample on the log-likelihood hyperparameter surface.
        if phi.ndim == 2:
            phi = phi.reshape(-1)
        elif phi.ndim > 2:
            raise ValueError("Invalid input type dimension")
        assert len(phi) == self.param_dim
        if self.param_dim == 4:
            raise NotImplementedError
        else:
            # Compute the log-likelihood of the data
            log_lik = self.compute_n_fold_log_lik(c=phi[0], gamma=phi[1])
            if x is not None:
                pred = self.predict(c=phi[0], gamma=phi[1], x_test=x)
            else:
                pred = None
        return log_lik, pred

    def load_data(self):
        # This load_data function is currently tailored to the Wisconsin Cancer dataset, to use it for another dataset,
        # there is a need to modify this function!
        raw_data = pd.read_csv(filepath_or_buffer=self.file_path, header=None,)
        drop_n = int(1. / self.train_ratio)
        raw_data = raw_data.iloc[::drop_n]
        raw_data.reset_index(inplace=True)
        X = raw_data.iloc[:, 2:-1]  # From second column to the penultimate column is the feature vector
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
        self.X_test = X[test_idx, :]
        self.Y_test = Y[test_idx].reshape(-1, 1)
        self.X_train = np.delete(X, test_idx, axis=0)
        self.Y_train = np.delete(Y, test_idx, axis=0)
        logging.info("Training set length "+str(self.X_train.shape[0]))
        logging.info("Test set length "+str(self.X_test.shape[0]))

    def compute_n_fold_log_lik(self, c, gamma):
        """
        Compute the N-fold log likelihood. The function will first separate the training data into disjoint, equally
        sized sets with the number of sets specified as n_partition. Then the function will compute the log-likelihood
        by iterating through the partitions, fitting the SVM model using all training data *except the data within the
        current partition*, and compute the log-likelihood from the *current partition*. The log-likelihoods obtained
        in the iteration will be summed up and returned as the N-fold log-likelihood.
        :param c: regularisation coefficient
        :param gamma: kernel parameter for the RBF kernel
        :return:
        """
        partition_size = int(self.X_train.shape[0] / self.n_partion)
        log_lik = 0.
        for i_partition in range(self.n_partion):
            hold_out_idx = np.arange(i_partition*partition_size,
                                     np.minimum((i_partition+1)*partition_size, self.X_train.shape[0]))
            X_holdout = self.X_train[hold_out_idx, :]
            y_holdout = self.Y_train[hold_out_idx]
            X_ex_holdout = np.delete(self.X_train, hold_out_idx, 0)
            y_ex_holdout = np.delete(self.Y_train, hold_out_idx, 0)
            log_lik += self.compute_log_lik(X_ex_holdout, y_ex_holdout, X_holdout, y_holdout, c, gamma)
        return log_lik

    def compute_log_lik(self, x_train, y_train,
                        x_test, y_test, c, gamma=None, ):
        clf = SVC(C=c, gamma=gamma, probability=True, kernel='rbf')
        clf.fit(x_train, y_train)

        # Compute the class probability from Platt Scaling
        assert x_test.shape[0] == y_test.shape[0]
        probs = clf.predict_log_proba(x_test)

        # Summing over all data points - note that if ground truth is +1, we sum the predictive probability of positive
        # class, otherwise we sum over the negative class
        pos_idx = (y_test == 1).reshape(-1)
        pos_class_log_prob = probs[pos_idx, 1]
        neg_class_log_prob = probs[~pos_idx, 0]

        log_lik = np.sum(pos_class_log_prob) + np.sum(neg_class_log_prob)
        return log_lik

    def predict(self, c, gamma, x_test):
        """
        Give prediction label and class probability
        :param c:
        :param gamma:
        :param x_test:
        :return:
        """
        assert x_test.ndim <= 2
        if x_test.ndim == 1:
            x_test = x_test.reshape(1, -1)
        clf = SVC(C=c, gamma=gamma, probability=True, kernel='rbf')
        clf.fit(self.X_train, self.Y_train)
        pred_prob = clf.predict_proba(x_test)[:, 1]#
        # Only return the probability of the positive class; since the problem is binary the predictive probability of
        # the negative class is simply 1 - P(pos)
        return pred_prob

    @staticmethod
    def score(y_test, y_label):
        """
        This function computes a range of metrics aiming to comprehensively evaluate the quality of the classification
        :return: score array
        """
        assert y_test.shape == y_label.shape
        accuracy = accuracy_score(y_label, y_test)
        precision = precision_score(y_label, y_test)
        recall = recall_score(y_label, y_test)
        f1 = f1_score(y_label, y_test)
        return accuracy, precision, recall, f1








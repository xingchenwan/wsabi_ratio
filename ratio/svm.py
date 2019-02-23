# SVM classification
# Xingchen Wan | xingchen.wan@st-annes.ox.ac.uk | Feb 2019

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from ratio.functions import Functions
from typing import Union, Tuple
from sklearn.model_selection import GridSearchCV
import pandas as pd, numpy as np
import logging
import random
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SVMClassification(Functions):
    """
    Bayesian (Binary) Support Vector Machine Classification
    """
    def __init__(self,
                 dataset='cancer',
                 file_path='/Users/xingchenwan/Dropbox/4YP/Codes/wsabi_ratio/data/breast-cancer-wisconsin.data.txt',
                 n_train=100,
                 n_test=30,
                 param_dim=4,
                 n_partition=5):
        super(SVMClassification, self).__init__()
        self.data_set = dataset
        self.file_path = file_path
        self.n_train = n_train
        self.n_test = n_test
        self.n_partition = n_partition  # number of partition in the splitting train data part
        self.param_dim = param_dim
        # 2 or 4 - if 2, then only the kernel parameter and the regularisation will be included as hyperparameters;
        # otherwise, the alpha and beta term in Platt estimation will be also included in the marginalisation process.
        self.X_test, self.Y_test, self.X_train, self.Y_train = [None]*4
        self.load_data()

    def log_sample(self, phi:np.ndarray, x:np.ndarray=None) -> Tuple[np.ndarray, np.ndarray]:
        # Sample on the log-likelihood hyperparameter surface.
        if phi.ndim == 2:
            phi = phi.reshape(-1)
        elif phi.ndim > 2:
            raise ValueError("Invalid input type dimension")
        assert len(phi) == self.param_dim
        pred = None
        if self.param_dim == 4:
            log_lik = self.compute_n_fold_log_lik(*phi)#
            if x is not None:
                pred = self.predict(c=phi[0], gamma=phi[1], alpha=phi[2], beta=phi[3], x_test=x)
        else:
            # Compute the log-likelihood of the data
            log_lik = self.compute_n_fold_log_lik(c=phi[0], gamma=phi[1])
            if x is not None:
                pred = self.predict(c=phi[0], gamma=phi[1], x_test=x)
        return log_lik, pred

    def load_data(self):
        if self.data_set == 'cancer':
            self._load_data_cancer()
        elif self.data_set == 'heart':
            raise self._load_data_heart()
        else:
            raise ValueError("The dataset keyword is not specified")

    # Load data functions for different data sets
    def _load_data_heart(self):
        # load_data function is tailored to Heart Disease Data Set.
        pass

    def _load_data_cancer(self):
        # This load_data function is currently tailored to the Wisconsin Cancer dataset, to use it for another dataset,
        # there is a need to modify this function!
        random.seed(1)
        np.random.seed(1)

        raw_data = pd.read_csv(filepath_or_buffer=self.file_path, header=None,)
        raw_data = raw_data.iloc[:, 2:]
        raw_data.iloc[:, -1] = raw_data.iloc[:, -1].replace([4, 2], [1, 0])
        raw_data = raw_data.values
        np.random.shuffle(raw_data)

        X = raw_data[:, :-1]  # From second column to the penultimate column is the feature vector
        Y = raw_data[:, -1]  # The last column is the label column
        # 4 in the original dataset means malignancy - replace with 1 for positive cases; 2 in the original dataset for
        # benign - replace with 0 for negative cases

        # Split into train and test data
        pts = int(np.minimum(self.n_test+self.n_train, Y.shape[0]))
        idx = np.array(random.sample(range(Y.shape[0]), pts))
        X = X[idx, :]
        Y = Y[idx]
        self.X_test = X[:self.n_test, :]
        self.Y_test = Y[:self.n_test].reshape(-1, 1)
        self.X_train = X[self.n_test:, :]
        self.Y_train = Y[self.n_test:].reshape(-1, 1)
        logging.info("Training set length "+str(self.X_train.shape[0]))
        logging.info("Test set length "+str(self.X_test.shape[0]))

    def compute_n_fold_log_lik(self, c, gamma, alpha=None, beta=None):
        """
        Compute the N-fold log likelihood. The function will first separate the training data into disjoint, equally
        sized sets with the number of sets specified as n_partition. Then the function will compute the log-likelihood
        by iterating through the partitions, fitting the SVM model using all training data *except the data within the
        current partition*, and compute the log-likelihood from the *current partition*. The log-likelihoods obtained
        in the iteration will be summed up and returned as the N-fold log-likelihood.
        :param c: regularisation coefficient
        :param gamma: kernel parameter for the RBF kernel
        :param alpha
        :param beta: Sigmoid parameter in Platt scaling
        :return:
        """
        partition_size = int(self.X_train.shape[0] / self.n_partition)
        log_lik = 0.
        for i_partition in range(self.n_partition):
            hold_out_idx = np.arange(i_partition*partition_size,
                                     np.minimum((i_partition+1)*partition_size, self.X_train.shape[0]))
            X_holdout = self.X_train[hold_out_idx, :]
            y_holdout = self.Y_train[hold_out_idx]
            X_ex_holdout = np.delete(self.X_train, hold_out_idx, 0)
            y_ex_holdout = np.delete(self.Y_train, hold_out_idx, 0)
            if self.param_dim == 2:
                log_lik += self.compute_log_lik(X_ex_holdout, y_ex_holdout, X_holdout, y_holdout, c, gamma)
            elif alpha is not None and beta is not None: # Full marginalisation
                log_lik += self.compute_full_log_lik(X_ex_holdout, y_ex_holdout, X_holdout, y_holdout,
                                                     c, gamma, alpha, beta)
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

    def compute_full_log_lik(self, x_train, y_train, x_test, y_test, c, gamma, alpha, beta):
        # compute log likelihood from a full set of hyperparameters including alpha/beta - the sigmoid coefficients
        clf = SVC(C=c, gamma=gamma, probability=False, kernel='rbf')
        clf.fit(x_train, y_train)
        decision_evals = clf.decision_function(x_test)
        #print(decision_evals)
        probs = self._sigmoid(decision_evals, alpha, beta)
        #print(alpha, beta)
        #print(probs)
        pos_idx = (y_test == 1).reshape(-1)
        pos_class_log_prob = np.log(probs[pos_idx])
        neg_class_log_prob = np.log(1 - probs[~pos_idx])

        log_lik = np.sum(pos_class_log_prob) + np.sum(neg_class_log_prob)
        return log_lik

    @staticmethod
    def _sigmoid(f: np.ndarray, alpha: float, beta: float) -> np.ndarray:
        return 1 / (1 + np.exp(-f * alpha + beta))

    def predict(self, c, gamma, x_test, alpha=None, beta=None):
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
        if alpha is None or beta is None:
            clf = SVC(C=c, gamma=gamma, probability=True, kernel='rbf')
            clf.fit(self.X_train, self.Y_train)
            pred_prob = clf.predict_proba(x_test)[:, 1]#
            # Only return the probability of the positive class; since the problem is binary the predictive probability
            # of the negative class is simply 1 - P(pos)
        else:
            clf = SVC(C=c, gamma=gamma, kernel='rbf')
            clf.fit(self.X_train, self.Y_train)
            pred_decision_evals = clf.decision_function(x_test)
            pred_prob = self._sigmoid(pred_decision_evals, alpha, beta)
        return pred_prob

    @staticmethod
    def score(y_test, y_label):
        """
        This function computes a range of metrics aiming to comprehensively evaluate the quality of the classification
        :return: score array
        """
        if y_label.shape != y_test.shape:
            y_test = np.squeeze(y_test)
            y_label = np.squeeze(y_label)
        assert y_test.shape == y_label.shape
        accuracy = accuracy_score(y_label, y_test)
        precision = precision_score(y_label, y_test)
        recall = recall_score(y_label, y_test)
        f1 = f1_score(y_label, y_test)
        return accuracy, precision, recall, f1

    def grid_search(self, bounds, n_points, objective):
        """
        Do a grid search based on some objective function - example: precision/accuracy etc.
        :param bounds:
        :param n_points:
        :param objective:
        :return:
        """
        gammas = np.exp(np.linspace(bounds[0][0], bounds[0][1], n_points))
        cs = np.exp(np.linspace(bounds[1][0], bounds[1][1], n_points))

        tuned_params = [{'kernel': ['rbf'],
                         'gamma': gammas,
                         'C': cs}]
        clf = GridSearchCV(SVC(), tuned_params, cv=self.n_partition, scoring=objective)
        clf.fit(self.X_train, self.Y_train)
        # print('Best parameter set:', clf.best_params_)
        y_pred = clf.predict(self.X_test)
        log_lik = self.compute_n_fold_log_lik(clf.best_params_['C'], clf.best_params_['gamma'])
        return log_lik, (np.log(clf.best_params_['C']), np.log(clf.best_params_['gamma'])), \
               self.score(y_pred, self.Y_test)



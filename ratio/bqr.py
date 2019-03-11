from ratio.functions import RBFGPRegression, PeriodicGPRegression
from bayesquad.priors import Gaussian
import numpy as np
from typing import Union, Tuple

import matplotlib
matplotlib.use("TkAgg")
# This is to prevent a macOS bug with matplotlib
import matplotlib.pyplot as plt
#  import seaborn as sns
#  import theano.tensor as tt
import pandas as pd

import GPy
from bayesquad.quadrature import WarpedIntegrandModel, WsabiLGP, WarpedGP, GP
from bayesquad.batch_selection import select_batch
from IPython.display import display

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
import time

def bqr_map(budget, test_x, test_y):
    # Allocating number of maximum evaluations
    start = time.time()
    dimensions = test_x.shape[1]

    # Allocate memory of the samples and results
    log_phi = np.zeros((budget, dimensions,))  # The log-hyperparameter sampling points
    log_r = np.zeros((budget,))  # The log-likelihood function
    q = np.zeros((test_x.shape[0], budget))  # Prediction
    var = np.zeros((test_x.shape[0], budget))  # Posterior variance


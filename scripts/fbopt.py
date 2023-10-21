import random
import numpy as np
from scipy.optimize import nnls, lsq_linear
from sklearn.base import BaseEstimator, RegressorMixin


class FeatureBaggingOptRegressor(BaseEstimator, RegressorMixin):

    def __init__(self, M, B, m=None, alpha=None, strategy="nnls", seed=None):
        pass
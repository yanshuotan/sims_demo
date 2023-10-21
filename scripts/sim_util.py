import copy
from functools import partial

import numpy as np

from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV
from sklearn.metrics import mean_squared_error

from .sim_util_dgp import sample_normal_X, linear_model
from .fbopt import FeatureBaggingOptRegressor


def run_one_dgp_iter(dgp_params_dict, model_params_dicts, metric=mean_squared_error,
                     testset=False, iter_num=None):

    make_X, make_y_train, make_y_test = make_dgp(dgp_params_dict)
    X_train = make_X()
    y_train = make_y_train(X_train)
    if testset:
        X_test = make_X()
    else:
        X_test = X_train
    y_test = make_y_test(X_test)
    results = []
    for model_params_dict in model_params_dicts:
        model = make_model(model_params_dict, iter_num)
        score = get_model_score(X_train, y_train, X_test, y_test, model, metric)
        result_entry = {**dgp_params_dict, **model_params_dict, "score": score}
        results.append(result_entry)
    return results


def get_model_score(X_train, y_train, X_test, y_test, model, metric):
    model.fit(X_train, y_train)
    preds_test = model.predict(X_test)
    score = metric(y_test, preds_test)
    return score


def make_model(params_dict, iter_num):
    if params_dict["model_type"] == "ols":
        model = LinearRegression()
    elif params_dict["model_type"] == "ridge":
        model = RidgeCV(alphas=np.logspace(-3, 3, 50))
    elif params_dict["model_type"] == "lasso":
        model = LassoCV()
    elif params_dict["model_type"] in ["fbopt-minangle", "fbopt-ols", "fbopt-nnls"]:
        strategy = params_dict["model_type"].split("-")[1]
        other_params = copy.deepcopy(params_dict)
        other_params.pop("model_type")
        model = FeatureBaggingOptRegressor(strategy=strategy, seed=iter_num,
                                           **other_params)
    else:
        raise ValueError("Invalid model type.")
    return model


def make_dgp(dgp_params_dict):
    make_X = partial(sample_normal_X, n=dgp_params_dict["n"], d=dgp_params_dict["d"],
                     corr=dgp_params_dict["rho"])
    make_y_train = partial(linear_model, sigma=0, s=dgp_params_dict["s"], beta=1,
                           heritability=dgp_params_dict["H"])
    make_y_test = partial(linear_model, sigma=0, s=dgp_params_dict["s"], beta=1)
    return make_X, make_y_train, make_y_test

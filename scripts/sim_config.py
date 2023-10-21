from sklearn.model_selection import ParameterGrid

dgp_params_grid = {
    "n": [100, 200],
    "d": [25, 50],
    "s": [5, 15],
    "rho": [0, 0.1],
    "H": [0.2, 0.5, 0.8]
}

model_params_grid1 = {
    "model_type": ["fbopt-minangle", "fbopt-ols", "fbopt-nnls"],
    "M": [10, 30],
    "alpha": [0.2, 0.5, 0.8],
    "B": [10, 30]
}

model_params_grid2 = {
    "model_type": ["ols", "ridge", "lasso"],
    "M": [None],
    "alpha": [None],
    "B": [None]
}

dgp_params_dict_list = list(ParameterGrid(dgp_params_grid))
model_params_dict_list = list(ParameterGrid(model_params_grid1)) + \
                         list(ParameterGrid(model_params_grid2))
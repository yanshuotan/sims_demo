import pickle
import warnings
from tqdm import tqdm
import multiprocessing
from functools import partial

from sklearn.metrics import mean_squared_error

from scripts.sim_util import run_one_dgp_iter
from scripts.sim_config import dgp_params_dict_list, model_params_dict_list

n_iter = 30
nprocesses = None


def main(dgp_params_dict_list, model_params_dict_list, metric=mean_squared_error,
         n_iter=30, testset=False, nprocesses=None):
    results = []
    for iter_num in tqdm(range(n_iter)):
        if nprocesses is None:
            for dgp_params_dict in tqdm(dgp_params_dict_list):
                results += run_one_dgp_iter(dgp_params_dict, model_params_dict_list,
                                            metric=metric, testset=testset, iter_num=iter_num)
        else:
            worker = partial(run_one_dgp_iter, model_params_dicts=model_params_dict_list,
                             metric=metric, testset=testset, iter_num=iter_num)
            with multiprocessing.Pool(processes=nprocesses) as pool:
                results += pool.map(worker, dgp_params_dict_list)
    return results


warnings.filterwarnings("ignore")

if __name__ == "__main__":
    results = main(dgp_params_dict_list, model_params_dict_list, n_iter=n_iter,
                   nprocesses=nprocesses)

with open(f"results/sim_results.pkl", "wb") as file:
    pickle.dump(results, file)

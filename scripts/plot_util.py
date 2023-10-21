import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

def extract_slice(results_df, n=None, d=None, s=None, rho=None, H=None):
    selector = np.array([True] * results_df.shape[0])
    if n is not None:
        selector &= results_df["n"] == n
    if d is not None:
        selector &= results_df["d"] == d
    if s is not None:
        selector &= results_df["s"] == s
    if rho is not None:
        selector &= results_df["rho"] == rho
    if H is not None:
        selector &= results_df["H"] == H
    return results_df[selector]

def make_id_col(subset_df):
    def helper(row):
        model_id = row["model_type"]
        if row["model_type"] not in ["ols", "lasso", "ridge"]:
            model_id += ", M=" + str(row["M"])
            model_id += ", alpha=" + str(row["alpha"])
            model_id += ", B=" + str(row["B"])
        return model_id
    return subset_df.apply(helper, axis=1)

def plot_one_panel(results_df, n, d, s, rho, H):
    subset_df = extract_slice(results_df, n, d, s, rho, H).copy()
    model_id = make_id_col(subset_df)
    subset_df["model_id"] = model_id
    model_id_order = subset_df.groupby("model_id").mean("score").sort_values("score").index
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    sns.boxplot(data=subset_df, x="score", y="model_id", ax=ax, order=model_id_order)
    plt.title(f"n={n}, d={d}, s={s}, rho={rho}, H={H}")
    plt.show()
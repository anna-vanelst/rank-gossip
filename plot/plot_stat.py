import os
import pickle
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
import numpy as np
from scipy.stats import trim_mean

import os
import pickle
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
import numpy as np
from src.trim import RankStatistic
from src.rank import GoRankAsync


def main(exp_names, save_path):
    ### Plot config ###
    fig, ax = plt.subplots(figsize=(3.2, 3.0))
    fontsize = 12
    plt.rcParams.update(
        {
            "font.size": 10,  # Base font size
            "axes.labelsize": 10,  # Axis label size
            "axes.titlesize": 10,  # Title size
            "xtick.labelsize": 8,  # X tick label size
            "ytick.labelsize": 8,  # Y tick label size
            "legend.fontsize": 10,  # Legend font size
        }
    )
    colors = ["C0", "C1", "C7"]
    markers = ["^", "o", "s"]
    ### End of plot config ###
    min_list = []
    for exp_name, color, marker in zip(exp_names, colors, markers):

        ### Load the configuration and results ###
        config_path = os.path.join("results", "outputs", exp_name, "config.yaml")
        config = OmegaConf.load(config_path)
        results_path = os.path.join("results", "outputs", exp_name, "results.pkl")
        with open(results_path, "rb") as f:
            results = pickle.load(f)
        mean_relative_error = results["mean_relative_error"]
        std_relative_error = results["std_relative_error"]
        horizon = config.horizon
        n = config.n
        graph_type = config.graph
        timesteps = np.arange(horizon)
        # estimate = GoRankEstimate(horizon, n, np.zeros(n))
        estimate = RankStatistic(horizon, n, np.zeros(n), GoRankAsync, np.zeros(n))
        ### End of loading ###

        (line,) = ax.plot(
            timesteps,
            mean_relative_error[estimate.name],
            marker=marker,
            color=color,
            label=graph_type,
            markevery=range(horizon // 10, horizon, horizon // 10),
            markersize=6,
        )
        line.set_rasterized(True)
        poly = ax.fill_between(
            timesteps,
            mean_relative_error[estimate.name] - std_relative_error[estimate.name],
            mean_relative_error[estimate.name] + std_relative_error[estimate.name],
            color=color,
            alpha=0.3,
        )
        poly.set_rasterized(True)
        min_list.append(min(mean_relative_error[estimate.name]))
    min_relative_error = min(min_list)
    ax.set_ylim(0.5 * min_relative_error, 0.4)
    ax.set_xlim(0, horizon)
    ticks = np.linspace(0, horizon, 5, dtype=int)
    ax.set_xticks(ticks)
    ax.set_xticklabels([0] + [f"{t/1e4:.0f}e4" for t in ticks[1:]])
    ax.set_yticks(np.round(np.linspace(min_relative_error * 0.5, 0.5, 5), 2))
    plt.title("Relative Error vs. Timesteps", fontsize=fontsize)
    plt.legend()
    plot_path = os.path.join("results", "figures", save_path)
    plt.savefig(plot_path, format="pdf", dpi=150, bbox_inches="tight")
    plt.show()

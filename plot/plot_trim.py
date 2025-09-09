import os
import pickle
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
import numpy as np
from scipy.stats import trim_mean


def main(exp_name, save_path="plot_trim_b.pdf"):
    ### Load the configuration and results ###
    config_path = os.path.join("results", "outputs", exp_name, "config.yaml")
    config = OmegaConf.load(config_path)
    results_path = os.path.join("results", "outputs", exp_name, "results.pkl")
    with open(results_path, "rb") as f:
        results = pickle.load(f)
    mean_relative_error = results["mean_relative_error"]
    std_relative_error = results["std_relative_error"]
    data = results["data"]
    horizon = config.horizon
    # n = config.data.n
    timesteps = np.arange(horizon)
    alpha = OmegaConf.select(config, "alpha") or 0
    trimmed_mean = trim_mean(data, alpha)
    print("Corrupted mean", np.mean(data))
    print("Robust mean", trimmed_mean)
    ### End of loading ###

    ### Plot config ###
    fig, ax = plt.subplots(figsize=(3.2, 3.0))
    fontsize = 12
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 10,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
        }
    )
    colors = ["C0", "C1", "C7"]
    markers = ["^", "o", "s"]
    ### End of plot config ###

    for name, color, marker in zip(mean_relative_error.keys(), colors, markers):
        last_name = name
        (line,) = ax.plot(
            timesteps,
            mean_relative_error[name],
            marker=marker,
            color=color,
            label=name,
            markevery=range(horizon // 10, horizon, horizon // 10),
            markersize=6,
        )
        line.set_rasterized(True)  # rasterize the line plot
        poly = ax.fill_between(
            timesteps,
            mean_relative_error[name] - std_relative_error[name],
            mean_relative_error[name] + std_relative_error[name],
            color=color,
            alpha=0.3,
        )
        poly.set_rasterized(True)  # rasterize the fill area

    max_list = [
        np.max(mean_relative_error[name]) for name in mean_relative_error.keys()
    ]
    min_list = [
        np.min(mean_relative_error[name]) for name in mean_relative_error.keys()
    ]
    original_error = np.mean(data) - trimmed_mean
    ax.set_ylim(0.5 * min(min_list), 0.4 * max(max_list))
    ax.axhline(
        original_error,
        color="black",
        linestyle="--",
        label="Error of the corrupted mean",
    )
    ax.set_xlim(0, horizon)
    ticks = np.linspace(0, horizon, 5, dtype=int)
    ax.set_xticks(ticks)
    ax.set_xticklabels([0] + [f"{t/1e4:.0f}e4" for t in ticks[1:]])
    plt.title("Absolute Error vs. Timesteps", fontsize=fontsize)
    plt.legend()
    plot_path = os.path.join("results", "figures", save_path)
    plt.savefig(plot_path, format="pdf", dpi=150, bbox_inches="tight")
    plt.show()

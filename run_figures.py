from plot.plot_rank import main as plot_rank
from plot.plot_trim import main as plot_trim
from plot.plot_stat import main as plot_stat
import argparse
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot_name", type=str, required=True, help="Name of plot")
    args = parser.parse_args()
    plot_name = args.plot_name
    path = os.path.join("results", "figures")
    os.makedirs(path, exist_ok=True)
    if plot_name == "exp_trim":
        plot_trim("exp_trim", save_path="exp_trim.pdf")
    elif plot_name == "exp_async":
        plot_rank("exp_async", save_path="exp_async.pdf")
    elif plot_name == "exp_stat":
        exp_names = ["exp_stat", "exp_stat_ws", "exp_stat_geo"]
        plot_stat(exp_names, save_path="exp_stat.pdf")


if __name__ == "__main__":
    main()

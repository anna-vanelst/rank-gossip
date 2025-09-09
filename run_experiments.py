import os
import random
import argparse
import pickle
import time
import numpy as np
import yaml
from omegaconf import OmegaConf

from src.task import TaskRank, TaskTrim, TaskStat
from src.utils import compute_weighted_connectivity

TASKS = {
    "ranking": TaskRank,
    "trim": TaskTrim,
    "stat": TaskStat,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, default="", help="Experiment name")
    args = parser.parse_args()
    exp_name = args.exp_name

    config = OmegaConf.load(os.path.join("configs", f"{exp_name}.yaml"))
    config.folder = exp_name
    print(f"Running experiment: {exp_name}")

    np.random.seed(config.seed)

    if config.task not in TASKS:
        raise ValueError(f"Unsupported task type: {config.task}")
    task = TASKS[config.task](config)
    n_trials = config.n_trials
    horizon = config.horizon
    task.init()
    method_names = [method.name for method in task.methods]
    all_errors = {name: [] for name in method_names}
    print("Names: ", method_names)
    sampling = OmegaConf.select(config, "sampling") or "uniform"
    n = len(task.graph.nodes())
    degrees = [task.graph.degree[i] for i in task.graph.nodes()]
    weights = [(1 / degrees[i] + 1 / degrees[j]) / n for i, j in task.edges]
    if sampling == "async":
        connectivity = compute_weighted_connectivity(task.edges, weights)
        print(f"Connectivity of weighted graph: {connectivity:.2e}")

    for trial in range(n_trials):
        if trial % 10 == 0:
            print(f"Trial {trial + 1}/{n_trials}")
        task.init()
        t = 1
        while t < horizon:
            if sampling == "uniform":
                i, j = random.choice(task.edges)
            else:
                i, j = random.choices(task.edges, weights=weights, k=1)[0]
            task.update(t, i, j)
            t += 1
        if config.verbose:
            for method in task.methods:
                method.print_info(-1)
        trial_errors = task.eval()
        for name, err in trial_errors.items():
            all_errors[name].append(err)

    mean_errors = {name: np.mean(errs, axis=0) for name, errs in all_errors.items()}
    std_errors = {name: np.std(errs, axis=0) for name, errs in all_errors.items()}
    if config.verbose:
        for method in method_names:
            print(f"{method} has mean error:", mean_errors[method][-1])

    folder_path = os.path.join("results", "outputs", config.folder)
    os.makedirs(folder_path, exist_ok=True)

    results = {
        "names": method_names,
        "config": config,
        "data": task.data,
        "mean_relative_error": mean_errors,
        "std_relative_error": std_errors,
    }

    results_path = os.path.join(folder_path, "results.pkl")
    with open(results_path, "wb") as f:
        pickle.dump(results, f)

    config_path = os.path.join(folder_path, "config.yaml")
    with open(config_path, "w") as f:
        yaml.dump(OmegaConf.to_container(config), f, sort_keys=False)

    print("Experiment completed and results saved.")


if __name__ == "__main__":
    print("Running experiments...")
    start = time.perf_counter()
    main()
    end = time.perf_counter()
    print(f"Execution time: {(end - start) / 60:.2f} minutes")

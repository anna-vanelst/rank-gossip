# Asynchronous Gossip Algorithms for Rank-Based Statistical Methods

This repository contains the official implementation of the experiments and figures in our paper, designed to make our research reproducible.
You can find the preprint [here](https://arxiv.org/pdf/2509.07543).

## Citation

If you find this work useful, please consider citing our paper:
```bibtex
@article{van2025asynchronous,
  title={Asynchronous Gossip Algorithms for Rank-Based Statistical Methods},
  author={Van Elst, Anna and Colin, Igor and Cl{\'e}men{\c{c}}on, Stephan},
  journal={To appear in International Conference on Federated Learning Technologies and Applications (FLTA)},
  year={2025}
}
```

## Repository Structure

```bash
├── configs/               # Configs used to run experiments
├── plot/                  # Utils to generate each figure
├── src/                   # Core source code (gossip algorithms)
├── requirements.txt       # Python dependencies
├── run_experiments.py     # Script to run experiments
└── run_figures.py         # Script to generate paper figures
```

## Getting Started

Install requirements using pip:
```bash
pip install -r requirements.txt
```
To run an experiment called "exp", you can simply use the following 
```bash
python run_experiments.py --exp_name "exp"
```

To generate a figure called "plot" from the paper, you can use the following
```bash
python run_figures.py --plot_name "plot"
```

## Reproducing the Figures

* **Figure (a)**

  * Plot name: `exp_async`
  * Required experiment: `exp_async`

* **Figure (b)**

  * Plot name: `exp_stat`
  * Required experiments: `exp_stat`, `exp_stat_ws`, `exp_stat_geo`

* **Figure (c)**

  * Plot name: `exp_trim`
  * Required experiment: `exp_trim`

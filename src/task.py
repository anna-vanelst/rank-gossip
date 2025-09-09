import numpy as np
from omegaconf import OmegaConf
from abc import ABC
from src.data import load_data_arange, load_data_groups
from src.rank import GoRank, GoRankAsync
from src.trim import GoTrim, AdaptiveGoTrim, RankStatistic

DATA_HANDLERS = {"arange": load_data_arange, "groups": load_data_groups}


class BaseTask(ABC):
    """
    Base class for tasks.
    """

    def __init__(self, config):
        self.config = config
        self.name = self.__class__.__name__.replace("Task", "")

        if config.data not in DATA_HANDLERS:
            raise ValueError(f"Unsupported data type: {config.data}")
        self.data, self.graph, _ = DATA_HANDLERS[config.data](config)
        self.edges = list(self.graph.edges)
        self.methods = []
        self.register = None

    def init(self):
        if self.config.shuffle:
            self.indices = np.random.permutation(self.config.n)
            self.data = self.data[self.indices]

    def update(self, t, i, j):
        for method in self.methods:
            method.update(t, i, j)

    def eval(self):
        horizon = self.config.horizon
        errors = {}
        for method in self.methods:
            true_value = method.true_value()
            rel_errors = np.zeros(horizon)
            for t in range(horizon):
                rel_errors[t] = np.mean(np.abs(method.estimates[t] - true_value))
            errors[method.name] = rel_errors
        return errors


RANKING = {
    "GoRank": GoRank,
    "GoRankAsync": GoRankAsync,
}


class TaskRank(BaseTask):
    def __init__(self, config):
        super().__init__(config)
        self.register = RANKING

    def init(self):
        super().init()
        config = self.config
        self.methods = [
            self.register[method](config.horizon, config.n, self.data)
            for method in self.config.methods
        ]

    def eval(self):
        errors = super().eval()
        for method in self.methods:
            errors[method.name] /= self.config.n
        return errors


TRIM = {
    "GoTrim": GoTrim,
    "AdaptiveGoTrim": AdaptiveGoTrim,
}


class TaskTrim(BaseTask):
    def __init__(self, config):
        super().__init__(config)
        self.register = TRIM

    def init(self):
        n = self.config.n
        horizon = self.config.horizon
        alpha = OmegaConf.select(self.config, "alpha", default=0.0)
        rank_class = RANKING[self.config.rank]
        self.methods = [
            self.register[method](horizon, n, self.data, alpha, rank_class)
            for method in self.config.methods
        ]

    def update(self, t, i, j):
        for method in self.methods:
            method.rank.update(t, i, j)
            method.update(t, i, j)


STAT = {
    "RankStatistic": RankStatistic,
}


class TaskStat(BaseTask):
    def __init__(self, config):
        self.config = config
        self.name = self.__class__.__name__.replace("Task", "")

        if config.data not in DATA_HANDLERS:
            raise ValueError(f"Unsupported data type: {config.data}")
        self.data, self.graph, self.mask = DATA_HANDLERS[config.data](config)
        self.edges = list(self.graph.edges)
        self.methods = []
        self.register = STAT

    def init(self):
        n = self.config.n
        horizon = self.config.horizon
        rank_class = RANKING[self.config.rank]
        self.methods = [
            self.register[method](horizon, n, self.data, rank_class, self.mask)
            for method in self.config.methods
        ]

    def update(self, t, i, j):
        for method in self.methods:
            method.rank.update(t, i, j)
            method.update(t, i, j)

    def eval(self):
        errors = super().eval()
        for method in self.methods:
            errors[method.name] /= method.true_value().mean()
        return errors

import numpy as np
from abc import ABC, abstractmethod
from scipy.stats import rankdata


class RankMethod(ABC):
    """
    Base class for ranking methods.
    """

    def __init__(self, horizon: int, n: int, data: np.ndarray):
        self.horizon = horizon
        self.n = n
        self.data = data.copy()
        self.aux_x = self.data.copy()
        self.estimates = np.zeros((horizon, n))
        self.aux_rank = np.zeros((horizon, n))
        self.name = self.__class__.__name__

    def update(self, t: int, i: int, j: int):
        self._copy_previous_state(t)
        self._rank(t, i, j)

    def _copy_previous_state(self, t: int):
        self.estimates[t] = self.estimates[t - 1].copy()
        self.aux_rank[t] = self.aux_rank[t - 1].copy()

    @abstractmethod
    def _rank(self, t: int, i: int, j: int):
        pass

    def true_value(self):
        return rankdata(self.data, method="average")

    def print_info(self, t):
        print(f"{self.name} estimates at final time step:, ", self.estimates[t])
        if t == -1:
            print(f"True value: {self.true_value()}")


class GoRank(RankMethod):
    """Implements the GoRank algorithm."""

    def __init__(self, horizon, n, data, ties=False):
        self.ties = ties
        super().__init__(horizon, n, data)

    def _rank(self, t, i, j):
        self.aux_x[[i, j]] = self.aux_x[[j, i]]
        self.aux_rank[t] = (t - 1) * self.aux_rank[t - 1]
        self.aux_rank[t] += self.data > self.aux_x
        if self.ties:
            self.aux_rank[t] += (self.data == self.aux_x) / 2
        self.aux_rank[t] /= t
        self.estimates[t] = (1 / 2 if self.ties else 1) + self.n * self.aux_rank[t]


class GoRankAsync(RankMethod):
    """Implements the asynchronous GoRank algorithm."""

    def __init__(self, horizon, n, data, ties=False):
        self.ties = ties
        super().__init__(horizon, n, data)
        self.count = np.ones(self.n)

    def _rank(self, t, i, j):
        self.aux_x[[i, j]] = self.aux_x[[j, i]]
        self.count[i] += 1
        self.count[j] += 1
        for node in [i, j]:
            count = self.count[node]
            r = self.aux_rank[t - 1][node]
            self.aux_rank[t][node] = (count - 1) * r
            self.aux_rank[t][node] += self.data[node] > self.aux_x[node]
            if self.ties:
                self.aux_rank[t][node] += (self.data[node] == self.aux_x[node]) / 2
            self.aux_rank[t][node] /= count
        self.estimates[t] = (1 / 2 if self.ties else 1) + self.n * self.aux_rank[t]

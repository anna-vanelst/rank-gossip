import numpy as np
from abc import ABC, abstractmethod
from src.utils import wn, Transform
from scipy.stats import trim_mean


class BaseEstimator(ABC):
    """
    Base class for estimators for rank-based methods.
    """

    def __init__(self, horizon, n, data, rank_class):
        self.horizon = horizon
        self.n = n
        self.data = data.copy()
        self.rank = rank_class(horizon, n, data)
        self.estimates = np.zeros((horizon, n))
        self.weights = np.zeros((horizon, n))
        self.name = self.__class__.__name__ + " + " + self.rank.name
        self.alpha = 0.2

    @abstractmethod
    def update_mean(self, t, i, j):
        pass

    def print_info(self, t):
        print(f"{self.name} estimates at final time step:, ", self.estimates[t])
        if t == -1:
            print(f"True value: {self.true_value()}")

    def _copy_previous_state(self, t):
        self.estimates[t] = self.estimates[t - 1].copy()
        self.weights[t] = self.weights[t - 1].copy()

    def update(self, t, i, j):
        self._copy_previous_state(t)
        self.update_mean(t, i, j)

    def true_value(self):
        value = trim_mean(self.data, self.alpha)
        return np.full(self.n, value)


class GoTrim(BaseEstimator):
    """
    Implements the GoTrim algorithm.
    """

    def __init__(self, horizon, n, data, alpha, rank_class):
        super().__init__(horizon, n, data, rank_class)
        self.alpha = alpha

    def update_mean(self, t, i, j):
        for node in [i, j]:
            self.weights[t][node] = wn(self.n, self.rank.estimates[t][node], self.alpha)
            delta = self.weights[t][node] - self.weights[t - 1][node]
            self.estimates[t][node] += delta * self.data[node]

        avg = (self.estimates[t][i] + self.estimates[t][j]) / 2
        self.estimates[t][i], self.estimates[t][j] = avg, avg


class RankStatistic(BaseEstimator):
    """
    Implements the extended GoTrim algorithm for rank statistics.
    """

    def __init__(self, horizon, n, data, rank_class, mask):
        super().__init__(horizon, n, data, rank_class)
        self.mask = mask
        self.transform = Transform(n, type="identity")
        self.aux = np.zeros((horizon, n))

    def _copy_previous_state(self, t):
        super()._copy_previous_state(t)
        self.aux[t] = self.aux[t - 1].copy()

    def update_mean(self, t, i, j):
        for node in [i, j]:
            r = self.rank.estimates[t][node]
            r_prev = self.rank.estimates[t - 1][node]
            delta = self.transform.apply(r) - self.transform.apply(r_prev)
            self.aux[t][node] += delta * self.mask[node]

        avg = (self.aux[t][i] + self.aux[t][j]) / 2
        self.aux[t][i], self.aux[t][j] = avg, avg
        self.estimates[t] = self.n * self.aux[t]

    def true_value(self):
        ranks = self.rank.true_value()
        value = self.n * np.mean(ranks * self.mask)
        return np.full(self.n, value)


class AdaptiveGoTrim(BaseEstimator):
    """
    Implements the Adaptive GoTrim algorithm."""

    def __init__(self, horizon, n, data, alpha, rank_class):
        super().__init__(horizon, n, data, rank_class)
        self.alpha = alpha
        self.top = np.zeros((horizon, n))
        self.bottom = np.zeros((horizon, n))

    def _copy_previous_state(self, t):
        super()._copy_previous_state(t)
        self.top[t] = self.top[t - 1].copy()
        self.bottom[t] = self.bottom[t - 1].copy()

    def update_mean(self, t, i, j):
        for node in [i, j]:
            self.weights[t][node] = wn(self.n, self.rank.estimates[t][node], self.alpha)
            delta = self.weights[t][node] - self.weights[t - 1][node]
            self.top[t][node] += delta * self.data[node]
            self.bottom[t][node] += delta

        avg = (self.top[t][i] + self.top[t][j]) / 2
        self.top[t][i], self.top[t][j] = avg, avg
        avg = (self.bottom[t][i] + self.bottom[t][j]) / 2
        self.bottom[t][i], self.bottom[t][j] = avg, avg

        for node in [i, j]:
            denom = max(float(self.bottom[t][node]), 1.0)
            self.estimates[t][node] = self.top[t][node] / denom

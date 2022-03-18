from __future__ import annotations
import abc


import numpy as np


class Model(abc.ABC):
    @abc.abstractmethod
    def fit(self, y: np.ndarray, X: np.ndarray, **kwargs) -> Model:
        pass

    @abc.abstractmethod
    def forecast(self, y: np.ndarray, X: np.ndarray) -> np.ndarray:
        pass

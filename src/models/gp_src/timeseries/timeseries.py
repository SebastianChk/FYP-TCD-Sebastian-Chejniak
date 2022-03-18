from typing import Tuple

import gpflow
import numpy as np
import pandas as pd
from gpflow import default_float


class Timeseries:
    def __init__(self, df: pd.DataFrame, covariates=None):
        if covariates is None:
            covariates = []
        self._covariates = covariates
        self._date = "dt"
        self._is_trainset = "is_trainset"
        self._responses = np.setdiff1d(
            list(df.columns),
            self._covariates + [self._date] + [self._is_trainset]
        )
        self._df = df.sort_values("dt")
        self._start_date, self._end_date, self._scale_date = None, None, None
        self._Y = df[self._responses].values
        self._X = self._encode_covariates(df)

    def data(self) -> Tuple[dict, dict]:
        train_data_dxs = self._df[self._is_trainset] == 1
        train_data = {"Y": self._Y[train_data_dxs], "X": self._X[train_data_dxs]}
        test_data = {"Y": self._Y[~train_data_dxs], "X": self._X[~train_data_dxs]}
        return train_data, test_data

    def _encode_covariates(self, df: pd.DataFrame):
        if df["dt"].dtype != "datetime64[ns]":
            raise ValueError("column 'dt' needs to have dtype 'datetime[ns]'")
        df = df.copy().sort_values("dt")
        self._init_dates(df)

        df.loc[:, "t"] = (df["dt"] - self._start_date) / self._scale_date
        X = np.asarray(df.t.values, dtype=np.float32)[:, np.newaxis]

        for reg in self._covariates[1:]:
            xo = df[reg]
            if xo.dtype in ["float", "int"]:
                xo = np.asarray(xo.values, dtype=default_float())[:, np.newaxis]
            elif xo.dtype == "category":
                xo = np.asarray(pd.get_dummies(xo).values, dtype=np.float64)
            else:
                raise TypeError(
                    f"dtype '{xo.dtype}' of column '{reg}' is not supported. "
                    f"convert to numerical/categorical instead"
                )
            X = np.hstack([X, xo])

        return X.astype(dtype=gpflow.default_float())

    def get_future_steps(self, dt: pd.Series):
        return ((dt - self._end_date) / self._scale_date).values

    def _init_dates(self, df: pd.DataFrame):
        if df["dt"].dtype != "datetime64[ns]":
            raise ValueError("column 'dt' needs to have dtype 'datetime[ns]'")
        dixs = df[self._is_trainset] == 1
        self._start_date = df[dixs][self._date].min()
        self._end_date = df[dixs][self._date].max()
        self._scale_date = np.timedelta64(365, "D")


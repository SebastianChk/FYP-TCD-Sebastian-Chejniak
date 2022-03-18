import logging
from typing import Tuple, Type, Any

import gpflow
import tensorflow as tf
import numpy as np
from gluonts.dataset.common import ListDataset
from gluonts.model import deepar
from gluonts.mx.distribution import NegativeBinomialOutput
from gluonts.mx.trainer import Trainer
from gpflow.ci_utils import ci_niter
from gpflow.kernels import Kernel, Matern12
from gpflow.likelihoods import Likelihood
from gpflow.models import GPModel

from timeseries import Model

logger = logging.getLogger(__name__)


class DeepAR(Model):
    def __init__(self, likelihood: Type[Likelihood], **kwargs):
        self._is_fit = False
        self._model: Any
        self._likelihood = likelihood

    def fit(self, y: np.ndarray, X: np.ndarray, prediction_length, **kwargs):
        dataset = self._reshape(y, X)
        logger.info("Fitting DeepAR model")
        logger.info(
            "Using observation model '%s'",
            str(self._likelihood.__name__),
        )

        self._is_fit = False
        self._model = deepar.DeepAREstimator(
            prediction_length=prediction_length,
            context_length=prediction_length,
            freq='D',
            distr_output=NegativeBinomialOutput(),
            trainer=Trainer(epochs=50)
        )
        self._predictor = self._model.train(dataset)
        self._is_fit = True
        logger.info("Fitting single models finished successfully")
        return self

    def forecast(self, y: np.ndarray, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if not self._is_fit:
            raise ValueError("Fit the model before making predictions")
        logger.info("Forecasting DeepAR model")
        logger.info(
            "Using observation model '%s'",
            str(self._likelihood.__name__),
        )

        pred = self._predictor.predict(self._dataset, num_samples=100)
        pred = next(pred)
        logger.info("Forecasting single models finished successfully")
        return pred.mean, np.var(pred.samples, axis=1)

    def _reshape(self, y, x):
        time_series_dicts = []
        for i in range(y.shape[1]):
            time_series_dicts.append(
                {"target": y[:, i], "start": "2011-01-29"})
        dataset = ListDataset(time_series_dicts, freq="D")
        self._dataset = dataset
        return dataset

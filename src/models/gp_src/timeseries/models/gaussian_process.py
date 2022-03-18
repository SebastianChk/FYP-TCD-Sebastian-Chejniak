import logging
from typing import Tuple, Type

import gpflow
import tensorflow as tf
import numpy as np
from gpflow.ci_utils import ci_niter
from gpflow.kernels import Kernel, Matern12
from gpflow.likelihoods import Likelihood
from gpflow.models import GPModel
import scipy.stats

from timeseries import Model

logger = logging.getLogger(__name__)


class GaussianProcess(Model):
    def __init__(self, kernel: Type[Kernel], likelihood: Type[Likelihood], batch=True, use_priors=True):
        self._is_fit = False
        self._model: GPModel
        self._kernel = kernel
        self._likelihood = likelihood
        self._batch = batch
        self._use_priors=use_priors

    def fit(self, y: np.ndarray, X: np.ndarray, iterations=500, **kwargs):
        y, X = self._reshape(y, X)
        logger.info("Fitting GP model")
        logger.info(
            "Using observation model '%s' and kernel '%s'",
            str(self._kernel.__name__),
            str(self._likelihood.__name__),
        )

        self._is_fit = False
        self._set_model(y, X)
        if  self._batch==False:
            self._fit_full(iterations)
        elif self._batch==True:
            self._fit_batch(y, X)

        self._is_fit = True
        logger.info("Fitting models finished successfully")
        return self

    def forecast(self, y: np.ndarray, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        _, X = self._reshape(y, X)
        if not self._is_fit:
            raise ValueError("Fit the model before making predictions")
        logger.info("Forecasting GP model")
        logger.info(
            "Using observation model '%s' and kernel '%s'",
            str(self._kernel.__name__),
            str(self._likelihood.__name__),
        )
        
        fmean, fvar = self._model.predict_f(X)
        ymean = self._likelihood().invlink(fmean+0.5*fvar).numpy()
        
        coef = ( self._model.likelihood.psi.numpy() + 1 ) / self._model.likelihood.psi.numpy()
        _invlink = self._likelihood().invlink
        yvar = coef * _invlink( 2*(fmean+fvar) ).numpy() + _invlink( fmean+fvar/2 ).numpy() - _invlink( 2*fmean+fvar ).numpy()
        
        # Monte carlo estimation of the pmf of counts:
        samples = True
        if samples:
            fsamples = self._model.predict_f_samples(X, num_samples=5_000)
            psi = self._model.likelihood.psi.numpy()
            n_ys = 500
            y_s = np.arange(n_ys).reshape((n_ys, 1, 1, 1))
            mu = np.exp(fsamples[np.newaxis, :, :, :])
            p = psi / (mu + psi)
            nb_pmfs = scipy.stats.nbinom.pmf(k=y_s, n=psi, p=p)
            ypmfs = np.mean(nb_pmfs, axis=1)
        # End pmf estimation
        
        ymean = ymean[:, 0].reshape(y.shape, order="F")
        yvar = yvar[:, 0].reshape(y.shape, order="F")
        logger.info("Forecasting models finished successfully")
        if samples:
            return ymean, yvar, ypmfs
        
        return ymean, yvar

    def _set_model(self, y: np.ndarray, X: np.ndarray, M=100):
        k = self._kernel(use_priors=self._use_priors)
        lik = self._likelihood()
        if y.shape[1] != 1:
            q = len(np.unique(y[:, 1]))
            logger.info("Setting coregional model with %i outputs", q)
            rank = int(np.sqrt(q)) + 1
            coregional = gpflow.kernels.Coregion(
                output_dim=q,
                rank=q,
                active_dims=[1]
            )
            coregional.W = np.random.randn(q * rank).reshape((q, rank))
            k = self._kernel(active_dims=[0])
            k = k * coregional
            lik = gpflow.likelihoods.SwitchedLikelihood(
                [self._likelihood() for _ in range(q)]
            )
        Z = X[np.random.choice(X.shape[0], 100), :].copy()
        if  self._batch==False:
            self._model = gpflow.models.VGP((X,y),
                kernel=k, likelihood=lik
            )
        else:
            self._model = gpflow.models.SVGP(
                kernel=k, likelihood=lik,
                inducing_variable=Z,
                num_data=y.shape[0]
            )

    def _fit_full(self, iterations=1000):
        opt = gpflow.optimizers.Scipy()
        res=opt.minimize(self._model.training_loss, self._model.trainable_variables)
        self.res=res
            
    def _fit_batch(self, y: np.ndarray, X: np.ndarray, iterations=1000, n_batch=128):
        train_dataset = tf.data.Dataset.from_tensor_slices(
            (X, y)).repeat().shuffle(X.shape[0])

        iterations = ci_niter(iterations)
        gpflow.set_trainable(self._model.inducing_variable, True)
        train_iter = iter(train_dataset.batch(n_batch))
        self.training_loss = self._model.training_loss_closure(train_iter, compile=True)
        optimizer = tf.optimizers.Adam()

        @tf.function
        def optimization_step():
            optimizer.minimize(self.training_loss, self._model.trainable_variables)

        for step in range(iterations):
            optimization_step()

    def _reshape(self, y, x):
        q = y.shape[1]
        if q > 1:
            cats = np.repeat(np.arange(q), y.shape[0]).reshape(-1, 1)
            xs = np.tile(x, y.shape[1]).reshape(-1, 1, order="F")
            ys = y.reshape(-1, 1, order="F")
            ys, xs = np.hstack([ys, cats]), np.hstack([xs, cats])
            return ys, xs
        return y, x

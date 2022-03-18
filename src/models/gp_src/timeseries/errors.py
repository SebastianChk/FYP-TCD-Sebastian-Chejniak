import numpy as np
import tensorflow_probability as tfp

tfd = tfp.distributions


def mae(y_test: np.ndarray, y_test_hat: np.ndarray):
    r"""
    Mean absolute error

    .. math::

        \frac{1}{n} \sum_{i=1}^n |y_i - \hat{y}_i|

    Parameters
    ----------
    y_test: np.ndarray
        array of true response values of the test set
    y_test_hat: np.ndarray
        array of predicted response values of the test set

    Returns
    -------
    float
        returns the mean absolute error
    """

    return np.mean(np.abs(y_test - y_test_hat))


def mase(
    y_train: np.ndarray, y_test: np.ndarray, y_test_hat: np.ndarray, lag=1
):
    r"""
    Mean absolute scaledd error

    Parameters
    ----------
    y_train: np.ndarray
        array of true response values of the train set
    y_test: np.ndarray
        array of true response values of the test set
    y_test_hat: np.ndarray
        array of predicted response values of the test set

    Returns
    -------
    float
        returns the mean absolute scaled error
    """

    mae_f = mae(y_test, y_test_hat)
    mae_naive = mae(y_train[lag:], y_train[: (len(y_train) - lag)])
    return mae_f / mae_naive


def compute_errors(y_train, y_test, y_test_hat):
    if y_train.shape[1] > 1:
        y_train = y_train.reshape(-1, 1, order="F")
        y_test = y_test.reshape(-1, 1, order="F")
        y_test_hat = y_test_hat.reshape(-1, 1, order="F")

    lerrors = {
        "mae": [],
        "mase": []
    }
    for ran in [slice(5), slice(10), slice(None)]:
        lerrors["mae"].append(mae(y_test[ran], y_test_hat[ran]))
        lerrors["mase"].append(mase(y_train, y_test[ran], y_test_hat[ran]))
    lerrors["th"] = ["05", "10", str(len(y_test_hat))]
    return lerrors

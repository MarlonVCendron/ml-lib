from typing import Self
import logging
import numpy as np
import numpy.typing as npt

logger = logging.getLogger(__name__)

class LinearRegression:
  _lr: float
  _n: int
  _n_coefs: int
  _coefs: npt.NDArray[np.float64]
  _X: npt.NDArray[np.float64]
  _y: npt.NDArray[np.float64]
  _X_base: npt.NDArray[np.float64]

  def __init__(self, lr: float = 0.001):
    self._lr = lr
    self._n = 0
    self._n_coefs = 1
    self._coefs = np.zeros(shape=(self._n_coefs))
    self._X = np.zeros(shape=(self._n_coefs - 1, self._n))
    self._y = np.zeros(shape=(self._n))
    self._X_base = np.zeros(shape=(self._n_coefs, self._n))

  def fit(self, X: npt.NDArray[np.float64], y: npt.NDArray[np.float64]) -> Self:
    assert X.shape[0] == y.shape[0], 'Shape mismatch between X and y'

    self._X = X
    self._y = y
    self._n = X.shape[0]
    self._X_base = np.c_[np.ones(self._n), self._X]  # Add ones for coef_0

    self._initialize_coefs(X)

    self._gradient_descent(X, y)

    return self

  def get_coefs(self) -> npt.NDArray[np.float64]:
    return self._coefs

  def _initialize_coefs(self, X: npt.NDArray[np.float64]) -> None:
    self._n_coefs = X.shape[1] + 1
    self._coefs = np.zeros(shape=(self._n_coefs))

  def _cost_function(self):
    mses = 0
    return 1/(2 * self._n) * mses

  def _gradient_descent(self, X: npt.NDArray[np.float64], y: npt.NDArray[np.float64]) -> None:
    converged = False

    while not converged:
      next_coefs = np.copy(self._coefs)

      for i in range(self._n_coefs):
        next_coef = self._compute_next_coef(index=i)
        next_coefs[i] = next_coef
      
      if np.any(np.isinf(next_coefs)):
        logger.error('Coefficients diverged')
        break

      if np.any(np.isnan(next_coefs)):
        logger.error('Failed to converge coefficients')
        break

      converged = abs(np.sum(next_coefs - self._coefs)) < 0.001

      self._coefs = next_coefs

    return

  def _compute_next_coef(self, index: int) -> float:
    return self._coefs[index] - self._lr * self._cost_function_partial_derivative(index)

  def _cost_function_partial_derivative(self, index: int) -> float:
    error = self._hypothesis() - self._y
    xs = self._X_base[:, index]
    return (1/self._n) * np.sum(xs * error)

  def _hypothesis(self) -> npt.NDArray[np.float64]:
    return self._X_base.dot(self._coefs)


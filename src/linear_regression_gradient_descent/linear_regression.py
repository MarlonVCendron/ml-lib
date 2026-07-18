from typing import Self
import logging
import numpy as np
import numpy.typing as npt

logger = logging.getLogger(__name__)


class LinearRegression:
  _classification: float
  _lr: float
  _n: int
  _n_params: int
  _params: npt.NDArray[np.float64]
  _X: npt.NDArray[np.float64]
  _y: npt.NDArray[np.float64]
  _X_base: npt.NDArray[np.float64]

  def __init__(self, classification: bool = False, lr: float = 0.001):
    self._classification = classification
    self._lr = lr
    self._n = 0
    self._n_params = 1
    self._params = np.zeros(shape=(self._n_params))
    self._X = np.zeros(shape=(self._n_params - 1, self._n))
    self._y = np.zeros(shape=(self._n))
    self._X_base = np.zeros(shape=(self._n_params, self._n))

  def fit(self, X: npt.NDArray[np.float64], y: npt.NDArray[np.float64]) -> Self:
    assert X.shape[0] == y.shape[0], 'Shape mismatch between X and y'

    self._X = X
    self._y = y
    self._n = X.shape[0]
    self._X_base = np.c_[np.ones(self._n), self._X]  # Add ones for param_0

    self._initialize_params()

    self._gradient_descent()

    return self

  def summary(self) -> str:
    total_cost = np.sum(self._cost_function(self._X_base, self._y))
    res = self.predict(self._X)
    
    if self._classification:
      accuracy = (len(res) - np.count_nonzero(np.round(res - self._y))) / len(res)
    else:
      accuracy = (len(res) - np.count_nonzero(res - self._y)) / len(res)
      
    return (
        f'Accuracy: {accuracy:.2f}\n'
        f'Total cost: {total_cost:.2f}'
    )

  def predict(self, X: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    X_base = np.c_[np.ones(X.shape[0]), X]
    return self._hypothesis(X_base)

  def get_params(self) -> npt.NDArray[np.float64]:
    return self._params

  def _initialize_params(self) -> None:
    self._n_params = self._X.shape[1] + 1
    self._params = np.zeros(shape=(self._n_params))

  def _gradient_descent(self) -> None:
    converged = False

    while not converged:
      next_params = np.copy(self._params)

      for i in range(self._n_params):
        next_param = self._compute_next_param(index=i)
        next_params[i] = next_param

      if np.any(np.isinf(next_params)):
        logger.error('paramficients diverged')
        break

      if np.any(np.isnan(next_params)):
        logger.error('Failed to converge paramficients')
        break

      converged = abs(np.sum(next_params - self._params)) < 0.0001

      self._params = next_params

    return

  def _compute_next_param(self, index: int) -> float:
    return self._params[index] - self._lr * self._cost_function_partial_derivative(index)

  def _cost_function_partial_derivative(self, index: int) -> float:
    error = self._hypothesis(self._X_base) - self._y
    xs = self._X_base[:, index]
    return (1/self._n) * np.sum(xs * error)

  def _cost_function(self, X: npt.NDArray[np.float64], y: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    mses = np.square((self._hypothesis(X) - y))
    return 1/(2 * self._n) * np.sum(mses)

  def _hypothesis(self, X: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    return X.dot(self._params)

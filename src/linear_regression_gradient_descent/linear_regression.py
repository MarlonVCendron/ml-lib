from typing import Self
import numpy as np
import numpy.typing as npt


class LinearRegression:
  _learning_rate: float
  _n: int
  _n_coefs: int
  _coefs: npt.NDArray[np.float64]
  _X: npt.NDArray[np.float64]
  _y: npt.NDArray[np.float64]
  _X_base: npt.NDArray[np.float64]

  def __init__(self, learning_rate: float = 0.1):
    self._learning_rate = learning_rate
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
    self._X_base = np.c_[np.ones(self._n), self._X] # Add ones for coef_0

    self._initialize_coefs(X)

    self._gradient_descent(X, y)

    return self

  def get_coefs(self) -> npt.NDArray[np.float64]:
    return self._coefs

  def _initialize_coefs(self, X: npt.NDArray[np.float64]) -> None:
    self._n_coefs = X.shape[1] + 1
    self._coefs = np.zeros(self._n_coefs)

  def _cost_function(self):
    mses = 0
    return 1/(2 * self._n) * mses

  def _gradient_descent(self, X: npt.NDArray[np.float64], y: npt.NDArray[np.float64]) -> None:
    converged = False

    while not converged:
      next_coefs = np.copy(self._n_coefs)

      for i, coef in enumerate(self._coefs):
        next_coef = self._compute_next_coef(index=i)
        next_coefs[i] = next_coef

      converged = np.all(next_coefs == self._coefs)

      self._coefs = next_coefs

      print(next_coefs, converged)

    return

  def _compute_next_coef(self, index: int) -> float:
    return self._coefs[index] - self._learning_rate * self._cost_function_partial_derivative(index)

  def _cost_function_partial_derivative(self, index: int) -> float:
    return (1/self._n) * np.sum(self._X_base.dot(self._hypothesis() - self._y))

  def _hypothesis(self) -> npt.NDArray[np.float64]:
    return self._X_base.dot(self._coefs)

  def _has_converged(self) -> bool:
    return True

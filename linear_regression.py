from typing import Self
import numpy as np
import numpy.typing as npt


class LinearRegression:
  _learning_rate: float
  _n: int
  _n_coefs: int
  _coefs: npt.NDArray[np.float64]

  def __init__(self, learning_rate: float = 0.1):
    self._learning_rate = learning_rate
    self._n = 0
    self._n_coefs = 1
    self._coefs = np.zeros(self._n_coefs)

  def fit(self, X: npt.NDArray[np.float64], y: npt.NDArray[np.float64]) -> Self:
    assert X.shape[0] == y.shape[0], 'Shape mismatch between X and y'

    self._initialize_coefs(X)
    self._initialize_n(X)

    self._gradient_descent(X, y)

    return self

  def get_coefs(self) -> npt.NDArray[np.float64]:
    return self._coefs

  def _initialize_n(self, X: npt.NDArray[np.float64]) -> None:
    self._n = X.shape[0]

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

      converged = self._has_converged()

    return

  def _compute_next_coef(self, index: int) -> float:
    return self._coefs[index] - self._learning_rate * self._cost_function_partial_derivative()

  def _cost_function_partial_derivative(self) -> float:
    return 0

  def _has_converged(self) -> bool:
    return True

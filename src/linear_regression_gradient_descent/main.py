import pandas as pd
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
from pathlib import Path

from .linear_regression import LinearRegression

DATA_PATH = Path.cwd() / 'data' / 'diabetes.csv'
Y_COL = 'Outcome'


def load_data() -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
  df = pd.read_csv(DATA_PATH)

  X = df.loc[:, df.columns != Y_COL].to_numpy()
  y = df[Y_COL].to_numpy()

  return X, y


def main():
  X, y = load_data()

  model = LinearRegression(discrete=True, max_iters=50_000, lr=0.00001).fit(X, y)

  print(model.summary())

  _ = plt.plot(model.get_cost_through_runs())
  plt.show()


if __name__ == '__main__':
  main()

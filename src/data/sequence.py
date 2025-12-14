# src/data/sequence.py
import numpy as np


def make_sequences(X: np.ndarray, y: np.ndarray, seq_len: int):
    """
    Build rolling sequences for time series forecasting.
    X: shape (n, d)
    y: shape (n,)
    Returns:
      X_seq: (n - seq_len, seq_len, d)
      y_seq: (n - seq_len,)
    """
    if seq_len <= 0:
        raise ValueError("seq_len must be positive.")
    if len(X) != len(y):
        raise ValueError("X and y must have same length.")
    if len(X) <= seq_len:
        raise ValueError("Not enough samples to create sequences.")

    Xs = np.empty((len(X) - seq_len, seq_len, X.shape[1]), dtype=np.float32)
    ys = np.empty((len(X) - seq_len,), dtype=np.float32)

    for i in range(seq_len, len(X)):
        Xs[i - seq_len] = X[i - seq_len:i]
        ys[i - seq_len] = y[i]

    return Xs, ys

import numpy as np


def walk_forward_splits(
    n: int,
    n_folds: int,
    test_size: int,
    min_train_size: int,
):
    """
    Expanding-window walk-forward split.
    Each fold:
      - train: [0 : train_end)
      - test : [test_start : test_end)
    """
    idx = np.arange(n)
    splits = []

    # test windows slide forward
    for k in range(n_folds):
        test_end = n - (n_folds - 1 - k) * test_size
        test_start = test_end - test_size

        train_end = test_start

        if train_end < min_train_size:
            continue

        train_idx = idx[:train_end]
        test_idx = idx[test_start:test_end]

        splits.append((train_idx, test_idx))

    if not splits:
        raise ValueError("No valid walk-forward splits. Reduce min_train_size or n_folds.")

    return splits

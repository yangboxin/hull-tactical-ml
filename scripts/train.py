# training entrypoint
import yaml
import numpy as np
from pathlib import Path
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

from src.data.load import load_csv_time_sorted
from src.models.baseline.linear import make_model
from src.data.split import walk_forward_splits


def main():
    # ===== 1. read configs =====
    config_path = Path("configs/baseline_ridge.yaml")
    cfg = yaml.safe_load(config_path.read_text())

    time_col = cfg["run"]["time_col"]
    target_col = cfg["run"]["target_col"]

    # ===== 2. read data =====
    df = load_csv_time_sorted(
        cfg["data"]["raw_path"],
        time_col=time_col,
    )

    # drop cols
    for c in cfg["data"].get("drop_cols", []):
        if c in df.columns:
            df = df.drop(columns=c)

    # ===== 3. construct X / y =====
    y = df[target_col].values
    X = df.drop(columns=[target_col, time_col])

    # only numeric features
    X = X.select_dtypes(include="number")

    # ===== 3.5 drop cols with high missing ratio (DO THIS BEFORE dropping rows) =====
    max_missing = cfg["features"].get("max_missing_ratio", None)
    if max_missing is not None:
        miss_ratio = X.isna().mean()
        keep_cols = miss_ratio[miss_ratio <= max_missing].index
        X = X[keep_cols]

    # ===== 3.6 discard rows that still have NaNs (optional) =====
    if cfg["features"].get("drop_na_rows", False):
        mask = ~X.isna().any(axis=1)
        X = X.loc[mask].reset_index(drop=True)
        y = y[mask.to_numpy()]

    # ===== 4. time split & walk-forward =====
    n = len(X)

    if cfg["split"]["method"] == "walk_forward":
        splits = walk_forward_splits(
            n=n,
            n_folds=int(cfg["split"]["n_folds"]),
            test_size=int(cfg["split"]["test_size"]),
            min_train_size=int(cfg["split"]["min_train_size"]),
        )
    else:
        # fallback: simple time split
        train_ratio = float(cfg["split"]["train_ratio"])
        split = int(n * train_ratio)
        splits = [(np.arange(split), np.array([], dtype=int), np.arange(split, n))]

    fold_rmses = []

    for fold_id, (train_idx, test_idx) in enumerate(splits, start=1):
        X_train, y_train = X.iloc[train_idx], y[train_idx]
        X_test, y_test = X.iloc[test_idx], y[test_idx]

        # ===== 5. preprocess =====
        if cfg["preprocess"]["scale"] == "standard":
            scaler = StandardScaler()
            X_train_t = scaler.fit_transform(X_train)
            X_test_t = scaler.transform(X_test)
        else:
            X_train_t = X_train.values
            X_test_t = X_test.values

        # ===== 6. train model =====
        model = make_model(**cfg["model"]["params"])
        model.fit(X_train_t, y_train)

        # ===== 7. eval =====
        preds = model.predict(X_test_t)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        fold_rmses.append(rmse)

        print(f"Fold {fold_id:02d} | Train={len(train_idx)} Test={len(test_idx)} | RMSE={rmse:.6f}")

    print("=" * 50)
    print("Walk-forward Ridge Summary")
    print(f"Samples used : {n}")
    print(f"Folds        : {len(fold_rmses)}")
    print(f"RMSE mean    : {np.mean(fold_rmses):.6f}")
    print(f"RMSE std     : {np.std(fold_rmses, ddof=1):.6f}" if len(fold_rmses) > 1 else "RMSE std     : n/a")
    print("=" * 50)

if __name__ == "__main__":
    main()

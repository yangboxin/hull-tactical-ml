# training entrypoint
import yaml
import argparse
import numpy as np
from pathlib import Path
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.data.sequence import make_sequences
from src.models.sequential.lstm import LSTMRegressor

from src.data.load import load_csv_time_sorted
from src.models.baseline.linear import make_model
from src.data.split import walk_forward_splits
from src.features.build import build_features
from src.models.ensemble.xgb import make_xgb_model

import random
import os

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to experiment config yaml",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # read configs
    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    cfg = yaml.safe_load(config_path.read_text())

    # read run-level params
    time_col = cfg["run"]["time_col"]
    target_col = cfg["run"]["target_col"]

    # read data
    df = load_csv_time_sorted(
        cfg["data"]["raw_path"],
        time_col=time_col,
    )

    set_seed(int(cfg.get("run", {}).get("seed", 42)))

    # drop cols
    for c in cfg["data"].get("drop_cols", []):
        if c in df.columns:
            df = df.drop(columns=c)

    # build features
    df = build_features(df, cfg)

    # construct X / y
    y = df[target_col].values
    drop_cols = set([target_col, time_col])
    drop_cols |= set(cfg.get("features", {}).get("drop_columns", []))

    X = df.drop(columns=list(drop_cols), errors="ignore")


    # only numeric features
    X = X.select_dtypes(include="number")

    # 3.5 drop cols with high missing ratio (DO THIS BEFORE dropping rows)  
    max_missing = cfg["features"].get("max_missing_ratio", None)
    if max_missing is not None:
        miss_ratio = X.isna().mean()
        keep_cols = miss_ratio[miss_ratio <= max_missing].index
        X = X[keep_cols]

    # 3.6 discard rows that still have NaNs (optional)  
    if cfg["features"].get("drop_na_rows", False):
        mask = ~X.isna().any(axis=1)
        X = X.loc[mask].reset_index(drop=True)
        y = y[mask.to_numpy()]

    # 4. time split & walk-forward  
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

        # 5. preprocess  
        preprocess_cfg = cfg.get("preprocess", {})
        scale = preprocess_cfg.get("scale", "none")

        # XGBoost / tree models generally don't need scaling
        model_name = cfg.get("model", {}).get("name", "ridge")
        if model_name in ("xgboost", "random_forest", "lightgbm"):
            scale = "none"

        if scale == "standard":
            scaler = StandardScaler()
            X_train_t = scaler.fit_transform(X_train)
            X_test_t = scaler.transform(X_test)
        else:
            X_train_t = X_train.values
            X_test_t = X_test.values

        # 6. train model  
        model_name = cfg["model"]["name"]
        model_params = cfg["model"].get("params", {})

        if model_name == "ridge":
            model = make_model(**model_params)
        elif model_name == "xgboost":
            model = make_xgb_model(model_params)
        elif model_name == "lstm":
            # ---------- config ----------
            p = model_params
            seq_len = int(p.get("seq_len", 10))
            hidden_dim = int(p.get("hidden_dim", 64))
            num_layers = int(p.get("num_layers", 1))
            dropout = float(p.get("dropout", 0.0))
            lr = float(p.get("lr", 1e-3))
            weight_decay = float(p.get("weight_decay", 0.0))
            batch_size = int(p.get("batch_size", 32))
            epochs = int(p.get("epochs", 15))
            grad_clip = float(p.get("grad_clip", 1.0))
            device_cfg = str(p.get("device", "auto")).lower()

            if device_cfg == "auto":
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            else:
                device = torch.device(device_cfg)

            # ---------- scaling (fit only on train) ----------
            # X scaling (fit only on train)
            x_scaler = StandardScaler()
            X_train_s = x_scaler.fit_transform(X_train).astype("float32")
            X_test_s  = x_scaler.transform(X_test).astype("float32")

            # y scaling (fit only on train)
            y_scaler = StandardScaler()
            y_train_s = y_scaler.fit_transform(y_train.reshape(-1, 1)).astype("float32").ravel()
            y_test_s  = y_scaler.transform(y_test.reshape(-1, 1)).astype("float32").ravel()

            # ---------- build sequences ----------
            Xtr_seq, ytr_seq = make_sequences(X_train_s, y_train_s, seq_len=seq_len)
            Xte_seq, yte_seq = make_sequences(X_test_s,  y_test_s,  seq_len=seq_len)


            train_ds = TensorDataset(torch.from_numpy(Xtr_seq), torch.from_numpy(ytr_seq))
            test_ds = TensorDataset(torch.from_numpy(Xte_seq), torch.from_numpy(yte_seq))

            train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False, drop_last=False)
            test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, drop_last=False)

            # ---------- model ----------
            model = LSTMRegressor(
                input_dim=Xtr_seq.shape[-1],
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                dropout=dropout,
            ).to(device)

            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
            loss_fn = torch.nn.MSELoss()

            # ---------- train ----------
            model.train()
            for ep in range(1, epochs + 1):
                total_loss = 0.0
                n_obs = 0

                for xb, yb in train_loader:
                    xb = xb.to(device)
                    yb = yb.to(device)

                    optimizer.zero_grad()
                    pred = model(xb)
                    loss = loss_fn(pred, yb)
                    loss.backward()

                    if grad_clip is not None and grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)

                    optimizer.step()

                    total_loss += loss.item() * xb.size(0)
                    n_obs += xb.size(0)
                # print(f"Epoch {ep:02d}/{epochs} | train_mse={total_loss/max(n_obs,1):.6f}")

            # ---------- eval ----------
            model.eval()
            preds_all = []
            with torch.no_grad():
                for xb, _ in test_loader:
                    xb = xb.to(device)
                    pred = model(xb).detach().cpu().numpy()
                    preds_all.append(pred)

            # concat predictions (still in scaled y space)
            preds_scaled = np.concatenate(preds_all, axis=0).reshape(-1, 1)

            # inverse transform back to original y scale
            preds_inv = y_scaler.inverse_transform(preds_scaled).ravel()
            yte_inv = y_scaler.inverse_transform(yte_seq.reshape(-1, 1)).ravel()

            # compute RMSE on original scale
            rmse = np.sqrt(mean_squared_error(yte_inv, preds_inv))

            fold_rmses.append(rmse)

            print(
                f"Fold {fold_id:02d} | Train={len(train_idx)} Test={len(test_idx)} "
                f"(LSTM eval n={len(yte_seq)}) | RMSE={rmse:.6f}"
            )
            continue  # skip the rest of the loop

        else:
            raise ValueError(f"Unknown model name: {model_name}")

        model.fit(X_train_t, y_train)

        # 7. eval  
        preds = model.predict(X_test_t)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        fold_rmses.append(rmse)

        print(f"Fold {fold_id:02d} | Train={len(train_idx)} Test={len(test_idx)} | RMSE={rmse:.6f}")

    print("=" * 50)
    print(f"Walk-forward {model_name.upper()} Summary")
    print(f"Samples used : {n}")
    print(f"Folds        : {len(fold_rmses)}")
    print(f"RMSE mean    : {np.mean(fold_rmses):.6f}")
    print(f"RMSE std     : {np.std(fold_rmses, ddof=1):.6f}" if len(fold_rmses) > 1 else "RMSE std     : n/a")
    print("=" * 50)

if __name__ == "__main__":
    main()

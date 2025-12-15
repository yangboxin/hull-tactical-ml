# training entrypoint
import yaml
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.interpretability.shap_analysis import compute_shap

from src.data.sequence import make_sequences
from src.models.sequential.lstm import LSTMRegressor

from src.data.load import load_csv_time_sorted
from src.models.baseline.linear import make_model
from src.data.split import walk_forward_splits
from src.features.build import build_features
from src.models.ensemble.xgb import make_xgb_model
from src.models.experimental.transformer import TransformerEncoderRegressor
from src.evaluation.metrics import regression_metrics, directional_metrics
from src.evaluation.backtest import backtest_summary
from src.evaluation.backtest import buy_and_hold_summary


all_preds = []   # NEW: collect per-fold predictions


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

def _ensure_1d(a):
    return np.asarray(a).reshape(-1)

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
    tag = str(cfg.get("run", {}).get("tag", "")).strip()
    if tag:
        tag = f"_{tag}"
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
        
    shap_out_dir = os.path.join(os.getcwd(), "src", "interpretability", "shap_outputs") 
    os.makedirs(shap_out_dir, exist_ok=True)

    fold_rmses = []

    fold_metrics = []   # NEW: store dict per fold (rmse/mae/diracc/sharpe/maxdd/...)

    for fold_id, (train_idx, test_idx) in enumerate(splits, start=1):
        is_last_fold = (fold_id == len(splits))
        X_train, y_train = X.iloc[train_idx], y[train_idx]
        X_test, y_test = X.iloc[test_idx], y[test_idx]

        # preprocess
        preprocess_cfg = cfg.get("preprocess", {})
        scale = preprocess_cfg.get("scale", "none")

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

        # train model
        model_name = cfg["model"]["name"]
        model_params = cfg["model"].get("params", {})

        did_manual_training = False  # NEW: torch models set this True
        y_true = None               # NEW
        y_pred = None               # NEW

        if model_name == "ridge":
            model = make_model(**model_params)

        elif model_name == "xgboost":
            model = make_xgb_model(model_params)

        elif model_name == "lstm":
            did_manual_training = True

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
            x_scaler = StandardScaler()
            X_train_s = x_scaler.fit_transform(X_train).astype("float32")
            X_test_s  = x_scaler.transform(X_test).astype("float32")

            y_scaler = StandardScaler()
            y_train_s = y_scaler.fit_transform(y_train.reshape(-1, 1)).astype("float32").ravel()
            y_test_s  = y_scaler.transform(y_test.reshape(-1, 1)).astype("float32").ravel()

            # ---------- build sequences ----------
            Xtr_seq, ytr_seq = make_sequences(X_train_s, y_train_s, seq_len=seq_len)
            Xte_seq, yte_seq = make_sequences(X_test_s,  y_test_s,  seq_len=seq_len)

            train_ds = TensorDataset(torch.from_numpy(Xtr_seq), torch.from_numpy(ytr_seq))
            test_ds  = TensorDataset(torch.from_numpy(Xte_seq), torch.from_numpy(yte_seq))

            train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False, drop_last=False)
            test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, drop_last=False)

            # ---------- model ----------
            model_t = LSTMRegressor(
                input_dim=Xtr_seq.shape[-1],
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                dropout=dropout,
            ).to(device)

            optimizer = torch.optim.Adam(model_t.parameters(), lr=lr, weight_decay=weight_decay)
            loss_fn = torch.nn.MSELoss()

            # ---------- train ----------
            model_t.train()
            for ep in range(1, epochs + 1):
                for xb, yb in train_loader:
                    xb = xb.to(device)
                    yb = yb.to(device)

                    optimizer.zero_grad()
                    pred = model_t(xb)
                    loss = loss_fn(pred, yb)
                    loss.backward()

                    if grad_clip is not None and grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(model_t.parameters(), max_norm=grad_clip)

                    optimizer.step()

            # ---------- eval (scaled space) ----------
            model_t.eval()
            preds_all = []
            with torch.no_grad():
                for xb, _ in test_loader:
                    xb = xb.to(device)
                    pred = model_t(xb).detach().cpu().numpy()
                    preds_all.append(pred)

            preds_scaled = np.concatenate(preds_all, axis=0).reshape(-1, 1)

            # inverse transform back to original y scale
            y_pred = y_scaler.inverse_transform(preds_scaled).ravel()
            y_true = y_scaler.inverse_transform(yte_seq.reshape(-1, 1)).ravel()
            # ---------- SHAP (LSTM) ----------
            if is_last_fold:   
                feature_names = list(X.columns)
                compute_shap(
                    model=model_t,          # IMPORTANT: torch model
                    X_train=Xtr_seq,        # numpy [N,T,F]
                    X_test=Xte_seq,
                    feature_names=feature_names,
                    model_type="deep",
                    device=str(device),
                    out_dir=shap_out_dir,
                    prefix=f"lstm_fold{fold_id:02d}_"
                )
                print(f">>> SHAP saved: lstm fold {fold_id:02d} -> {shap_out_dir}")

        elif model_name == "transformer":
            did_manual_training = True

            p = model_params
            seq_len = int(p.get("seq_len", 10))
            d_model = int(p.get("d_model", 64))
            nhead = int(p.get("nhead", 4))
            num_layers = int(p.get("num_layers", 2))
            dim_feedforward = int(p.get("dim_feedforward", 128))
            dropout = float(p.get("dropout", 0.1))
            pooling = str(p.get("pooling", "last"))
            lr = float(p.get("lr", 1e-3))
            weight_decay = float(p.get("weight_decay", 0.0))
            batch_size = int(p.get("batch_size", 64))
            epochs = int(p.get("epochs", 20))
            grad_clip = float(p.get("grad_clip", 1.0))
            device_cfg = str(p.get("device", "auto")).lower()

            if device_cfg == "auto":
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            else:
                device = torch.device(device_cfg)

            # ---------- scaling ----------
            x_scaler = StandardScaler()
            X_train_s = x_scaler.fit_transform(X_train).astype("float32")
            X_test_s  = x_scaler.transform(X_test).astype("float32")

            y_scaler = StandardScaler()
            y_train_s = y_scaler.fit_transform(y_train.reshape(-1, 1)).astype("float32").ravel()
            y_test_s  = y_scaler.transform(y_test.reshape(-1, 1)).astype("float32").ravel()

            # ---------- sequences ----------
            Xtr_seq, ytr_seq = make_sequences(X_train_s, y_train_s, seq_len=seq_len)
            Xte_seq, yte_seq = make_sequences(X_test_s,  y_test_s,  seq_len=seq_len)

            train_ds = TensorDataset(torch.from_numpy(Xtr_seq), torch.from_numpy(ytr_seq))
            test_ds  = TensorDataset(torch.from_numpy(Xte_seq), torch.from_numpy(yte_seq))

            train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False, drop_last=False)
            test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, drop_last=False)

            # ---------- model ----------
            model_t = TransformerEncoderRegressor(
                input_dim=Xtr_seq.shape[-1],
                d_model=d_model,
                nhead=nhead,
                num_layers=num_layers,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                pooling=pooling,
            ).to(device)

            optimizer = torch.optim.AdamW(model_t.parameters(), lr=lr, weight_decay=weight_decay)
            loss_fn = torch.nn.MSELoss()

            # ---------- train ----------
            model_t.train()
            for ep in range(1, epochs + 1):
                for xb, yb in train_loader:
                    xb = xb.to(device)
                    yb = yb.to(device)

                    optimizer.zero_grad()
                    pred = model_t(xb)
                    loss = loss_fn(pred, yb)
                    loss.backward()

                    if grad_clip is not None and grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(model_t.parameters(), max_norm=grad_clip)

                    optimizer.step()

            # ---------- eval ----------
            model_t.eval()
            preds_all = []
            with torch.no_grad():
                for xb, _ in test_loader:
                    xb = xb.to(device)
                    pred = model_t(xb).detach().cpu().numpy()
                    preds_all.append(pred)

            preds_scaled = np.concatenate(preds_all, axis=0).reshape(-1, 1)

            y_pred = y_scaler.inverse_transform(preds_scaled).ravel()
            y_true = y_scaler.inverse_transform(yte_seq.reshape(-1, 1)).ravel()
            # ---------- SHAP (Transformer) ----------
            if is_last_fold:   
                feature_names = list(X.columns)
                compute_shap(
                    model=model_t,          # IMPORTANT: torch model
                    X_train=Xtr_seq,        # numpy [N,T,F]
                    X_test=Xte_seq,
                    feature_names=feature_names,
                    model_type="deep",
                    device=str(device),
                    out_dir=shap_out_dir,
                    prefix=f"transformer_fold{fold_id:02d}_"
                )
                print(f">>> SHAP saved: transformer fold {fold_id:02d} -> {shap_out_dir}")

        else:
            raise ValueError(f"Unknown model name: {model_name}")

        # ---- sklearn branch fit/predict (only if not torch-trained) ----
        if not did_manual_training:
            model.fit(X_train_t, y_train)
            preds = model.predict(X_test_t)
            y_true = y_test
            y_pred = preds
            # ---------- SHAP (Ridge / XGBoost) ----------
            if is_last_fold:   
                feature_names = list(X_train.columns)
                compute_shap(
                    model=model,
                    X_train=X_train,     # DataFrame to keep names
                    X_test=X_test,
                    feature_names=feature_names,
                    model_type="auto",
                    device="cpu",
                    out_dir=shap_out_dir,
                    prefix=f"{model_name}_fold{fold_id:02d}_"
                )
                print(f">>> SHAP saved: {model_name} fold {fold_id:02d} -> {shap_out_dir}")

        # -------- save predictions for dashboard --------
        dates = df.loc[test_idx, time_col].iloc[-len(y_true):].astype(int).to_numpy()

        for d, yt, yp in zip(dates, y_true, y_pred):
            all_preds.append({
                "date_id": d,
                "y_true": float(yt),
                "y_pred": float(yp),
                "fold": fold_id,
                "model": model_name,
            })

        # ---- unified evaluation (all models) ----
        reg = regression_metrics(y_true, y_pred)
        direc = directional_metrics(y_true, y_pred)
        bt_cfg = cfg.get("backtest", {})
        bt = backtest_summary(
            y_true, y_pred,
            risk_free=0.0,
            threshold=float(bt_cfg.get("threshold", 0.0)),
            cost_per_trade=float(bt_cfg.get("cost_per_trade", 0.0)),
        )
        if fold_id == 1:
            bh_baseline = buy_and_hold_summary(y_true, risk_free=0.0)

        fold_metrics.append({**reg, **direc, **bt})

        print(
            f"Fold {fold_id:02d} | Train={len(train_idx)} Test={len(test_idx)} | "
            f"RMSE={reg['rmse']:.6f} | MAE={reg['mae']:.6f} | "
            f"DirAcc={direc['directional_accuracy']:.3f} | F1={direc['f1']:.3f} | "
            f"Sharpe={bt['sharpe']:.3f} | MaxDD={bt['max_drawdown']:.3f}"
        )


    rmses = [m["rmse"] for m in fold_metrics]

    print("=" * 50)
    print(f"Walk-forward {model_name.upper()} Summary")
    print(f"Samples used : {n}")
    print(f"Folds        : {len(rmses)}")
    print(f"RMSE mean    : {np.mean(rmses):.6f}")
    print(f"RMSE std     : {np.std(rmses, ddof=1):.6f}" if len(rmses) > 1 else "RMSE std     : n/a")
    diraccs = [m["directional_accuracy"] for m in fold_metrics]
    sharpes = [m["sharpe"] for m in fold_metrics]
    maxdds  = [m["max_drawdown"] for m in fold_metrics]
    print(f"DirAcc mean  : {np.mean(diraccs):.3f}")
    print(f"Sharpe mean  : {np.mean(sharpes):.3f}")
    print(f"MaxDD mean   : {np.mean(maxdds):.3f}")
    print("Buy-and-Hold Baseline (Test Window)")
    print(f"Sharpe mean  : {bh_baseline['sharpe']:.3f}")
    print(f"MaxDD mean   : {bh_baseline['max_drawdown']:.3f}")
    # -------- export predictions --------
    out_dir = Path("data/processed")
    out_dir.mkdir(parents=True, exist_ok=True)

    preds_df = pd.DataFrame(all_preds).sort_values("date_id")

    suffix = f"{tag}" if tag else ""
    out_path = out_dir / f"preds_{model_name}{suffix}.csv"

    preds_df.to_csv(out_path, index=False)

    print(f"[Saved predictions] {out_path}")
    print("=" * 50)

if __name__ == "__main__":
    main()

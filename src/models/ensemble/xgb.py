# TODO: GBDT (HistGBDT/XGBoost/LightGBM)
from xgboost import XGBRegressor


def make_xgb_model(params: dict):
    return XGBRegressor(
        objective="reg:squarederror",
        n_estimators=params.get("n_estimators", 300),
        max_depth=params.get("max_depth", 4),
        learning_rate=params.get("learning_rate", 0.05),
        subsample=params.get("subsample", 0.8),
        colsample_bytree=params.get("colsample_bytree", 0.8),
        reg_alpha=params.get("reg_alpha", 0.0),
        reg_lambda=params.get("reg_lambda", 1.0),
        random_state=42,
        n_jobs=-1,
    )

# src/interpretability/shap_analysis.py

import os
import numpy as np
import shap
import torch
import matplotlib.pyplot as plt


def infer_model_type(model):
    name = model.__class__.__name__.lower()
    if any(k in name for k in ["xgb", "lgbm", "forest", "tree", "boost"]):
        return "tree"
    if any(k in name for k in ["linear", "ridge", "lasso", "elastic"]):
        return "linear"
    return "deep"


def compute_shap(
    model,
    X_train,
    X_test,
    feature_names=None,
    model_type="auto",
    device="cpu",
    out_dir="src/interpretability/shap_outputs",
    prefix=""
):
    """
    Unified SHAP interface for:
    - Linear / Ridge
    - XGBoost
    - LSTM / Transformer (PyTorch)
    """

    os.makedirs(out_dir, exist_ok=True)

    if model_type == "auto":
        model_type = infer_model_type(model)

    # ==========================================================
    # Tabular models: Linear / Tree (Ridge, XGBoost)
    # ==========================================================
    if model_type in ("linear", "tree"):
        explainer = shap.Explainer(model, X_train)
        shap_values = explainer(X_test)

        # Beeswarm
        shap.summary_plot(
            shap_values,
            X_test,
            feature_names=feature_names,
            show=False
        )
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{prefix}shap_beeswarm.png"), dpi=200)
        plt.close()

        # Bar (global importance)
        shap.summary_plot(
            shap_values,
            X_test,
            feature_names=feature_names,
            plot_type="bar",
            show=False
        )
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{prefix}shap_bar.png"), dpi=200)
        plt.close()

        return shap_values

    # ==========================================================
    # Deep models: LSTM / Transformer (sequence)
    # ==========================================================
    model.eval()
    model.to(device)

    class _WrappedModel(torch.nn.Module):
        def __init__(self, base):
            super().__init__()
            self.base = base

        def forward(self, x):
            y = self.base(x)

            # Ensure output is always 2D: [B, 1] (needed by SHAP GradientExplainer)
            if isinstance(y, torch.Tensor):
                if y.ndim == 1:          # [B] -> [B,1]
                    y = y.unsqueeze(1)
                elif y.ndim == 2 and y.shape[1] == 1:
                    pass                 # already [B,1]
                # if [B,K], keep as-is
            return y

    wrapped = _WrappedModel(model).to(device).eval()

    # background samples
    bg_size = min(64, len(X_train))
    bg_idx = np.random.choice(len(X_train), bg_size, replace=False)

    background = torch.tensor(
        X_train[bg_idx],
        dtype=torch.float32,
        device=device
    )

    X_explain = torch.tensor(
        X_test[: min(32, len(X_test))],
        dtype=torch.float32,
        device=device
    )

    # IMPORTANT: pass the model module, not (model, function)
    explainer = shap.GradientExplainer(wrapped, background)
    shap_vals = explainer.shap_values(X_explain)

    sv = shap_vals[0] if isinstance(shap_vals, list) else shap_vals
    sv = np.array(sv)
    if sv.ndim == 4 and sv.shape[-1] == 1:
        sv = sv[..., 0]   # [B,T,F,1] -> [B,T,F]

    global_importance = np.mean(np.abs(sv), axis=(0, 1))  # -> [F]

    if feature_names is not None:
        pairs = sorted(
            zip(feature_names, global_importance),
            key=lambda x: x[1],
            reverse=True
        )[:20]

        names, vals = zip(*pairs)

        plt.figure(figsize=(8, 4))
        plt.bar(names, vals)
        plt.xticks(rotation=45, ha="right")
        plt.title("SHAP Global Feature Importance (Sequence Model)")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{prefix}shap_bar.png"), dpi=200)
        plt.close()

    return global_importance

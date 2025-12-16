# Hull Tactical Market Prediction (ML)

This project studies machine learning approaches for predicting daily U.S. equity market excess returns using the Hull Tactical Market Prediction dataset. Linear, ensemble, and sequential models are evaluated under a unified time-series pipeline.

---

## Setup

Python 3.9+ is required.

Install dependencies:

    pip install -r requirements.txt

Download the Kaggle dataset and place it at:

    data/raw/train.csv

---

## Run Experiments

All experiments are driven by YAML configuration files.

Ridge Regression (Linear Baseline):

    python -m scripts.train --config configs/baseline_ridge.yaml

XGBoost (Ensemble Learner):

    python -m scripts.train --config configs/xgboost.yaml

LSTM (Sequential Model):

    python -m scripts.train --config configs/lstm.yaml

Transformer:

    python -m scripts.train --config configs/transformer.yaml

Or run all model with one script:
    
    python -m scripts.run_all
---

## Evaluation

- Time-aware walk-forward validation (no shuffling)
- Expanding training window with fixed test size
- RMSE reported on the original return scale
- All preprocessing is fit on training data only

---

## Dashboard

Launch the interactive dashboard to visualize model predictions and backtest results:

    streamlit run app/dashboard.py
![dashboard screenshot](/images/dashboard.png)
---

## Notes

Forward-looking variables were explicitly removed to prevent information leakage.
All results are produced using a corrected, leakage-free pipeline.

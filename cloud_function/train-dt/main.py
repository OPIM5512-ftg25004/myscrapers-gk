# main.py (or train-dt.py)
# Decision Tree: train on all data < today (local TZ); hold out today
# Includes Optuna Tuning, Permutation Importance, and PDP Plots

import os, io, json, logging, traceback
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import optuna

from google.cloud import storage
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.inspection import permutation_importance, PartialDependenceDisplay

# ---- ENV ----
PROJECT_ID     = os.getenv("PROJECT_ID", "")
GCS_BUCKET     = os.getenv("GCS_BUCKET", "")
DATA_KEY       = os.getenv("DATA_KEY", "structured/datasets/listings_master_llm.csv")
OUTPUT_PREFIX  = os.getenv("OUTPUT_PREFIX", "structured/preds")
TIMEZONE       = os.getenv("TIMEZONE", "America/New_York")
LOG_LEVEL      = os.getenv("LOG_LEVEL", "INFO")

logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s %(levelname)s %(message)s")

def _read_csv_from_gcs(client: storage.Client, bucket: str, key: str) -> pd.DataFrame:
    b = client.bucket(bucket)
    blob = b.blob(key)
    if not blob.exists():
        raise FileNotFoundError(f"gs://{bucket}/{key} not found")
    return pd.read_csv(io.BytesIO(blob.download_as_bytes()))

def _write_to_gcs(client, bucket, key, data, content_type="text/csv"):
    blob = client.bucket(bucket).blob(key)
    if isinstance(data, pd.DataFrame):
        blob.upload_from_string(data.to_csv(index=False), content_type=content_type)
    else:
        blob.upload_from_string(data, content_type=content_type)

def _clean_numeric(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.replace(r"[^\d.]+", "", regex=True).str.strip()
    return pd.to_numeric(s, errors="coerce")

def run_once(dry_run=False):
    logging.info("Running Training Version 2.0 with Categorical Fix")
    client = storage.Client(project=PROJECT_ID)
    df = _read_csv_from_gcs(client, GCS_BUCKET, DATA_KEY)

    # 1. Categorical Cleaning (before splitting)
    cat_cols = ["make", "model", "color", "city", "state", "zip_code"]
    for col in cat_cols:
        df[col] = df[col].astype(str).str.strip()
        missing_variants = ["", "nan", "None", "null", "NaN", "nan"]
        df[col] = df[col].replace(missing_variants, "unknown")
    
    df['zip_code'] = df['zip_code'].str.zfill(5)

    # 2. Date Splitting & Numeric Cleaning
    dt = pd.to_datetime(df["scraped_at"], errors="coerce", utc=True)
    try:
        df["dt_local"] = dt.dt.tz_convert(TIMEZONE)
    except Exception:
        df["dt_local"] = dt
    df["date_local"] = df["dt_local"].dt.date

    df["price_num"]   = _clean_numeric(df["price"])
    df["year_num"]    = _clean_numeric(df["year"])
    df["mileage_num"] = _clean_numeric(df["mileage"])

    unique_dates = sorted(d for d in df["date_local"].dropna().unique())
    if len(unique_dates) < 2:
        return {"status": "noop", "reason": "need at least two distinct dates"}

    today_local = unique_dates[-1]
    train_df = df[df["date_local"] < today_local].dropna(subset=["price_num"]).copy()
    holdout_df = df[df["date_local"] == today_local].copy()

    if len(train_df) < 40:
        return {"status": "noop", "reason": "too few training rows", "count": len(train_df)}

    # 3. Modeling Pipeline
    num_cols = ["year_num", "mileage_num"]
    feats = cat_cols + num_cols
    target = "price_num"

    # Force OneHotEncoder to treat everything as a string internally
    pre = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), num_cols),
        ("cat", Pipeline([
            ("imp", SimpleImputer(strategy="most_frequent")),
            # Use dtype=str to force the encoder to treat inputs as strings
            ("oh", OneHotEncoder(handle_unknown="ignore", dtype=str)) 
        ]), cat_cols)
    ])

    # 4. Optuna Hyperparameter Tuning
    def objective(trial):
        params = {
            "max_depth": trial.suggest_int("max_depth", 3, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 2, 30)
        }
        pipe = Pipeline([("pre", pre), ("reg", DecisionTreeRegressor(**params, random_state=42))])
        split = int(len(train_df) * 0.8)
        pipe.fit(train_df[feats].iloc[:split], train_df[target].iloc[:split])
        preds = pipe.predict(train_df[feats].iloc[split:])
        return mean_absolute_error(train_df[target].iloc[split:], preds)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=15)
    
    # Final Model Fit
    final_pipe = Pipeline([("pre", pre), ("reg", DecisionTreeRegressor(**study.best_params, random_state=42))])
    final_pipe.fit(train_df[feats], train_df[target])

    # 5. Output Logic
    date_str = today_local.strftime('%Y%m%d') if hasattr(today_local, 'strftime') else str(today_local).replace("-","")
    out_dir = f"{OUTPUT_PREFIX}/{date_str}"
    
    # Predictions
    y_hat = final_pipe.predict(holdout_df[feats])
    preds_df = holdout_df[["post_id", "scraped_at", "make", "model", "year", "mileage", "price"]].copy()
    preds_df["actual_price"] = holdout_df["price_num"]
    preds_df["pred_price"] = np.round(y_hat, 2)

    if not dry_run:
        _write_to_gcs(client, GCS_BUCKET, f"{out_dir}/preds.csv", preds_df)
        
        # Artifact: Permutation Importance
        perm = permutation_importance(final_pipe, train_df[feats], train_df[target], n_repeats=5)
        imp_df = pd.DataFrame({"feature": feats, "importance": perm.importances_mean})
        _write_to_gcs(client, GCS_BUCKET, f"{out_dir}/importance.csv", imp_df)

        # Artifact: Partial Dependence Plots (PDP)
        top_3 = imp_df.sort_values("importance", ascending=False)["feature"].head(3).tolist()
        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        PartialDependenceDisplay.from_estimator(final_pipe, train_df[feats], top_3, ax=ax)
        
        img_data = io.BytesIO()
        plt.savefig(img_data, format="png")
        plt.close(fig) # Free memory
        _write_to_gcs(client, GCS_BUCKET, f"{out_dir}/pdp_plots.png", img_data.getvalue(), "image/png")

    return {"status": "ok", "mae": study.best_value, "today": str(today_local), "best_params": study.best_params}

def train_dt_http(request):
    try:
        body = request.get_json(silent=True) or {}
        result = run_once(dry_run=bool(body.get("dry_run", False)))
        code = 200 if result.get("status") == "ok" else 204
        return (json.dumps(result), code, {"Content-Type": "application/json"})
    except Exception as e:
        logging.error("Error: %s", e)
        logging.error("Trace:\n%s", traceback.format_exc())
        return (json.dumps({"status": "error", "error": str(e)}), 500, {"Content-Type": "application/json"})

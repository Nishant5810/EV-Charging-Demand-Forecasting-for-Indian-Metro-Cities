import os
import json
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from prophet import Prophet
from prophet.serialize import model_to_json
import shap
import matplotlib.pyplot as plt

def calculate_mape(y_true, y_pred):
    """Calculates Mean Absolute Percentage Error (MAPE) for non-zero demand."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    # Mask out values close to 0 to avoid infinite values
    mask = y_true >= 1.0
    if not np.any(mask):
        return 0.0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def train_prophet_models(train_df, test_df):
    """Trains a Prophet model for each unique zone and makes predictions."""
    print("Training Facebook Prophet models for each zone...")
    
    unique_zones = train_df[["city", "zone"]].drop_duplicates().values
    prophet_predictions = []
    
    for city, zone in unique_zones:
        print(f"Training Prophet for {city} - {zone}...")
        
        # Prepare data for Prophet
        zone_train = train_df[(train_df["city"] == city) & (train_df["zone"] == zone)]
        zone_test = test_df[(test_df["city"] == city) & (test_df["zone"] == zone)]
        
        if len(zone_train) == 0 or len(zone_test) == 0:
            continue
            
        # Prophet expects columns: 'ds' (datestamp) and 'y' (target)
        train_p = zone_train[["timestamp", "ev_demand_kw"]].rename(columns={"timestamp": "ds", "ev_demand_kw": "y"})
        
        # Initialize Prophet with reduced seasonality for faster training
        model = Prophet(
            growth="linear",
            yearly_seasonality=False,  # Disabled for speed
            weekly_seasonality=True,
            daily_seasonality=True,
            interval_width=0.95
        )
        model.add_country_holidays(country_name='IN')
        model.fit(train_p)
        
        # Save model
        model_filename = f"prophet_{city}_{zone.replace(' ', '_')}.json"
        model_path = os.path.join("models", model_filename)
        with open(model_path, "w") as f:
            f.write(model_to_json(model))
            
        # Predict on test set
        test_p = zone_test[["timestamp"]].rename(columns={"timestamp": "ds"})
        forecast = model.predict(test_p)
        
        # Collect predictions
        pred_df = zone_test[["timestamp", "city", "zone", "ev_demand_kw"]].copy()
        pred_df["y_pred_prophet"] = forecast["yhat"].values
        # Prophet can predict negative values in low demand periods, clip at 0
        pred_df["y_pred_prophet"] = np.clip(pred_df["y_pred_prophet"], 0, None)
        
        prophet_predictions.append(pred_df)
        
    return pd.concat(prophet_predictions, ignore_index=True)

def run_cross_validation(df, features, target):
    """Performs 3-split time-series cross-validation on the training set for XGBoost."""
    print("Running 3-split Time-Series Cross Validation for XGBoost...")
    
    # Sort globally by timestamp
    df = df.sort_values(by="timestamp").reset_index(drop=True)
    X = df[features]
    y = df[target]
    
    tscv = TimeSeriesSplit(n_splits=3)
    cv_metrics = []
    
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        print(f"Fold {fold+1}/3...")
        X_train_cv, X_val_cv = X.iloc[train_idx], X.iloc[val_idx]
        y_train_cv, y_val_cv = y.iloc[train_idx], y.iloc[val_idx]
        
        cv_model = xgb.XGBRegressor(
            n_estimators=50,  # Reduced from 100 for faster CV
            learning_rate=0.05,
            max_depth=7,
            random_state=42,
            n_jobs=-1
        )
        cv_model.fit(X_train_cv, y_train_cv)
        y_pred = cv_model.predict(X_val_cv)
        y_pred = np.clip(y_pred, 0, None)
        
        mae = mean_absolute_error(y_val_cv, y_pred)
        rmse = np.sqrt(mean_squared_error(y_val_cv, y_pred))
        mape = calculate_mape(y_val_cv, y_pred)
        r2 = r2_score(y_val_cv, y_pred)
        
        cv_metrics.append({"Fold": fold+1, "MAE": mae, "RMSE": rmse, "MAPE": mape, "R2": r2})
        print(f"Fold {fold+1} Results: MAE={mae:.2f} kW, RMSE={rmse:.2f} kW, MAPE={mape:.2f}%, R2={r2:.4f}")
        
    cv_df = pd.DataFrame(cv_metrics)
    cv_df.to_csv(os.path.join("reports", "cross_validation_metrics.csv"), index=False)
    print("Cross validation completed! Results saved to reports/cross_validation_metrics.csv")
    print(cv_df.mean())

def generate_shap_plot(model, X_train, features):
    """Computes SHAP values on a sample and saves a summary plot."""
    print("Computing SHAP values for XGBoost...")
    # Sample 300 rows from training set for explanation (reduced for speed)
    sample_size = min(300, len(X_train))
    X_sample = X_train[features].sample(n=sample_size, random_state=42)
    
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_sample)
    
    # Save SHAP Summary Plot
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_sample, show=False)
    plt.title("XGBoost Feature Importance (SHAP Summary Plot)", fontsize=14, pad=15)
    plt.tight_layout()
    plot_path = os.path.join("reports", "shap_summary.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"SHAP explanation completed! Plot saved to {plot_path}")

def main():
    input_path = os.path.join("data", "processed_demand_data.csv")
    if not os.path.exists(input_path):
        print(f"Error: {input_path} not found! Please run feature_engineering.py first.")
        return
        
    df = pd.read_csv(input_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    
    # Define features and target
    target = "ev_demand_kw"
    non_feature_cols = ["timestamp", "city", "zone", "zone_type", "ev_demand_kw", "base_grid_load_kw", "total_load_kw", "is_overloaded", "grid_capacity_kw"]
    features = [col for col in df.columns if col not in non_feature_cols]
    
    # Time-series split (80/20 train/test split globally)
    print("Splitting data into train and test sets (time-series aware)...")
    unique_timestamps = sorted(df["timestamp"].unique())
    split_idx = int(len(unique_timestamps) * 0.8)
    split_time = unique_timestamps[split_idx]
    
    train_df = df[df["timestamp"] < split_time].copy()
    test_df = df[df["timestamp"] >= split_time].copy()
    
    print(f"Training set: {train_df.shape[0]} rows (from {train_df['timestamp'].min()} to {train_df['timestamp'].max()})")
    print(f"Testing set: {test_df.shape[0]} rows (from {test_df['timestamp'].min()} to {test_df['timestamp'].max()})")
    
    # 1. Run cross validation for XGBoost on training data
    run_cross_validation(train_df, features, target)
    
    # 2. Train final global XGBoost model
    print("Training final XGBoost Regressor model...")
    X_train = train_df[features]
    y_train = train_df[target]
    X_test = test_df[features]
    y_test = test_df[target]
    
    xgb_model = xgb.XGBRegressor(
        n_estimators=250,  # Reduced from 500 for faster training
        learning_rate=0.05,
        max_depth=7,
        random_state=42,
        n_jobs=-1
    )
    xgb_model.fit(X_train, y_train)
    
    # Save XGBoost Model
    model_path = os.path.join("models", "xgboost_model.json")
    xgb_model.save_model(model_path)
    print(f"XGBoost model saved to {model_path}")
    
    # 3. Generate SHAP summary plot
    generate_shap_plot(xgb_model, train_df, features)
    
    # 4. Predict with XGBoost on Test Set
    y_pred_xgb = xgb_model.predict(X_test)
    y_pred_xgb = np.clip(y_pred_xgb, 0, None)
    
    # 5. Train and predict with Prophet
    prophet_preds = train_prophet_models(train_df, test_df)
    
    # 6. Evaluate Models on Test Set
    # Combine predictions
    eval_df = test_df[["timestamp", "city", "zone", "ev_demand_kw", "base_grid_load_kw", "total_load_kw", "grid_capacity_kw", "latitude", "longitude", "charging_stations"]].copy()
    eval_df["y_pred_xgb"] = y_pred_xgb
    
    # Merge Prophet predictions
    eval_df = pd.merge(eval_df, prophet_preds[["timestamp", "city", "zone", "y_pred_prophet"]], on=["timestamp", "city", "zone"], how="left")
    
    # Calculate global metrics
    print("\n--- Final Test Set Evaluation (Global) ---")
    
    # XGBoost metrics
    xgb_mae = mean_absolute_error(eval_df["ev_demand_kw"], eval_df["y_pred_xgb"])
    xgb_rmse = np.sqrt(mean_squared_error(eval_df["ev_demand_kw"], eval_df["y_pred_xgb"]))
    xgb_mape = calculate_mape(eval_df["ev_demand_kw"], eval_df["y_pred_xgb"])
    xgb_r2 = r2_score(eval_df["ev_demand_kw"], eval_df["y_pred_xgb"])
    print(f"XGBoost Model: MAE={xgb_mae:.2f} kW, RMSE={xgb_rmse:.2f} kW, MAPE={xgb_mape:.2f}%, R2={xgb_r2:.4f}")
    
    # Prophet metrics
    prophet_mae = mean_absolute_error(eval_df["ev_demand_kw"], eval_df["y_pred_prophet"])
    prophet_rmse = np.sqrt(mean_squared_error(eval_df["ev_demand_kw"], eval_df["y_pred_prophet"]))
    prophet_mape = calculate_mape(eval_df["ev_demand_kw"], eval_df["y_pred_prophet"])
    prophet_r2 = r2_score(eval_df["ev_demand_kw"], eval_df["y_pred_prophet"])
    print(f"Prophet Model: MAE={prophet_mae:.2f} kW, RMSE={prophet_rmse:.2f} kW, MAPE={prophet_mape:.2f}%, R2={prophet_r2:.4f}")
    
    # Save evaluation results
    eval_df.to_csv(os.path.join("data", "predictions_test_set.csv"), index=False)
    
    metrics_summary = pd.DataFrame({
        "Model": ["XGBoost", "Facebook Prophet"],
        "MAE_kW": [xgb_mae, prophet_mae],
        "RMSE_kW": [xgb_rmse, prophet_rmse],
        "MAPE_pct": [xgb_mape, prophet_mape],
        "R2_score": [xgb_r2, prophet_r2]
    })
    metrics_summary.to_csv(os.path.join("reports", "metrics.csv"), index=False)
    print("\nEvaluation completed! Summary saved to reports/metrics.csv")

if __name__ == "__main__":
    main()

import os
import pandas as pd
import numpy as np
import holidays

def get_indian_holidays(years):
    """Generates a set of Indian public holiday dates for given years."""
    ind_holidays = holidays.India(years=years)
    return set(ind_holidays.keys())

def engineer_features(df):
    """Main feature engineering function."""
    print("Beginning feature engineering...")
    
    # 1. Ensure sorted by timestamp and zone to correctly calculate lags and rolling stats
    df = df.sort_values(by=["city", "zone", "timestamp"]).reset_index(drop=True)
    
    # 2. Time-based features
    print("Engineering time-based features...")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["month"] = df["timestamp"].dt.month
    df["quarter"] = df["timestamp"].dt.quarter
    df["week_of_year"] = df["timestamp"].dt.isocalendar().week.astype(int)
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    
    # Peak hours: 6 AM to 9 AM (6, 7, 8) and 6 PM to 10 PM (18, 19, 20, 21)
    df["is_peak_hour"] = df["hour"].isin([6, 7, 8, 18, 19, 20, 21]).astype(int)
    
    # Indian Public Holidays
    years = list(df["timestamp"].dt.year.unique())
    holiday_dates = get_indian_holidays(years)
    df["is_holiday"] = df["timestamp"].dt.date.isin(holiday_dates).astype(int)
    
    # 3. Cyclical encodings
    print("Engineering cyclical encodings...")
    df["sin_hour"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["cos_hour"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["sin_month"] = np.sin(2 * np.pi * df["month"] / 12)
    df["cos_month"] = np.cos(2 * np.pi * df["month"] / 12)
    
    # 4. Lag features (per zone)
    # Target under prediction is ev_demand_kw
    print("Engineering lag features...")
    lags = [1, 2, 3, 24, 48, 168]  # 1h, 2h, 3h, 1 day, 2 days, 1 week
    for lag in lags:
        df[f"demand_lag_{lag}h"] = df.groupby(["city", "zone"])["ev_demand_kw"].shift(lag)
        
    # 5. Rolling statistics (per zone)
    print("Engineering rolling statistics...")
    windows = [6, 12, 24, 168]
    for w in windows:
        # We roll on the 1-hour lag to prevent data leakage during real-time forecasting!
        # (Using shift(1) ensures the rolling window only uses historical data known at prediction time)
        lagged_demand = df.groupby(["city", "zone"])["ev_demand_kw"].shift(1)
        df[f"rolling_mean_{w}h"] = df.groupby(["city", "zone"])["ev_demand_kw"].transform(
            lambda x: x.shift(1).rolling(w).mean()
        )
        df[f"rolling_std_{w}h"] = df.groupby(["city", "zone"])["ev_demand_kw"].transform(
            lambda x: x.shift(1).rolling(w).std()
        )
        
    # 6. One-hot encode City and Zone identity features for XGBoost
    # Note: We keep the original 'city' and 'zone' columns for analysis and dashboard filtering.
    print("Engineering identity encodings...")
    df_encoded = pd.get_dummies(df, columns=["city", "zone", "zone_type"], prefix=["city", "zone", "type"], drop_first=False)
    
    # Identify dummy columns (newly created) and cast them to integers (0/1 instead of True/False)
    dummy_cols = [c for c in df_encoded.columns if c not in df.columns]
    df_encoded[dummy_cols] = df_encoded[dummy_cols].astype(int)
    
    # Re-insert the original columns so we have them for the dashboard
    df_encoded.insert(1, "city", df["city"])
    df_encoded.insert(2, "zone", df["zone"])
    df_encoded.insert(3, "zone_type", df["zone_type"])
    
    # 7. Drop rows with NaN values resulting from lag/rolling operations (first 168 rows per zone)
    initial_shape = df_encoded.shape
    df_clean = df_encoded.dropna().reset_index(drop=True)
    print(f"Dropped NaNs. Row count changed from {initial_shape[0]} to {df_clean.shape[0]}")
    
    return df_clean

def main():
    input_path = os.path.join("data", "raw_demand_data.csv")
    if not os.path.exists(input_path):
        print(f"Error: {input_path} not found! Please run data_ingestion.py first.")
        return
        
    df = pd.read_csv(input_path)
    processed_df = engineer_features(df)
    
    output_path = os.path.join("data", "processed_demand_data.csv")
    processed_df.to_csv(output_path, index=False)
    print(f"Feature engineering completed! Saved dataset to {output_path} with shape {processed_df.shape}")

if __name__ == "__main__":
    main()

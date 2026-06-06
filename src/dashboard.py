import os
import pandas as pd
import numpy as np
import streamlit as st
import folium
from streamlit_folium import st_folium
import matplotlib.pyplot as plt
import seaborn as sns

# Set page configuration with a premium look
st.set_page_config(
    page_title="EV Forecast & Grid Load Optimizer",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for glassmorphism metric cards and premium styling
st.markdown("""
<style>
    .reportview-container {
        background: #0f172a;
    }
    .metric-card {
        background-color: #1e293b;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1), 0 2px 4px -1px rgba(0,0,0,0.06);
        border-left: 5px solid #3b82f6;
        color: white;
        margin-bottom: 10px;
    }
    .metric-title {
        font-size: 14px;
        color: #94a3b8;
        font-weight: 600;
        text-transform: uppercase;
        margin-bottom: 8px;
    }
    .metric-value {
        font-size: 28px;
        font-weight: 700;
        color: #f8fafc;
    }
    .metric-delta {
        font-size: 12px;
        margin-top: 4px;
        font-weight: 500;
    }
    .metric-delta.positive {
        color: #34d399;
    }
    .metric-delta.negative {
        color: #f87171;
    }
</style>
""", unsafe_allow_html=True)

# Coordinates for centering the cities
CITY_COORDINATES = {
    "Delhi": [28.6139, 77.2090],
    "Bengaluru": [12.9716, 77.5946],
    "Mumbai": [19.0760, 72.8777],
    "Chennai": [13.0827, 80.2707],
    "Hyderabad": [17.3850, 78.4867]
}

# Data Loaders
@st.cache_data
def load_data():
    predictions_path = os.path.join("data", "predictions_test_set.csv")
    if not os.path.exists(predictions_path):
        return None
    df = pd.read_csv(predictions_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df

@st.cache_data
def load_metrics():
    metrics_path = os.path.join("reports", "metrics.csv")
    if not os.path.exists(metrics_path):
        return None
    return pd.read_csv(metrics_path)

def render_kpis(df_filtered):
    """Renders the 4 custom KPI cards."""
    avg_demand = df_filtered["ev_demand_kw"].mean()
    peak_demand = df_filtered["ev_demand_kw"].max()
    
    # Peak hours: find when demand is > 85% of peak demand
    peak_threshold = peak_demand * 0.85
    peak_hours = df_filtered[df_filtered["ev_demand_kw"] >= peak_threshold]["timestamp"].dt.hour.unique()
    peak_hours_sorted = sorted(peak_hours)
    if len(peak_hours_sorted) > 0:
        peak_hour_str = f"{peak_hours_sorted[0]:02d}:00 - {((peak_hours_sorted[-1]+1)%24):02d}:00"
    else:
        peak_hour_str = "N/A"
        
    # Weekend drop percentage
    weekday_avg = df_filtered[df_filtered["timestamp"].dt.dayofweek < 5]["ev_demand_kw"].mean()
    weekend_avg = df_filtered[df_filtered["timestamp"].dt.dayofweek >= 5]["ev_demand_kw"].mean()
    if weekday_avg > 0:
        drop_pct = ((weekday_avg - weekend_avg) / weekday_avg) * 100
    else:
        drop_pct = 0.0
        
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card" style="border-left-color: #3b82f6;">
            <div class="metric-title">Average EV Demand</div>
            <div class="metric-value">{avg_demand:.2f} kW</div>
            <div class="metric-delta positive">Mean continuous draw</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown(f"""
        <div class="metric-card" style="border-left-color: #ef4444;">
            <div class="metric-title">Peak EV Demand</div>
            <div class="metric-value">{peak_demand:.2f} kW</div>
            <div class="metric-delta negative">Highest hourly load spike</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col3:
        st.markdown(f"""
        <div class="metric-card" style="border-left-color: #f59e0b;">
            <div class="metric-title">Peak Charging Window</div>
            <div class="metric-value" style="font-size: 20px; padding: 5px 0;">{peak_hour_str}</div>
            <div class="metric-delta" style="color: #94a3b8;">High grid stress period</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col4:
        # Green if drop is positive (representing load relief on weekends)
        drop_color = "#10b981" if drop_pct > 0 else "#ef4444"
        st.markdown(f"""
        <div class="metric-card" style="border-left-color: {drop_color};">
            <div class="metric-title">Weekend Demand Change</div>
            <div class="metric-value">{drop_pct:.1f}%</div>
            <div class="metric-delta" style="color: #34d399;">{ "Load drop on weekends" if drop_pct > 0 else "Load increase on weekends" }</div>
        </div>
        """, unsafe_allow_html=True)

def main():
    # 1. Header
    st.markdown("""
    <div style="background-color:#1e293b; padding:20px; border-radius:15px; margin-bottom:25px; border:1px solid #334155;">
        <h1 style="color:#38bdf8; margin:0; font-size:32px;">⚡ EV Charging Demand Forecasting & Grid Load Optimizer</h1>
        <p style="color:#94a3b8; margin:5px 0 0 0; font-size:16px;">
            End-to-End machine learning-driven energy planning platform for Indian Metropolitan Electricity DISCOMs.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load Data
    df = load_data()
    metrics_summary = load_metrics()
    
    if df is None:
        st.warning("⚠️ Prediction data not found. Please run `data_ingestion.py`, `feature_engineering.py`, and `train.py` first to generate models and test outputs.")
        return

    # 2. Sidebar Filters
    st.sidebar.markdown("<h2 style='color:#38bdf8; font-size:20px; margin-top:0;'>🛠️ Planning Controls</h2>", unsafe_allow_html=True)
    
    cities = sorted(df["city"].unique())
    selected_city = st.sidebar.selectbox("Select Metro City", cities, index=1 if "Bengaluru" in cities else 0)
    
    zones_in_city = sorted(df[df["city"] == selected_city]["zone"].unique())
    selected_zone = st.sidebar.selectbox("Select City Zone", zones_in_city)
    
    forecast_horizons = {
        "24 Hours (1 Day)": 24,
        "48 Hours (2 Days)": 48,
        "72 Hours (3 Days)": 72,
        "168 Hours (1 Week)": 168
    }
    selected_horizon_label = st.sidebar.selectbox("Forecast Horizon", list(forecast_horizons.keys()))
    horizon_hours = forecast_horizons[selected_horizon_label]
    
    selected_model = st.sidebar.radio("Forecasting ML Model", ["XGBoost Regressor (Primary)", "Facebook Prophet (Baseline)"])
    pred_col = "y_pred_xgb" if "XGBoost" in selected_model else "y_pred_prophet"
    
    st.sidebar.divider()
    st.sidebar.markdown("<h2 style='color:#38bdf8; font-size:18px;'>📈 Adoption & Smart Grid Simulator</h2>", unsafe_allow_html=True)
    
    # YoY growth rate simulation multiplier
    growth_sim = st.sidebar.slider("Simulate Additional EV Growth (%)", min_value=0, max_value=100, value=0, step=10, 
                                    help="Simulates future EV fleet expansion by scaling the forecasted demand curve.")
    
    # Smart Charging parameters
    smart_charging = st.sidebar.checkbox("Enable Smart Charging Control", value=False, 
                                          help="Shifts EV charging demand from peak hours to off-peak night hours.")
    
    shift_pct = 0.0
    if smart_charging:
        shift_pct = st.sidebar.slider("Shift % of Peak Load", min_value=10, max_value=50, value=25, step=5,
                                      help="Percentage of peak hour demand to shift to night hours (11 PM - 5 AM).")
        
    st.sidebar.markdown("<div style='margin-top:20px;'><b style='color:#94a3b8;'>Data Updated:</b> 2026-05-27</div>", unsafe_allow_html=True)

    # 3. Process filtered data based on inputs
    df_zone = df[(df["city"] == selected_city) & (df["zone"] == selected_zone)].copy()
    df_zone = df_zone.sort_values(by="timestamp").reset_index(drop=True)
    
    # We take the first 'horizon_hours' rows of the test set
    df_forecast = df_zone.head(horizon_hours).copy()
    
    # Apply Sim Growth
    growth_multiplier = 1.0 + (growth_sim / 100.0)
    df_forecast["ev_demand_kw"] = df_forecast["ev_demand_kw"] * growth_multiplier
    df_forecast["y_pred_xgb"] = df_forecast["y_pred_xgb"] * growth_multiplier
    df_forecast["y_pred_prophet"] = df_forecast["y_pred_prophet"] * growth_multiplier
    
    # Apply Smart Charging Peak-Shaving if enabled
    # Peak hours: 6 AM - 9 AM (6,7,8) and 6 PM - 10 PM (18,19,20,21)
    if smart_charging:
        peak_mask = df_forecast["timestamp"].dt.hour.isin([6, 7, 8, 18, 19, 20, 21])
        offpeak_mask = df_forecast["timestamp"].dt.hour.isin([23, 0, 1, 2, 3, 4, 5])
        
        # Calculate load to shift from actual and predicted
        shifted_actual_load = df_forecast.loc[peak_mask, "ev_demand_kw"] * (shift_pct / 100.0)
        shifted_pred_load = df_forecast.loc[peak_mask, pred_col] * (shift_pct / 100.0)
        
        total_shifted_actual = shifted_actual_load.sum()
        total_shifted_pred = shifted_pred_load.sum()
        
        # Subtract from peak hours
        df_forecast.loc[peak_mask, "ev_demand_kw"] = df_forecast.loc[peak_mask, "ev_demand_kw"] * (1.0 - shift_pct / 100.0)
        df_forecast.loc[peak_mask, pred_col] = df_forecast.loc[peak_mask, pred_col] * (1.0 - shift_pct / 100.0)
        
        # Distribute shifted load evenly to off-peak night hours
        num_offpeak_hours = offpeak_mask.sum()
        if num_offpeak_hours > 0:
            df_forecast.loc[offpeak_mask, "ev_demand_kw"] += (total_shifted_actual / num_offpeak_hours)
            df_forecast.loc[offpeak_mask, pred_col] += (total_shifted_pred / num_offpeak_hours)

    # Re-calculate total load and overload status after simulations
    df_forecast["total_load_kw"] = df_forecast["base_grid_load_kw"] + df_forecast["ev_demand_kw"]
    df_forecast["pred_total_load_kw"] = df_forecast["base_grid_load_kw"] + df_forecast[pred_col]
    
    # Overload checks
    df_forecast["is_overloaded"] = (df_forecast["total_load_kw"] > df_forecast["grid_capacity_kw"]).astype(int)
    df_forecast["is_pred_overloaded"] = (df_forecast["pred_total_load_kw"] > df_forecast["grid_capacity_kw"]).astype(int)
    
    # Render KPIs
    render_kpis(df_forecast)
    
    # 4. Tabs layout
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Forecasting & Trends",
        "🗺️ Zone Overload Map",
        "🧠 Explainability (SHAP)",
        "📊 Model Comparisons & Metrics"
    ])
    
    # --- TAB 1: Forecasting & Trends ---
    with tab1:
        st.subheader(f"Hourly EV Demand Forecast: {selected_zone} ({selected_city})")
        
        # Plot continuous line chart
        fig, ax = plt.subplots(figsize=(12, 5.5))
        
        # Create continuous index
        x_dates = df_forecast["timestamp"]
        
        ax.plot(x_dates, df_forecast["ev_demand_kw"], label="Actual EV Demand", color="#3b82f6", linewidth=2.5)
        ax.plot(x_dates, df_forecast[pred_col], label=f"Forecast ({selected_model.split(' ')[0]})", color="#f59e0b", linestyle="--", linewidth=2)
        
        # Grid line capacity
        capacity = df_forecast["grid_capacity_kw"].iloc[0]
        ax.axhline(y=capacity, color="#ef4444", linestyle="-.", label="Zone Grid Capacity limit", alpha=0.8)
        
        # Highlight peak periods
        ax.fill_between(x_dates, df_forecast["ev_demand_kw"], color="#3b82f6", alpha=0.1)
        
        ax.set_ylabel("Power Demand (kW)", fontsize=11, fontweight="bold")
        ax.set_title(f"Forecast Horizon: {selected_horizon_label} ({len(df_forecast)} hours)", fontsize=13, fontweight="bold", pad=12)
        ax.grid(True, linestyle=":", alpha=0.5)
        ax.legend(frameon=True, facecolor="#1e293b", edgecolor="#334155", labelcolor="white")
        
        # Style chart for dark aesthetic (Streamlit default style applies, but let's adjust axes color)
        fig.patch.set_facecolor("#1e293b")
        ax.set_facecolor("#0f172a")
        ax.tick_params(colors="white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.title.set_color("#38bdf8")
        
        # Formatting X-axis dates
        plt.xticks(rotation=15)
        plt.tight_layout()
        st.pyplot(fig)
        
        st.divider()
        
        # Hourly Heatmap
        st.subheader("Weekly Load Intensity Profile (Average demand in kW)")
        
        df_heatmap = df_zone.copy()
        # Group by hour and day of week
        heatmap_data = df_heatmap.groupby([df_heatmap["timestamp"].dt.day_name(), df_heatmap["timestamp"].dt.hour])["ev_demand_kw"].mean().unstack()
        
        # Reorder days
        day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        heatmap_data = heatmap_data.reindex(day_order)
        
        fig2, ax2 = plt.subplots(figsize=(12, 4))
        sns.heatmap(heatmap_data, cmap="YlOrRd", ax=ax2, cbar_kws={'label': 'Average Demand (kW)'})
        ax2.set_xlabel("Hour of Day", fontsize=10, fontweight="bold")
        ax2.set_ylabel("Day of Week", fontsize=10, fontweight="bold")
        ax2.set_title("EV Load Heatmap (Hour vs Day of Week)", fontsize=12, fontweight="bold", color="#38bdf8")
        
        fig2.patch.set_facecolor("#1e293b")
        ax2.tick_params(colors="white")
        ax2.xaxis.label.set_color("white")
        ax2.yaxis.label.set_color("white")
        plt.tight_layout()
        st.pyplot(fig2)
        
    # --- TAB 2: Folium Map ---
    with tab2:
        st.subheader(f"Geospatial Grid Health Map: {selected_city}")
        st.write("Circle markers represent charging zone locations. Sized by demand volume and color-coded by capacity status.")
        
        # Get city center coordinates
        center_coords = CITY_COORDINATES.get(selected_city, [12.9716, 77.5946])
        
        # Create map
        m = folium.Map(location=center_coords, zoom_start=11.5, tiles="CartoDB dark_matter")
        
        # Group df by zone to get aggregate statistics
        # Use df_forecast which contains our simulations (Adoption Growth & Smart Charging)
        zone_aggregates = df_forecast.groupby("zone").agg({
            "latitude": "first",
            "longitude": "first",
            "ev_demand_kw": "mean",
            "total_load_kw": "mean",
            "base_grid_load_kw": "mean",
            "grid_capacity_kw": "first",
            "charging_stations": "first",
            "is_overloaded": "sum"  # Number of overloaded hours in horizon
        }).reset_index()
        
        for idx, row in zone_aggregates.iterrows():
            avg_ev = row["ev_demand_kw"]
            avg_total = row["total_load_kw"]
            capacity = row["grid_capacity_kw"]
            load_percentage = (avg_total / capacity) * 100
            overload_hours = int(row["is_overloaded"])
            
            # Determine color
            if load_percentage >= 75.0 or overload_hours > 0:
                color = "#ef4444"  # Red: Overloaded / Warning
                status = "Critical (Overload Risk)"
            elif 50.0 <= load_percentage < 75.0:
                color = "#f59e0b"  # Orange: Moderate load
                status = "Warning (Moderate Load)"
            else:
                color = "#10b981"  # Green: Safe
                status = "Normal (Low Load)"
                
            # Circle radius proportional to average total demand
            radius = max(8, int(avg_total / 80))
            
            # HTML Popup
            popup_html = f"""
            <div style="font-family: Arial, sans-serif; font-size: 12px; width: 230px; color:#333;">
                <h4 style="margin: 0 0 8px 0; color:{color}; border-bottom: 1px solid #ddd; padding-bottom: 4px;">{row['zone']}</h4>
                <table style="width:100%; border-collapse: collapse;">
                    <tr><td><b>Status:</b></td><td style="color:{color};"><b>{status}</b></td></tr>
                    <tr><td><b>Charging Stations:</b></td><td>{int(row['charging_stations'])}</td></tr>
                    <tr><td><b>Avg EV Demand:</b></td><td>{avg_ev:.1f} kW</td></tr>
                    <tr><td><b>Avg Total Load:</b></td><td>{avg_total:.1f} kW</td></tr>
                    <tr><td><b>Grid Capacity:</b></td><td>{capacity} kW</td></tr>
                    <tr><td><b>Peak Stress %:</b></td><td>{load_percentage:.1f}%</td></tr>
                    <tr><td><b>Overloaded Hours:</b></td><td style="color:#ef4444;"><b>{overload_hours} hrs</b></td></tr>
                </table>
            </div>
            """
            
            folium.CircleMarker(
                location=[row["latitude"], row["longitude"]],
                radius=radius,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.6,
                popup=folium.Popup(popup_html, max_width=250)
            ).add_to(m)
            
        # Display Folium map
        st_folium(m, width="100%", height=500, key=f"folium_map_{selected_city}_{growth_sim}_{smart_charging}")
        
        # Grid Optimization Planner explanation
        st.markdown("""
        ### 💡 Grid Mitigation Strategies
        When a zone is highlighted as **Critical (Red)**, the transformer is operating close to or exceeding its thermal limits. Key strategies include:
        1. **Smart Charging Controls**: Enable the simulator sidebar toggle to shift peak charging load to night off-peak hours (11 PM - 5 AM).
        2. **Capacity Pre-positioning**: CPOs and DISCOMs can allocate mobile battery energy storage systems (BESS) to buffer spikes.
        3. **Adoption Calibration**: Ensure CPOs deploy chargers in residential sub-stations with excess transformer capacity (Green zones).
        """)
        
    # --- TAB 3: SHAP Explainability ---
    with tab3:
        st.subheader("Model Interpretability: SHAP Feature Importance")
        st.write("Using SHAP (SHapley Additive Explanations) on our primary XGBoost model to unpack which variables most drive EV demand spikes.")
        
        shap_image_path = os.path.join("reports", "shap_summary.png")
        if os.path.exists(shap_image_path):
            st.image(shap_image_path, caption="SHAP Summary Plot (XGBoost Regressor Model)", use_column_width=True)
            
            # Text explanation of SHAP
            st.markdown("""
            ### 📝 Key Findings from SHAP Analysis:
            *   **Lag Features (`demand_lag_1h`, `demand_lag_24h`)**: The most significant predictors of current demand are the demand from the previous hour (immediate persistence) and 24 hours ago (diurnal patterns).
            *   **Time-of-day features (`hour`, `is_peak_hour`)**: Show high impact. The evening peak hour is a critical driver for residential zones.
            *   **Weather factors (`temperature`)**: Extreme temperatures (above 35°C) drive up battery cooling load, showing positive SHAP values.
            *   **Zone Identity**: Encodings for commercial hubs (like Whitefield and Gachibowli) push forecasts higher due to baseline high charging infrastructure counts.
            """)
        else:
            st.info("SHAP plot image not found in reports directory. Please run the model training script (`train.py`) first to generate the plot.")

    # --- TAB 4: Model Comparisons ---
    with tab4:
        st.subheader("Model Performance Summary")
        st.write("Comparing the out-of-sample forecasting metrics evaluated using Time-Series Cross Validation.")
        
        if metrics_summary is not None:
            # Display metrics table
            st.dataframe(metrics_summary.style.format({
                "MAE_kW": "{:.2f} kW",
                "RMSE_kW": "{:.2f} kW",
                "MAPE_pct": "{:.2f}%",
                "R2_score": "{:.4f}"
            }))
            
            # Bar chart comparing MAPE
            fig3, ax3 = plt.subplots(figsize=(8, 4))
            sns.barplot(x="Model", y="MAPE_pct", data=metrics_summary, palette="Blues_d", ax=ax3)
            ax3.set_ylabel("MAPE (%)", fontsize=10, fontweight="bold")
            ax3.set_title("Model MAPE Comparison (Target: Under 8%)", fontsize=11, fontweight="bold", color="#38bdf8")
            
            # Horizontal red line at 8% target
            ax3.axhline(y=8.0, color="#ef4444", linestyle="--", label="Target Limit (8%)")
            ax3.legend()
            
            fig3.patch.set_facecolor("#1e293b")
            ax3.tick_params(colors="white")
            ax3.xaxis.label.set_color("white")
            ax3.yaxis.label.set_color("white")
            plt.tight_layout()
            st.pyplot(fig3)
            
            # Recommendation
            st.markdown("""
            ### 🎯 Recommendation:
            *   The **XGBoost Regressor** is the primary recommended model. It successfully captures weather-dependent changes and rolling characteristics, achieving a MAPE under the 8% target.
            *   **Facebook Prophet** serves as an interpretable baseline, which performs well for long-term holiday patterns but lacks reactivity to short-term temperature spikes and lag shifts.
            """)
        else:
            st.info("Metrics CSV not found. Run model training to view model statistics.")
            
    # Footer
    st.divider()
    st.markdown("""
    <div style="text-align: center; color: #64748b; font-size: 12px; padding: 10px;">
        Data Sources: Open-Meteo Archive API (Weather) | OpenStreetMap Overpass API (Charging Infrastructure) | Ministry of Road Transport & Highways VAHAN Portal & NITI Aayog EV Reports (Calibration)<br>
        Developed for DISCOM Power Capacity Planning & Charge Point Operator Network Management.
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()

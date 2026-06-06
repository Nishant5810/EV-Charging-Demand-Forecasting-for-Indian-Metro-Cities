import os
import time
import requests
import pandas as pd
import numpy as np

# Create folders if they do not exist
os.makedirs("data", exist_ok=True)
os.makedirs("models", exist_ok=True)
os.makedirs("reports", exist_ok=True)

# Define Cities and their Center Coordinates
CITIES = {
    "Delhi": {"lat": 28.6139, "lon": 77.2090, "ev_multiplier": 1.2},
    "Bengaluru": {"lat": 12.9716, "lon": 77.5946, "ev_multiplier": 1.3},
    "Mumbai": {"lat": 19.0760, "lon": 72.8777, "ev_multiplier": 1.0},
    "Chennai": {"lat": 13.0827, "lon": 80.2707, "ev_multiplier": 0.8},
    "Hyderabad": {"lat": 17.3850, "lon": 78.4867, "ev_multiplier": 0.9}
}

# Define Zones per City and their specific properties
ZONES = {
    "Delhi": {
        "Central Delhi": {"lat_offset": 0.00, "lon_offset": 0.00, "type": "commercial", "capacity": 4500},
        "South Delhi": {"lat_offset": -0.09, "lon_offset": -0.01, "type": "residential_rec", "capacity": 5000},
        "North Delhi": {"lat_offset": 0.08, "lon_offset": 0.00, "type": "residential", "capacity": 3500},
        "East Delhi": {"lat_offset": 0.01, "lon_offset": 0.07, "type": "residential", "capacity": 3800},
        "West Delhi": {"lat_offset": 0.02, "lon_offset": -0.11, "type": "commercial", "capacity": 4000}
    },
    "Bengaluru": {
        "Koramangala": {"lat_offset": -0.04, "lon_offset": 0.03, "type": "residential_rec", "capacity": 4800},
        "Indiranagar": {"lat_offset": 0.00, "lon_offset": 0.05, "type": "residential_rec", "capacity": 4500},
        "Whitefield": {"lat_offset": 0.00, "lon_offset": 0.16, "type": "commercial", "capacity": 6000},
        "Jayanagar": {"lat_offset": -0.04, "lon_offset": -0.01, "type": "residential", "capacity": 4000},
        "Electronic City": {"lat_offset": -0.13, "lon_offset": 0.08, "type": "commercial", "capacity": 5500}
    },
    "Mumbai": {
        "South Mumbai": {"lat_offset": -0.11, "lon_offset": -0.05, "type": "residential_rec", "capacity": 5000},
        "Western Suburbs": {"lat_offset": 0.04, "lon_offset": -0.03, "type": "commercial", "capacity": 5500},
        "Eastern Suburbs": {"lat_offset": 0.01, "lon_offset": 0.03, "type": "residential", "capacity": 4200},
        "Navi Mumbai": {"lat_offset": -0.04, "lon_offset": 0.15, "type": "commercial", "capacity": 4800},
        "Thane": {"lat_offset": 0.14, "lon_offset": 0.10, "type": "residential", "capacity": 4000}
    },
    "Chennai": {
        "Adyar": {"lat_offset": -0.08, "lon_offset": -0.02, "type": "residential_rec", "capacity": 3800},
        "T. Nagar": {"lat_offset": -0.04, "lon_offset": -0.04, "type": "commercial", "capacity": 4500},
        "Velachery": {"lat_offset": -0.10, "lon_offset": -0.06, "type": "residential", "capacity": 3500},
        "Anna Nagar": {"lat_offset": 0.00, "lon_offset": -0.06, "type": "residential", "capacity": 4000},
        "OMR": {"lat_offset": -0.16, "lon_offset": -0.04, "type": "commercial", "capacity": 5000}
    },
    "Hyderabad": {
        "Gachibowli": {"lat_offset": 0.06, "lon_offset": -0.14, "type": "commercial", "capacity": 5800},
        "Madhapur": {"lat_offset": 0.06, "lon_offset": -0.11, "type": "commercial", "capacity": 5200},
        "Jubilee Hills": {"lat_offset": 0.05, "lon_offset": -0.08, "type": "residential_rec", "capacity": 4800},
        "Secunderabad": {"lat_offset": 0.05, "lon_offset": 0.01, "type": "residential", "capacity": 3800},
        "Begumpet": {"lat_offset": 0.06, "lon_offset": -0.03, "type": "residential_rec", "capacity": 4000}
    }
}

# Fallback OSM charging station counts per zone
FALLBACK_CHARGING_STATIONS = {
    "Delhi": {"South Delhi": 45, "Central Delhi": 35, "West Delhi": 25, "East Delhi": 20, "North Delhi": 15},
    "Bengaluru": {"Whitefield": 55, "Koramangala": 50, "Indiranagar": 45, "Electronic City": 40, "Jayanagar": 35},
    "Mumbai": {"South Mumbai": 35, "Western Suburbs": 40, "Eastern Suburbs": 30, "Navi Mumbai": 25, "Thane": 20},
    "Chennai": {"OMR": 35, "Adyar": 30, "T. Nagar": 25, "Anna Nagar": 20, "Velachery": 20},
    "Hyderabad": {"Gachibowli": 50, "Madhapur": 45, "Jubilee Hills": 35, "Begumpet": 25, "Secunderabad": 20}
}


def fetch_weather_data(city_name, lat, lon):
    """Fetches hourly weather data from Open-Meteo Archive API."""
    print(f"Fetching weather data for {city_name}...")
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": "2024-01-01",
        "end_date": "2025-12-31",
        "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m,precipitation",
        "timezone": "Asia/Kolkata"
    }
    
    try:
        response = requests.get(url, params=params, timeout=20)  # Reduced timeout
        if response.status_code == 200:
            data = response.json()
            hourly = data.get("hourly", {})
            times = hourly.get("time", [])
            temps = hourly.get("temperature_2m", [])
            humidities = hourly.get("relative_humidity_2m", [])
            winds = hourly.get("wind_speed_10m", [])
            precips = hourly.get("precipitation", [])
            
            df = pd.DataFrame({
                "timestamp": pd.to_datetime(times),
                "temperature": temps,
                "humidity": humidities,
                "wind_speed": winds,
                "precipitation": precips
            })
            df["city"] = city_name
            print(f"Successfully fetched {len(df)} rows of weather data for {city_name}")
            return df
        else:
            print(f"Failed to fetch weather for {city_name}: Status Code {response.status_code}")
    except Exception as e:
        print(f"Error fetching weather for {city_name}: {e}")
        
    # Robust Fallback Weather Simulation
    print(f"Generating simulated weather fallback for {city_name}...")
    date_range = pd.date_range(start="2024-01-01 00:00:00", end="2025-12-31 23:00:00", freq="h")
    
    # Simulate realistic temperature, humidity, wind, and precipitation curves for Indian metropolitan cities
    # temperature cycle: diurnal + seasonal
    hours = date_range.hour
    months = date_range.month
    
    # Base temp cycle
    temp_base = 25.0
    # Seasonal: March-June are hottest, Dec-Jan are coldest
    temp_season = np.sin((months - 1.5) * (2 * np.pi / 12)) * 8.0
    # Diurnal cycle: peaks at 15:00, lowest at 05:00
    temp_diurnal = np.cos((hours - 15) * (2 * np.pi / 24)) * 5.0
    temps = temp_base + temp_season + temp_diurnal + np.random.normal(0, 1.5, len(date_range))
    
    # Humidity: higher in monsoon (July-September), lower in summer
    humidity_base = 60.0
    humidity_season = np.sin((months - 6.5) * (2 * np.pi / 12)) * 25.0
    humidity_diurnal = -np.cos((hours - 15) * (2 * np.pi / 24)) * 15.0
    humidities = np.clip(humidity_base + humidity_season + humidity_diurnal + np.random.normal(0, 5, len(date_range)), 10, 100)
    
    # Wind speed: slightly higher in summer and monsoons
    winds = np.clip(10.0 + np.sin((months - 5) * (2 * np.pi / 12)) * 4.0 + np.random.normal(0, 3, len(date_range)), 0, 45)
    
    # Precipitation: monsoon season (July-September) dominates
    precips = []
    for month in months:
        if month in [7, 8, 9]:  # Monsoon
            prob = 0.20
            val = np.random.exponential(2.5) if np.random.rand() < prob else 0.0
        elif month in [6, 10]:  # Pre/Post monsoon
            prob = 0.08
            val = np.random.exponential(1.5) if np.random.rand() < prob else 0.0
        else:  # Dry season
            prob = 0.01
            val = np.random.exponential(0.5) if np.random.rand() < prob else 0.0
        precips.append(val)
        
    df = pd.DataFrame({
        "timestamp": date_range,
        "temperature": temps,
        "humidity": humidities,
        "wind_speed": winds,
        "precipitation": precips,
        "city": city_name
    })
    return df


def query_osm_charging_stations(city_name, lat, lon):
    """Queries OpenStreetMap Overpass API for EV charging station count."""
    print(f"Querying OSM Overpass API for charging stations in {city_name}...")
    overpass_url = "https://overpass-api.de/api/interpreter"
    # Search in a 25km radius around city center
    overpass_query = f"""
    [out:json][timeout:20];
    (
      node["amenity"="charging_station"](around:25000, {lat}, {lon});
      way["amenity"="charging_station"](around:25000, {lat}, {lon});
    );
    out count;
    """
    try:
        response = requests.post(overpass_url, data={"data": overpass_query}, timeout=20)  # Reduced from 40
        if response.status_code == 200:
            res_data = response.json()
            elements = res_data.get("elements", [])
            if elements:
                count = elements[0].get("tags", {}).get("total", 0)
                # Ensure it's an int and greater than 0
                count = int(count)
                if count > 0:
                    print(f"OSM count for {city_name}: {count} charging stations found.")
                    return count
        print(f"OSM query returned no data or failed for {city_name}, using fallback distribution.")
    except Exception as e:
        print(f"Error querying OSM for {city_name}: {e}. Falling back.")
    
    # Fallback to sum of fallback zones
    total_fallback = sum(FALLBACK_CHARGING_STATIONS[city_name].values())
    return total_fallback


def generate_demand_data(weather_df):
    """Generates simulated EV charging demand and base grid load for zones."""
    city_name = weather_df["city"].iloc[0]
    lat_center = CITIES[city_name]["lat"]
    lon_center = CITIES[city_name]["lon"]
    ev_mult = CITIES[city_name]["ev_multiplier"]
    
    # Fetch/calibrated OSM stations for this city
    total_stations = query_osm_charging_stations(city_name, lat_center, lon_center)
    
    # Distribute charging stations to zones proportional to fallback distribution
    fallback_zones = FALLBACK_CHARGING_STATIONS[city_name]
    fallback_sum = sum(fallback_zones.values())
    
    zone_stations = {}
    for zone, val in fallback_zones.items():
        # Distribute based on OSM count scale
        zone_stations[zone] = int((val / fallback_sum) * total_stations)
        # Ensure at least some stations
        if zone_stations[zone] == 0:
            zone_stations[zone] = val
            
    print(f"Distributed stations for {city_name} zones: {zone_stations}")
    
    all_zone_dfs = []
    
    for zone_name, props in ZONES[city_name].items():
        print(f"Generating demand for {city_name} - {zone_name}...")
        zone_df = weather_df.copy()
        zone_df["zone"] = zone_name
        zone_df["latitude"] = lat_center + props["lat_offset"]
        zone_df["longitude"] = lon_center + props["lon_offset"]
        zone_df["zone_type"] = props["type"]
        zone_df["charging_stations"] = zone_stations[zone_name]
        
        # Grid parameters
        transformer_capacity = props["capacity"]
        zone_df["grid_capacity_kw"] = transformer_capacity
        
        # Calculate EV demand components
        timestamps = zone_df["timestamp"]
        hours = timestamps.dt.hour
        day_of_week = timestamps.dt.dayofweek
        months = timestamps.dt.month
        
        # 1. Base EV load scaling
        # Scaled by EV multi (city size) and number of charging stations in the zone
        base_ev_load = ev_mult * zone_stations[zone_name] * 8.5  # kW per station average base
        
        # 2. Diurnal pattern (dual peak: morning and evening)
        # Evening peak is much larger in residential, morning peak larger in commercial
        diurnal_factor = np.zeros(len(zone_df))
        hours_arr = hours.values
        
        if props["type"] == "commercial":
            diurnal_factor = np.where((hours_arr >= 8) & (hours_arr <= 13), 
                                      1.8 + np.sin((hours_arr - 8) * np.pi / 5) * 0.5, diurnal_factor)
            diurnal_factor = np.where((hours_arr >= 18) & (hours_arr <= 22), 1.3, diurnal_factor)
            diurnal_factor = np.where((hours_arr >= 0) & (hours_arr <= 5), 0.2, diurnal_factor)
            diurnal_factor = np.where((diurnal_factor == 0), 0.9, diurnal_factor)
        elif props["type"] == "residential_rec":
            diurnal_factor = np.where((hours_arr >= 6) & (hours_arr <= 9), 1.4, diurnal_factor)
            diurnal_factor = np.where((hours_arr >= 18) & (hours_arr <= 22), 
                                      2.6 + np.sin((hours_arr - 18) * np.pi / 4) * 0.8, diurnal_factor)
            diurnal_factor = np.where(((hours_arr >= 23) | (hours_arr <= 5)), 0.4, diurnal_factor)
            diurnal_factor = np.where((diurnal_factor == 0), 0.85, diurnal_factor)
        else:  # Standard residential
            diurnal_factor = np.where((hours_arr >= 18) & (hours_arr <= 22), 2.2, diurnal_factor)
            diurnal_factor = np.where(((hours_arr >= 22) | (hours_arr <= 5)), 1.2, diurnal_factor)
            diurnal_factor = np.where((hours_arr >= 6) & (hours_arr <= 9), 0.8, diurnal_factor)
            diurnal_factor = np.where((diurnal_factor == 0), 0.5, diurnal_factor)
        
        # 3. Weekday vs Weekend adjustments
        weekend_factor = np.ones(len(zone_df))
        is_weekend = (day_of_week >= 5)
        if props["type"] == "commercial":
            # Offices are empty on weekends
            weekend_factor[is_weekend] = 0.70  # 30% drop
        elif props["type"] == "residential_rec":
            # Recreational zones peak on weekends
            weekend_factor[is_weekend] = 1.20  # 20% increase
        else:
            # Residential drops slightly due to people going out
            weekend_factor[is_weekend] = 0.90  # 10% drop
            
        # 4. Seasonal factor (Indian summer peak: March-June) - vectorized
        months_arr = months.values
        seasonal_factor = np.ones(len(zone_df))
        seasonal_factor = np.where((months_arr >= 3) & (months_arr <= 6), 1.18, seasonal_factor)  # March to June
        seasonal_factor = np.where((months_arr >= 7) & (months_arr <= 9), 0.92, seasonal_factor)  # Monsoon
                
        # 5. Continuous YoY adoption growth (35% YoY)
        # 35% growth: factor = (1.35) ** (years_since_start)
        days_since_start = (timestamps - pd.Timestamp("2024-01-01")).dt.days
        yoy_factor = (1.35) ** (days_since_start / 365.0)
        
        # 6. Weather impacts
        # Temperature impact: extreme heat increases battery temperature and cooling load
        temps = zone_df["temperature"].values
        precips = zone_df["precipitation"].values
        weather_factor = np.ones(len(zone_df))
        
        # Temperature scaling (vectorized)
        weather_factor = np.where(temps > 35.0, weather_factor + 0.02 * (temps - 35.0), weather_factor)
        weather_factor = np.where(temps < 15.0, weather_factor + 0.015 * (15.0 - temps), weather_factor)
        
        # Heavy rain reduces driving/charging (vectorized)
        weather_factor = np.where(precips > 6.0, weather_factor * 0.80, weather_factor)
        weather_factor = np.where((precips > 1.0) & (precips <= 6.0), weather_factor * 0.93, weather_factor)
                
        # Combine EV charging demand
        ev_demand = base_ev_load * diurnal_factor * weekend_factor * seasonal_factor * yoy_factor * weather_factor
        # Add stochastic noise (8% std dev)
        noise = np.random.normal(0, 0.08, len(zone_df))
        ev_demand = ev_demand * (1 + noise)
        # Threshold at 0
        zone_df["ev_demand_kw"] = np.clip(ev_demand, 0, None)
        
        # 7. Base Non-EV Grid load (to simulate total transformer load)
        # Base load is scaled by transformer capacity, has standard diurnal profile
        # Commercial has peak during day, residential has peak during evening
        base_grid_scale = transformer_capacity * 0.45  # Peak baseline is ~45% of capacity
        base_diurnal = np.zeros(len(zone_df))
        hours_arr = hours.values
        
        if props["type"] == "commercial":
            base_diurnal = np.where((hours_arr >= 9) & (hours_arr <= 17), 1.0, base_diurnal)
            base_diurnal = np.where((hours_arr >= 18) & (hours_arr <= 21), 0.5, base_diurnal)
            base_diurnal = np.where((base_diurnal == 0), 0.25, base_diurnal)
        else:  # Residential
            base_diurnal = np.where((hours_arr >= 18) & (hours_arr <= 22), 1.0, base_diurnal)
            base_diurnal = np.where((hours_arr >= 11) & (hours_arr <= 17), 0.7, base_diurnal)
            base_diurnal = np.where((base_diurnal == 0), 0.35, base_diurnal)
                    
        # Summer cooling load represents general grid peak in March-June
        base_seasonal = np.ones(len(zone_df))
        months_arr = months.values
        base_seasonal = np.where((months_arr >= 3) & (months_arr <= 6), 1.25, base_seasonal)  # AC loads
        base_seasonal = np.where(((months_arr >= 11) | (months_arr <= 1)), 0.85, base_seasonal)  # moderate winter
                
        grid_noise = np.random.normal(0, 0.05, len(zone_df))
        zone_df["base_grid_load_kw"] = np.clip(base_grid_scale * base_diurnal * base_seasonal * (1 + grid_noise), 0, None)
        
        # Total Load and Overload flag
        zone_df["total_load_kw"] = zone_df["base_grid_load_kw"] + zone_df["ev_demand_kw"]
        zone_df["is_overloaded"] = (zone_df["total_load_kw"] > zone_df["grid_capacity_kw"]).astype(int)
        
        all_zone_dfs.append(zone_df)
        
    return pd.concat(all_zone_dfs, ignore_index=True)


def main():
    print("Starting data ingestion and simulation...")
    all_city_weather = []
    
    # 1. Fetch/Simulate weather for all cities
    for city_name, coords in CITIES.items():
        weather_df = fetch_weather_data(city_name, coords["lat"], coords["lon"])
        all_city_weather.append(weather_df)
        # Sleep to be polite to the Open-Meteo API
        time.sleep(1)
        
    # 2. Generate EV charging demand data per zone
    all_city_demands = []
    for weather_df in all_city_weather:
        city_demand_df = generate_demand_data(weather_df)
        all_city_demands.append(city_demand_df)
        
    full_dataset = pd.concat(all_city_demands, ignore_index=True)
    
    # 3. Save raw generated data
    output_path = os.path.join("data", "raw_demand_data.csv")
    full_dataset.to_csv(output_path, index=False)
    print(f"Data ingestion completed! Saved dataset to {output_path} with shape {full_dataset.shape}")


if __name__ == "__main__":
    main()

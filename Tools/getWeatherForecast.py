import json
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import os
import requests

def load_weather_models():
    """
    Load all multi-output weather forecasting models from disk.
    Returns a dictionary of model_name: model_object
    """
    models = {}
    model_paths = {
        "gradient_boosting": "../models/Weather_Forecasting/GradientBoosting_multioutput.pkl",
        "ada_boost": "../models/Weather_Forecasting/AdaBoost_multioutput.pkl"
    }
    print(f"[INFO] Current working directory: {os.getcwd()} ")
    for model_name, path in model_paths.items():
        try:
            with open(path, 'rb') as file:
                models[model_name] = pickle.load(file)
                print(f"[INFO] Loaded {model_name} model.")
        except Exception as e:
            print(f"[ERROR] Failed to load {model_name} model: {e}")
    return models


def get_weather_defaults(year, month, day, latitude, longitude):
    try:
        date_str = f"{year}-{month:02d}-{day:02d}"
        url = (
            f"https://archive-api.open-meteo.com/v1/archive?"
            f"latitude={latitude}&longitude={longitude}"
            f"&start_date={date_str}&end_date={date_str}"
            "&daily=temperature_2m_max,temperature_2m_min,precipitation_sum,"
            "windspeed_10m_max,winddirection_10m_dominant,cloudcover_mean,"
            "humidity_2m_max,humidity_2m_min,pressure_msl_max,pressure_msl_min"
            "&timezone=UTC"
        )

        r = requests.get(url, timeout=10)  # timeout to avoid hanging
        r.raise_for_status()  # raise HTTPError if bad status
        data = r.json()

        if "daily" not in data:
            return {}

        daily = data["daily"]

        defaults = {
            "wind_degree": daily.get("winddirection_10m_dominant", [0])[0],
            "cloud": daily.get("cloudcover_mean", [0])[0],
            "humidity_lag1": np.mean([
                daily.get("humidity_2m_max", [0])[0],
                daily.get("humidity_2m_min", [0])[0]
            ]),
            "wind_kph_lag1": daily.get("windspeed_10m_max", [0])[0] * 1.60934,
            "pressure_mb_lag1": np.mean([
                daily.get("pressure_msl_max", [0])[0],
                daily.get("pressure_msl_min", [0])[0]
            ]),
            "precip_mm_lag1": daily.get("precipitation_sum", [0])[0],
            "temperature_celsius_lag1": np.mean([
                daily.get("temperature_2m_max", [0])[0],
                daily.get("temperature_2m_min", [0])[0]
            ])
        }

        return defaults

    except (requests.RequestException, ValueError, KeyError, IndexError) as e:
        print(f"[Weather Defaults] API fetch failed: {e}")
        return {}

def prepare_weather_input(year, month, day, latitude, longitude, scaler_path=None):
    """
    Prepare input features for weather forecasting model.
    Only year, month, day, latitude, longitude are required as input.
    All other features are set to zero (or NaN if preferred).
    Returns: DataFrame with all required columns, ready for model.
    """
    feature_cols = [
        'wind_degree', 'cloud', 'visibility_km', 'visibility_miles', 'uv_index',
        'gust_mph', 'gust_kph', 'air_quality_Carbon_Monoxide',
        'air_quality_Ozone', 'air_quality_Nitrogen_dioxide',
        'air_quality_Sulphur_dioxide', 'air_quality_PM2.5', 'air_quality_PM10',
        'air_quality_us-epa-index', 'air_quality_gb-defra-index', 'latitude',
        'longitude', 'year', 'month', 'day', 'dayofweek', 'dayofyear',
        'temperature_celsius_lag1', 'temperature_celsius_lag2',
        'temperature_celsius_lag3', 'temperature_celsius_lag7', 'wind_kph_lag1',
        'wind_kph_lag2', 'wind_kph_lag3', 'wind_kph_lag7', 'humidity_lag1',
        'humidity_lag2', 'humidity_lag3', 'humidity_lag7', 'pressure_mb_lag1',
        'pressure_mb_lag2', 'pressure_mb_lag3', 'pressure_mb_lag7',
        'precip_mm_lag1', 'precip_mm_lag2', 'precip_mm_lag3', 'precip_mm_lag7',
        'temperature_celsius_rollmean3', 'temperature_celsius_rollstd3',
        'temperature_celsius_rollmean7', 'temperature_celsius_rollstd7',
        'wind_kph_rollmean3', 'wind_kph_rollstd3', 'wind_kph_rollmean7',
        'wind_kph_rollstd7', 'humidity_rollmean3', 'humidity_rollstd3',
        'humidity_rollmean7', 'humidity_rollstd7', 'pressure_mb_rollmean3',
        'pressure_mb_rollstd3', 'pressure_mb_rollmean7', 'pressure_mb_rollstd7',
        'precip_mm_rollmean3', 'precip_mm_rollstd3', 'precip_mm_rollmean7',
        'precip_mm_rollstd7'
    ]
    # Set all features to zero
    input_dict = {col: 0.0 for col in feature_cols}

    defaults = get_weather_defaults(year, month, day, latitude, longitude)
    print(defaults)
    for k, v in defaults.items():
        if k in input_dict:
            input_dict[k] = v

    # Set provided values
    input_dict['year'] = year
    input_dict['month'] = month
    input_dict['day'] = day
    input_dict['latitude'] = latitude
    input_dict['longitude'] = longitude
    # Optionally, set dayofweek and dayofyear
    import datetime as dt
    try:
        d = dt.date(int(year), int(month), int(day))
        input_dict['dayofweek'] = d.weekday()
        input_dict['dayofyear'] = d.timetuple().tm_yday
    except Exception:
        input_dict['dayofweek'] = 0
        input_dict['dayofyear'] = 1
    input_df = pd.DataFrame([input_dict])
    if scaler_path is not None and os.path.exists(scaler_path):
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        X_scaled = scaler.transform(input_df.values)
        return X_scaled, feature_cols
    return input_df.values, feature_cols

def weather_forecast_inference(year, month, day, latitude, longitude, model_type='gradient_boosting', scaler_path=None):
    """
    Perform multi-output weather forecasting inference.
    Only year, month, day, latitude, longitude are required as input.
    Returns: JSON string with predictions and metadata
    """
    try:
        models = load_weather_models()
        if model_type not in models:
            raise ValueError(f"Model type '{model_type}' not available. Available: {list(models.keys())}")
        model = models[model_type]
        X_input, feature_cols = prepare_weather_input(year, month, day, latitude, longitude, scaler_path)
        preds = model.predict(X_input)
        # If single row, flatten
        if preds.shape[0] == 1:
            preds = preds[0]
        # Target order (should match training)
        targets = ['temperature_celsius', 'wind_kph', 'humidity', 'pressure_mb', 'precip_mm']
        # Prepare result
        result = {
            "status": "success",
            "model_used": model_type,
            "input_features": {
                "year": year,
                "month": month,
                "day": day,
                "latitude": latitude,
                "longitude": longitude
            },
            "predictions": {},
            "metadata": {
                "prediction_timestamp": datetime.now().isoformat(),
                "model_version": "v1.0",
                "data_source": "Historical weather data"
            }
        }
        if preds.ndim == 1:
            result["predictions"] = {t: float(p) for t, p in zip(targets, preds)}
        else:
            result["predictions"] = [
                {t: float(p) for t, p in zip(targets, row)} for row in preds
            ]
        return json.dumps(result, indent=2)
    except Exception as e:
        error_result = {
            "status": "error",
            "error_message": str(e),
            "input_features": {
                "year": year,
                "month": month,
                "day": day,
                "latitude": latitude,
                "longitude": longitude
            },
            "error_timestamp": datetime.now().isoformat()
        }
        return json.dumps(error_result, indent=2)

# Example usage:
if __name__ == "__main__":
    # Major Indian cities with coordinates
    indian_cities = [
        ("Delhi", 28.6139, 77.2090),
        ("Mumbai", 19.0760, 72.8777),
        ("Kolkata", 22.5726, 88.3639),
        ("Chennai", 13.0827, 80.2707),
        ("Bengaluru", 12.9716, 77.5946),
        ("Hyderabad", 17.3850, 78.4867),
        ("Ahmedabad", 23.0225, 72.5714),
        ("Pune", 18.5204, 73.8567),
        ("Jaipur", 26.9124, 75.7873),
        ("Patna", 25.5941, 85.1376),
    ]

    # Test both past and future dates
    test_dates = [
        (2024, 12, 25),  # past date
        (2025, 8, 11),   # today/future date
    ]

    for city, lat, lon in indian_cities:
        for year, month, day in test_dates:
            print(f"\n📍 {city} — {year}-{month:02d}-{day:02d}")
            result = weather_forecast_inference(
                year=year,
                month=month,
                day=day,
                latitude=lat,
                longitude=lon,
                model_type='ada_boost',
                scaler_path=None
            )
            print("Prediction:", result)


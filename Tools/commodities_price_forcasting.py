import os
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
import yfinance as yf
from datetime import timedelta


# -------- Load Model and Scaler --------
def load_forecasting_model(path: str):
    return tf.keras.models.load_model(path)

def load_scaler(path: str):
    return joblib.load(path)


# -------- Forecast Function --------
def forecast_future(model, recent_data, scaler, context_length, prediction_length, num_features, last_date=None):
    input_seq = recent_data[-context_length:].reshape(1, context_length, num_features)
    pred = model.predict(input_seq, verbose=0)
    pred_rescaled = scaler.inverse_transform(pred[0])

    df_pred = pd.DataFrame(pred_rescaled, columns=["Corn", "Soybeans", "Wheat", "DE", "ADM", "CTVA"])
    if last_date is not None:
        future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=prediction_length, freq="D")
        df_pred.insert(0, "Date", future_dates)

    return df_pred


# -------- Data Fetch Function --------
def fetch_data():
    commodity_tickers = ["ZC=F", "ZS=F", "KE=F"]
    stock_tickers = ["DE", "ADM", "CTVA"]

    data_commodities = yf.download(commodity_tickers, start="2020-01-01")["Close"]
    data_stocks = yf.download(stock_tickers, start="2020-01-01")["Close"]

    data = pd.concat([data_commodities, data_stocks], axis=1)
    data.columns = ["Corn", "Soybeans", "Wheat", "DE", "ADM", "CTVA"]
    data = data.fillna(method="ffill").dropna()
    return data


# -------- Main Tool Function --------
def predict_future_days(prediction_days: int) -> pd.DataFrame:
    """
    Predict next `prediction_days` days using the trained hybrid LSTM-GRU model.
    """
    # Fixed configs
    save_dir = "../models/Commodities_price_Forcasting/"
    model_path = os.path.join(save_dir, "hybrid_lstm_gru_tf.keras")
    scaler_path = os.path.join(save_dir, "scaler.pkl")
    context_length = 90

    # Load model + scaler
    model = load_forecasting_model(model_path)
    scaler = load_scaler(scaler_path)

    # Get data
    df = fetch_data()
    values = df.values
    data_scaled = scaler.transform(values)
    num_features = data_scaled.shape[1]
    last_date = df.index[-1]

    # Forecast
    forecast_df = forecast_future(
        model=model,
        recent_data=data_scaled,
        scaler=scaler,
        context_length=context_length,
        prediction_length=prediction_days,
        num_features=num_features,
        last_date=last_date,
    )
    return forecast_df


# -------- Example Usage --------
if __name__ == "__main__":
    forecast_df = predict_future_days(prediction_days=30)
    print(forecast_df)

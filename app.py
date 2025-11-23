import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple
import yfinance as yf

import joblib
from tensorflow.keras.models import load_model

LOOKBACK = 10
MODEL_DIR = Path("models")
MODEL_FILENAME = "eth_cnn_gru_saved.keras"
SCALER_FILENAME = "minmax_scaler.joblib"
LABELS_FILENAME = "original_labels.joblib"
LIVE_PERIOD = "400d"
LIVE_INTERVAL = "1d"

FEATURE_DROP_COLUMNS = [
    "Close",
    "High",
    "Low",
    "Open",
    "Volume",
    "atr",
    "hull_sma",
    "label",
    "hull_sma_gradient",
    "trend_label",
    "Train_label",
]

CLASS_DESCRIPTIONS = [
    "Positive inflection (potential short-term upside).",
    "Negative inflection (potential pullback).",
    "Uptrend continuation signal.",
    "Downtrend continuation signal.",
]

TREND_LABEL_MAP = {
    1: "Positive inflection",
    2: "Negative inflection",
    3: "Uptrend",
    4: "Downtrend",
    0: "No change",
}


@st.cache_resource(show_spinner=False)
def load_artifacts():
    """Load model, scaler, and original labels once per session."""
    model_path = MODEL_DIR / MODEL_FILENAME
    scaler_path = MODEL_DIR / SCALER_FILENAME
    labels_path = MODEL_DIR / LABELS_FILENAME

    if not model_path.exists():
        st.error(f"Missing model file at {model_path}.")
        st.stop()
    if not scaler_path.exists():
        st.error(f"Missing scaler file at {scaler_path}.")
        st.stop()
    if not labels_path.exists():
        st.error(f"Missing label file at {labels_path}.")
        st.stop()

    model = load_model(model_path)
    scaler = joblib.load(scaler_path)
    labels = joblib.load(labels_path)
    return model, scaler, labels


def initialize_price_dataframe():
    live_df, current_price = fetch_latest_dataframe()
    st.session_state.price_df = live_df
    st.session_state.data_source = f"Live yfinance ({LIVE_PERIOD})"
    st.session_state.current_price = current_price
    st.session_state.last_refresh = pd.Timestamp.utcnow()


def fetch_latest_dataframe(period: str = LIVE_PERIOD, interval: str = LIVE_INTERVAL) -> Tuple[pd.DataFrame, float]:
    if fetch_eth_history is None:
        raise RuntimeError("get_eth_prices.fetch_eth_history is unavailable. Ensure the module is on PYTHONPATH.")
    live_df, current_price = fetch_eth_history(period=period, interval=interval)
    live_df.index.name = "Date"
    live_df.sort_index(inplace=True)
    live_df = live_df.astype(float)
    return live_df, float(current_price)

def fetch_eth_history(period: str = "30d", interval: str = "1d") -> Tuple[pd.DataFrame, float]:
    """
    Returns a DataFrame with columns: Price, Close, High, Low, Open, Volume
    and the current market price (float).
    - period: e.g. "1d", "5d", "30d", "1mo", "1y"
    - interval: e.g. "1m", "5m", "1h", "1d"
    """
    ticker = yf.Ticker("ETH-USD")
    hist = ticker.history(period=period, interval=interval)

    if hist.empty:
        raise RuntimeError(f"No historical data returned for period={period}, interval={interval}")

    # Try to get a live/current price; fall back to the latest Close in history
    current_price = None
    try:
        info = ticker.info or {}
        current_price = info.get("regularMarketPrice")
    except Exception:
        current_price = None

    if current_price is None:
        # Sometimes info is unavailable; use latest Close
        current_price = float(hist["Close"].iloc[-1])

    # Prepare DataFrame with requested columns
    df = hist[["Open", "High", "Low", "Close", "Volume"]].copy()

    # Reorder columns to: Price, Close, High, Low, Open, Volume
    df = df[["Close", "High", "Low", "Open", "Volume"]]

    # Ensure index is timezone-aware string if desired (optional)
    df.index = df.index.tz_convert(None) if hasattr(df.index, "tz") else df.index

    return df, current_price


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Re-create the engineered columns that were used for training."""
    engineered = df.copy()
    engineered["Log_Close"] = np.log(engineered["Close"] / engineered["Close"].shift(1))
    engineered["Log_High"] = np.log(engineered["High"] / engineered["High"].shift(1))
    engineered["Log_Low"] = np.log(engineered["Low"] / engineered["Low"].shift(1))
    engineered["Log_Open"] = np.log(engineered["Open"] / engineered["Open"].shift(1))
    engineered["Log_Volume"] = np.log(engineered["Volume"] / engineered["Volume"].shift(1))
    engineered["Body"] = engineered["Close"] - engineered["Open"]
    engineered["Range"] = engineered["High"] - engineered["Low"]
    engineered["Upper_Wick"] = engineered["High"] - np.maximum(engineered["Close"], engineered["Open"])
    engineered["Lower_Wick"] = np.minimum(engineered["Close"], engineered["Open"]) - engineered["Low"]
    engineered.replace([np.inf, -np.inf], np.nan, inplace=True)
    return engineered


def prepare_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    engineered = engineer_features(df)
    features = engineered.iloc[1:].drop(columns=FEATURE_DROP_COLUMNS, errors="ignore")
    features = features.replace([np.inf, -np.inf], np.nan).dropna()
    return features


def get_latest_prediction(
    model, scaler, labels, feature_df: pd.DataFrame
) -> Optional[Dict[str, object]]:
    if feature_df.empty:
        return None

    scaled = pd.DataFrame(
        scaler.transform(feature_df),
        columns=feature_df.columns,
        index=feature_df.index,
    )

    if len(scaled) < LOOKBACK:
        return None

    latest_window = scaled.iloc[-LOOKBACK:].to_numpy(dtype=np.float32)
    latest_window = latest_window[np.newaxis, ...]
    probs = model.predict(latest_window, verbose=0)[0]
    pred_idx = int(np.argmax(probs))

    label_value = labels[pred_idx] if pred_idx < len(labels) else pred_idx
    description = (
        CLASS_DESCRIPTIONS[pred_idx]
        if pred_idx < len(CLASS_DESCRIPTIONS)
        else "No description available."
    )

    prediction_date = feature_df.index[-1]
    if isinstance(prediction_date, pd.Timestamp):
        prediction_date = prediction_date + pd.Timedelta(days=1)

    return {
        "pred_idx": pred_idx,
        "label": label_value,
        "description": description,
        "date": prediction_date,
    }


def render_price_chart(df: pd.DataFrame, days: int = 180):
    chart_df = df[["Close"]].copy()
    if isinstance(chart_df.index, pd.DatetimeIndex):
        start_date = chart_df.index.max() - pd.Timedelta(days=days)
        chart_df = chart_df.loc[chart_df.index >= start_date]
    chart_df = chart_df.rename(columns={"Close": "ETH Close (USD)"})
    st.line_chart(chart_df)


def main():
    st.set_page_config(page_title="ETH CNN-GRU Monitor", layout="wide")
    st.title("Ethereum CNN + GRU Signal Monitor")
    st.caption("Streaming the latest ETH close and model signal.")

    if "price_df" not in st.session_state:
        with st.spinner("Fetching latest ETH candles…"):
            try:
                initialize_price_dataframe()
            except Exception as exc:
                st.error(f"Unable to load live data: {exc}")
                st.stop()

    refresh_col, info_col = st.columns([1, 3])
    with refresh_col:
        if st.button("Refresh latest data", type="primary"):
            with st.spinner("Pulling fresh ETH candles…"):
                try:
                    live_df, current_price = fetch_latest_dataframe()
                except Exception as exc:  # pragma: no cover
                    st.error(f"Unable to refresh data: {exc}")
                else:
                    st.session_state.price_df = live_df
                    st.session_state.data_source = f"Live yfinance ({LIVE_PERIOD})"
                    st.session_state.current_price = current_price
                    st.session_state.last_refresh = pd.Timestamp.utcnow()
                    st.toast("Price data refreshed", icon="✅")
                    st.rerun()

    with info_col:
        source = st.session_state.get("data_source", "Saved dataset")
        last_refresh = st.session_state.get("last_refresh")
        if last_refresh is None:
            st.info(f"Data source: {source}")
        else:
            st.info(f"Data source: {source} | Last refresh (UTC): {last_refresh:%Y-%m-%d %H:%M:%S}")

    model, scaler, labels = load_artifacts()
    price_df = st.session_state.price_df
    features = prepare_feature_frame(price_df)
    prediction_info = get_latest_prediction(model, scaler, labels, features)

    latest_date = price_df.index[-1]
    latest_close = float(price_df["Close"].iloc[-1])

    st.subheader("Latest Model Prediction")
    if prediction_info is None:
        st.info("Not enough clean data to form a lookback window yet.")
    else:
        trend_label = TREND_LABEL_MAP.get(
            prediction_info["label"], f"Label {prediction_info['label']}"
        )
        st.metric(
            label=f"Next-day signal for {prediction_info['date']:%Y-%m-%d}",
            value=f"{trend_label} (class {prediction_info['pred_idx']})",
        )
        st.write(prediction_info["description"])

    st.subheader("Recent ETH Close Price")
    render_price_chart(price_df)


if __name__ == "__main__":
    main()

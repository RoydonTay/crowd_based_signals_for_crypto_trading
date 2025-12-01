import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple
import yfinance as yf

import joblib
from tensorflow.keras.models import load_model

# Optional import for interactive charts
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

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
        st.error(f"🚫 Missing model file at `{model_path}`")
        st.stop()
    if not scaler_path.exists():
        st.error(f"🚫 Missing scaler file at `{scaler_path}`")
        st.stop()
    if not labels_path.exists():
        st.error(f"🚫 Missing label file at `{labels_path}`")
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
    """Render interactive candlestick chart with Plotly or fallback to basic chart"""
    chart_df = df.copy()
    if isinstance(chart_df.index, pd.DatetimeIndex):
        start_date = chart_df.index.max() - pd.Timedelta(days=days)
        chart_df = chart_df.loc[chart_df.index >= start_date]
    
    if PLOTLY_AVAILABLE:
        # Create interactive candlestick + volume chart
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            row_heights=[0.7, 0.3],
            subplot_titles=('<b>ETH Price (USD)</b>', '<b>Trading Volume</b>')
        )
        
        # Candlestick chart with custom colors
        fig.add_trace(
            go.Candlestick(
                x=chart_df.index,
                open=chart_df['Open'],
                high=chart_df['High'],
                low=chart_df['Low'],
                close=chart_df['Close'],
                name='ETH-USD',
                increasing_line_color='#00f2fe',
                decreasing_line_color='#fa709a',
                increasing_fillcolor='rgba(0, 242, 254, 0.8)',
                decreasing_fillcolor='rgba(250, 112, 154, 0.8)'
            ),
            row=1, col=1
        )
        
        # Volume bars with gradient colors
        colors = ['rgba(0, 242, 254, 0.6)' if chart_df['Close'].iloc[i] >= chart_df['Open'].iloc[i] 
                  else 'rgba(250, 112, 154, 0.6)' for i in range(len(chart_df))]
        
        fig.add_trace(
            go.Bar(
                x=chart_df.index, 
                y=chart_df['Volume'], 
                name='Volume',
                marker_color=colors,
                showlegend=False
            ),
            row=2, col=1
        )
        
        # Enhanced layout with better styling
        fig.update_layout(
            height=650,
            xaxis_rangeslider_visible=False,
            hovermode='x unified',
            template='plotly_dark',
            showlegend=False,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(17,17,17,0.4)',
            font=dict(
                family="Arial, sans-serif",
                size=12,
                color='#ffffff'
            ),
            margin=dict(l=10, r=10, t=40, b=10)
        )
        
        # Update axes styling
        fig.update_xaxes(
            title_text="<b>Date</b>", 
            row=2, col=1,
            gridcolor='rgba(128, 128, 128, 0.2)',
            showgrid=True
        )
        fig.update_yaxes(
            title_text="<b>Price (USD)</b>", 
            row=1, col=1,
            gridcolor='rgba(128, 128, 128, 0.2)',
            showgrid=True
        )
        fig.update_yaxes(
            title_text="<b>Volume</b>", 
            row=2, col=1,
            gridcolor='rgba(128, 128, 128, 0.2)',
            showgrid=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        # Fallback to basic line chart
        simple_chart = chart_df[["Close"]].copy()
        simple_chart = simple_chart.rename(columns={"Close": "💰 ETH Close (USD)"})
        st.line_chart(simple_chart, height=500)


def main():
    st.set_page_config(
        page_title="ETH CNN-GRU Monitor", 
        layout="wide",
        page_icon="📈"
    )
    
    # Custom CSS for better styling
    st.markdown("""
        <style>
        /* Main title styling */
        .main-title {
            color: #ffffff;
            font-size: 2.5rem;
            font-weight: 700;
            margin-bottom: 0.5rem;
            letter-spacing: -0.5px;
        }
        
        /* Card styling */
        .prediction-card {
            background: #667eea;
            padding: 2rem;
            border-radius: 15px;
            box-shadow: 0 8px 32px rgba(102, 126, 234, 0.3);
            margin: 1.5rem 0;
            color: white;
        }
        
        .signal-positive {
            background: #11998e;
        }
        
        .signal-negative {
            background: #eb3349;
        }
        
        .signal-uptrend {
            background: #4facfe;
        }
        
        .signal-downtrend {
            background: #fa709a;
        }
        
        /* Info box styling */
        .info-box {
            background: rgba(102, 126, 234, 0.1);
            border-left: 4px solid #667eea;
            padding: 1rem 1.5rem;
            border-radius: 8px;
            margin: 1rem 0;
        }
        
        /* Price metric styling */
        .price-metric {
            text-align: center;
            padding: 1.5rem;
            background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
            border-radius: 12px;
            margin: 1rem 0;
        }
        
        .price-value {
            font-size: 2.5rem;
            font-weight: 700;
            color: #667eea;
        }
        
        .price-label {
            font-size: 1rem;
            color: #888;
            margin-bottom: 0.5rem;
        }
        
        /* Button styling */
        .stButton>button {
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 8px;
            padding: 0.5rem 2rem;
            font-weight: 600;
            transition: transform 0.2s;
        }
        
        .stButton>button:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 20px rgba(102, 126, 234, 0.4);
        }
        
        /* Section headers */
        .section-header {
            font-size: 1.8rem;
            font-weight: 700;
            margin: 2rem 0 1rem 0;
            color: #ffffff;
            border-bottom: 3px solid #667eea;
            padding-bottom: 0.5rem;
        }
        
        /* Emoji styling */
        .emoji-large {
            font-size: 3rem;
            margin-right: 1rem;
        }
        </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<h1 class="main-title"> Ethereum CNN + GRU Signal Monitor</h1>', unsafe_allow_html=True)
    st.caption("🔄 Real-time ETH price tracking with AI-powered trend predictions")

    if "price_df" not in st.session_state:
        with st.spinner("🔄 Fetching latest ETH data from yfinance..."):
            try:
                initialize_price_dataframe()
            except Exception as exc:
                st.error(f"❌ Unable to load live data: {exc}")
                st.stop()

    # Refresh button and info section
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🔄 Refresh Latest Data", type="primary", use_container_width=True):
            with st.spinner("📡 Pulling fresh ETH candles…"):
                try:
                    live_df, current_price = fetch_latest_dataframe()
                except Exception as exc:  # pragma: no cover
                    st.error(f"❌ Unable to refresh data: {exc}")
                else:
                    st.session_state.price_df = live_df
                    st.session_state.data_source = f"Live yfinance ({LIVE_PERIOD})"
                    st.session_state.current_price = current_price
                    st.session_state.last_refresh = pd.Timestamp.utcnow()
                    st.toast("✅ Price data refreshed successfully!", icon="🎉")
                    st.rerun()
    
    # Data source info box
    source = st.session_state.get("data_source", "Saved dataset")
    last_refresh = st.session_state.get("last_refresh")
    if last_refresh is None:
        st.markdown(f'<div class="info-box">📊 <b>Data source:</b> {source}</div>', unsafe_allow_html=True)
    else:
        st.markdown(
            f'<div class="info-box">📊 <b>Data source:</b> {source} | ⏰ <b>Last refresh (UTC):</b> {last_refresh:%Y-%m-%d %H:%M:%S}</div>', 
            unsafe_allow_html=True
        )

    model, scaler, labels = load_artifacts()
    price_df = st.session_state.price_df
    features = prepare_feature_frame(price_df)
    prediction_info = get_latest_prediction(model, scaler, labels, features)

    latest_date = price_df.index[-1]
    latest_close = float(price_df["Close"].iloc[-1])

    # Current price display
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown(f"""
            <div class="price-metric">
                <div class="price-label">💰 Current ETH Price</div>
                <div class="price-value">${latest_close:,.2f}</div>
                <div class="price-label">As of {latest_date:%Y-%m-%d}</div>
            </div>
        """, unsafe_allow_html=True)

    # Prediction section with enhanced styling
    st.markdown('<div class="section-header">AI Model Prediction</div>', unsafe_allow_html=True)
    
    if prediction_info is None:
        st.info("⏳ Not enough clean data to form a lookback window yet. Please refresh data.")
    else:
        trend_label = TREND_LABEL_MAP.get(
            prediction_info["label"], f"Label {prediction_info['label']}"
        )
        
        # Choose card style based on prediction
        signal_class = ""
        emoji = "📊"
        if prediction_info['pred_idx'] == 0:
            signal_class = "signal-positive"
            emoji = "📈"
        elif prediction_info['pred_idx'] == 1:
            signal_class = "signal-negative"
            emoji = "📉"
        elif prediction_info['pred_idx'] == 2:
            signal_class = "signal-uptrend"
            emoji = "🚀"
        elif prediction_info['pred_idx'] == 3:
            signal_class = "signal-downtrend"
            emoji = "⚠️"
        
        st.markdown(f"""
            <div class="prediction-card {signal_class}">
                <h3 style="margin: 0 0 0.75rem 0; color: white;">{emoji} {trend_label}</h3>
                <p style="margin: 0.5rem 0; color: rgba(255, 255, 255, 0.9);">📅 {prediction_info['date']:%Y-%m-%d}</p>
                <p style="margin: 0.5rem 0; color: rgba(255, 255, 255, 0.9);">{prediction_info['description']}</p>
            </div>
        """, unsafe_allow_html=True)

    # Price chart section
    st.markdown('<div class="section-header">Historical Price Chart</div>', unsafe_allow_html=True)
    if not PLOTLY_AVAILABLE:
        st.markdown("""
            <div class="info-box" style="background: rgba(255, 193, 7, 0.1); border-left-color: #ffc107;">
                💡 <b>Tip:</b> Install <code>plotly</code> for interactive candlestick charts with volume: <code>pip install plotly</code>
            </div>
        """, unsafe_allow_html=True)
    render_price_chart(price_df)
    
    st.markdown("---")


if __name__ == "__main__":
    main()

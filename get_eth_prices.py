import sys
from typing import Tuple
import yfinance as yf
import pandas as pd

"""
get_eth_prices.py

Fetch ETH/USD price data (Price, Close, High, Low, Open, Volume) using yfinance.

Install dependency:
    pip install yfinance pandas
"""



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


def main(argv):
    # Allow optional CLI args: period, interval
    period = argv[1] if len(argv) > 1 else "1825d"
    interval = argv[2] if len(argv) > 2 else "1d"

    df, current_price = fetch_eth_history(period=period, interval=interval)

    # # Print summary to stdout and save CSV
    # print(f"ETH-USD current price: {current_price}")
    # print(df.tail(10).to_string())

    out_csv = "eth_prices.csv"
    df.to_csv(out_csv)
    print(f"Saved {len(df)} rows to {out_csv}")


if __name__ == "__main__":
    main(sys.argv)
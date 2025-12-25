import yfinance as yf
import pandas as pd
import numpy as np
import os


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../tests"))
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

class YFDataLoader:
    """Helper class to download and process option chain data from Yahoo Finance using yfinance."""
    
    def __init__(self, relative_output_path: str):
        self.output_path = os.path.join(DATA_DIR, relative_output_path)
    
    def load_option_chain(self, ticker: str, min_bid: float) -> pd.DataFrame:
        tk = yf.Ticker(ticker)
        expiries = tk.options
        
        print(f"Found {len(expiries)} expiries for {ticker}")
        
        all_options = []
        for expiry in expiries:
            chain = tk.option_chain(expiry)
            for opt_type, df in [("call", chain.calls), ("put", chain.puts)]:
                df = df.copy()
                df["option_type"] = opt_type
                df["expiry"] = expiry
                df["mid"] = (df["bid"] + df["ask"]) / 2
                df = df[df["bid"] > min_bid]  # basic liquidity filter
                all_options.append(df)
        
        df = pd.concat(all_options, ignore_index=True)
        
        df = df[[
            "expiry", "strike", "option_type",
            "bid", "ask", "mid", "lastPrice",
            "volume", "openInterest"
        ]]
        
        df = df.dropna(subset=["mid"])
        df["expiry"] = pd.to_datetime(df["expiry"])
        today = pd.Timestamp.today().normalize()

        df["time_to_expiry"] = (df["expiry"] - today).dt.days / 365.0
        df = df[df["time_to_expiry"] > 0]  # drop already-expired

        df["days_to_expiry"] = (df["expiry"] - today).dt.days
        df.to_csv(self.output_path, index=False)
        print(f"Saved data → {self.output_path}")
        
    def get_spot(self, ticker):
        tk = yf.Ticker(ticker)
        hist = tk.history(period="5d")
        if len(hist) > 0:
            return float(hist["Close"].iloc[-1])
        return float(tk.fast_info.last_price)
    
    def _compute_forward(S, T, r, q):
        return S * np.exp((r - q) * T)
        
    def retrieve_data(self) -> pd.DataFrame:
        df = pd.read_csv(self.output_path)
        df["expiry"] = pd.to_datetime(df["expiry"])
        return df
    
    
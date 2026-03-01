import os
import pandas as pd
import numpy as np

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../tests"))
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)


class OptionDXDataLoader:
    """
    OptionsDX loader for 'wide' txt files with bracketed headers and call/put columns side-by-side.
    Produces:
      - load_wide(): wide dataframe
      - load_long(): long dataframe in the requested schema + forward via put-call parity
    """

    def __init__(self, relative_path: str):
        self.path = os.path.join(DATA_DIR, relative_path)

    @staticmethod
    def _parse_quote_date(series: pd.Series) -> pd.Series:
        """
        Handles QUOTE_DATE that may be:
          - int like 20230104
          - string like '2023-01-04'
        """
        s = series.copy()

        # If numeric, decide if it's YYYYMMDD (8 digits) or epoch (10/13 digits)
        if pd.api.types.is_numeric_dtype(s):
            s_int = s.astype("Int64")
            s_str = s_int.astype(str)

            lens = s_str.str.len().dropna()
            mode_len = int(lens.mode().iloc[0]) if len(lens) else 0

            if mode_len == 8:
                # YYYYMMDD
                return pd.to_datetime(s_str, format="%Y%m%d", errors="coerce")
            elif mode_len == 10:
                # unix seconds
                return pd.to_datetime(s_int, unit="s", utc=True, errors="coerce").dt.tz_convert(None).dt.normalize()
            elif mode_len == 13:
                # unix milliseconds
                return pd.to_datetime(s_int, unit="ms", utc=True, errors="coerce").dt.tz_convert(None).dt.normalize()

            # fallback
            return pd.to_datetime(s, errors="coerce")

        # Otherwise treat as string date
        s_str = s.astype(str).str.strip()
        # try ISO first, then YYYYMMDD
        out = pd.to_datetime(s_str, format="%Y-%m-%d", errors="coerce")
        if out.isna().mean() > 0.5:
            out = pd.to_datetime(s_str, format="%Y%m%d", errors="coerce")
        return out

    def load_wide(self) -> pd.DataFrame:
        df = pd.read_csv(self.path, sep=",", engine="python", skipinitialspace=True)

        # Strip square brackets from headers
        df.columns = [c.strip().strip("[]") for c in df.columns]

        # Parse QUOTE_READTIME (if present)
        if "QUOTE_READTIME" in df.columns:
            df["QUOTE_READTIME"] = pd.to_datetime(df["QUOTE_READTIME"], errors="coerce")

        # Parse QUOTE_DATE robustly (this is your problem column)
        if "QUOTE_DATE" in df.columns:
            df["QUOTE_DATE"] = self._parse_quote_date(df["QUOTE_DATE"])
        else:
            raise ValueError("Missing QUOTE_DATE column")

        # Parse EXPIRE_DATE (often string 'YYYY-MM-DD' or int YYYYMMDD)
        if "EXPIRE_DATE" in df.columns:
            df["EXPIRE_DATE"] = self._parse_quote_date(df["EXPIRE_DATE"])
        else:
            raise ValueError("Missing EXPIRE_DATE column")

        # Convert numerics (exclude SIZE columns like '7 x 7')
        numeric_cols = [
            c for c in df.columns
            if any(c.startswith(prefix) for prefix in ("C_", "P_", "STRIKE", "DTE", "QUOTE_", "UNDERLYING", "EXPIRE_"))
            and not c.endswith("SIZE")
            and c not in ("QUOTE_READTIME", "QUOTE_DATE", "EXPIRE_DATE")
        ]
        for c in numeric_cols:
            df[c] = pd.to_numeric(df[c], errors="coerce")

        # Strike is numeric even though it doesn't start with STRIKE_ sometimes
        if "STRIKE" in df.columns:
            df["STRIKE"] = pd.to_numeric(df["STRIKE"], errors="coerce")

        return df

    def load_long(
        self,
        *,
        min_bid: float = 0.01,
        drop_0dte: bool = True,
        forward_strike_window_pct: float = 0.20,
        forward_smooth_neighbors: int = 5,
    ) -> pd.DataFrame:
        wide = self.load_wide()

        # Compute time to expiry
        wide = wide.copy()
        wide["days_to_expiry"] = (wide["EXPIRE_DATE"] - wide["QUOTE_DATE"]).dt.days
        wide["time_to_expiry"] = wide["days_to_expiry"] / 365.0

        if drop_0dte:
            wide = wide[wide["days_to_expiry"] > 0].copy()

        forwards = self._compute_forward_per_expiry(
            wide,
            strike_window_pct=forward_strike_window_pct,
            smooth_neighbors=forward_smooth_neighbors,
        )

        # Build long calls/puts
        calls = pd.DataFrame({
            "quote_date": wide["QUOTE_DATE"],
            "expiry": wide["EXPIRE_DATE"],
            "strike": wide["STRIKE"],
            "option_type": "call",
            "bid": wide["C_BID"],
            "ask": wide["C_ASK"],
            "lastPrice": wide.get("C_LAST", np.nan),
            "volume": wide.get("C_VOLUME", np.nan),
            "time_to_expiry": wide["time_to_expiry"],
            "days_to_expiry": wide["days_to_expiry"],
        })
        calls["mid"] = (calls["bid"] + calls["ask"]) / 2.0

        puts = pd.DataFrame({
            "quote_date": wide["QUOTE_DATE"],
            "expiry": wide["EXPIRE_DATE"],
            "strike": wide["STRIKE"],
            "option_type": "put",
            "bid": wide["P_BID"],
            "ask": wide["P_ASK"],
            "lastPrice": wide.get("P_LAST", np.nan),
            "volume": wide.get("P_VOLUME", np.nan),
            "time_to_expiry": wide["time_to_expiry"],
            "days_to_expiry": wide["days_to_expiry"],
        })
        puts["mid"] = (puts["bid"] + puts["ask"]) / 2.0

        long_df = pd.concat([calls, puts], ignore_index=True)

        # Hygiene filters
        long_df = long_df.dropna(subset=["expiry", "strike", "bid", "ask"])
        long_df = long_df[(long_df["ask"] > long_df["bid"]) & (long_df["bid"] >= min_bid)].copy()

        # Merge forward
        out = long_df.merge(forwards, on=["quote_date", "expiry"], how="left")

        # Output schema requested (+ forward)
        out = out[[
            "quote_date","expiry", "strike", "option_type",
            "bid", "ask", "mid", "lastPrice",
            "volume","time_to_expiry", "days_to_expiry",
            "forward",
        ]].sort_values(["quote_date","expiry", "strike", "option_type"], ignore_index=True)

        return out

    @staticmethod
    def _compute_forward_per_expiry(wide: pd.DataFrame, *, strike_window_pct: float, smooth_neighbors: int) -> pd.DataFrame:
        d = wide.copy()

        d["C_mid"] = (d["C_BID"] + d["C_ASK"]) / 2.0
        d["P_mid"] = (d["P_BID"] + d["P_ASK"]) / 2.0

        d["strike_dist_pct"] = (d["STRIKE"] - d["UNDERLYING_LAST"]).abs() / d["UNDERLYING_LAST"].replace(0, np.nan)

        good = (
            d["C_mid"].notna() & d["P_mid"].notna() &
            (d["C_BID"] > 0) & (d["P_BID"] > 0) &
            (d["C_ASK"] > d["C_BID"]) & (d["P_ASK"] > d["P_BID"]) &
            (d["strike_dist_pct"] <= strike_window_pct)
        )
        d = d.loc[good].copy()
        if d.empty:
            return wide[["QUOTE_DATE", "EXPIRE_DATE"]].drop_duplicates().rename(
                columns={"QUOTE_DATE": "quote_date", "EXPIRE_DATE": "expiry"}
            ).assign(forward=np.nan)

        d["F_candidate"] = d["STRIKE"] + (d["C_mid"] - d["P_mid"])
        d["abs_cp_diff"] = (d["C_mid"] - d["P_mid"]).abs()

        key = ["QUOTE_DATE", "EXPIRE_DATE"]

        best = (
            d.sort_values(key + ["abs_cp_diff"])
             .groupby(key, as_index=False)
             .first()[key + ["STRIKE"]]
             .rename(columns={"STRIKE": "best_strike"})
        )

        merged = d.merge(best, on=key, how="inner")
        merged["dist_from_best"] = (merged["STRIKE"] - merged["best_strike"]).abs()

        neighbors = (
            merged.sort_values(key + ["dist_from_best"])
                  .groupby(key, as_index=False)
                  .head(max(1, int(smooth_neighbors)))
        )

        forwards = (
            neighbors.groupby(key, as_index=False)["F_candidate"]
                     .mean()
                     .rename(columns={"F_candidate": "forward"})
        )

        return forwards.rename(columns={"QUOTE_DATE": "quote_date", "EXPIRE_DATE": "expiry"})
    
    def retrieve_data(self) -> pd.DataFrame:
        df = pd.read_csv(self.path)
        df["expiry"] = pd.to_datetime(df["expiry"])
        return df


if __name__ == "__main__":
    loader = OptionDXDataLoader("spx_eod_202301.txt")

    df_wide = loader.load_wide()
    print("WIDE:")
    print(df_wide.head())
    print(df_wide.dtypes)

    df_long = loader.load_long(
        min_bid=0.10,
        drop_0dte=True,
        forward_strike_window_pct=0.20,
        forward_smooth_neighbors=5,
    )

    print(df_long.head(10))
    print(df_long.dtypes)

    out_path = os.path.join(DATA_DIR, "spx_eod_202301.csv")
    df_long.to_csv(out_path, index=False)
    print(f"\nSaved → {out_path}")
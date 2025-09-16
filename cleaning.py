import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


class Cleaner:
    def __init__(self, normalize: bool = True, iqr_mult: float = 1.5):
        self.normalize = normalize
        self.iqr_mult = iqr_mult
        self.scaler = None

    def cap_outliers_iqr(self, df: pd.DataFrame, cols):
        capped = df.copy()
        for c in cols:
            q1, q3 = np.percentile(capped[c].dropna(), [25, 75])
            iqr = q3 - q1
            lo = q1 - self.iqr_mult * iqr
            hi = q3 + self.iqr_mult * iqr
            capped[c] = capped[c].clip(lo, hi)
        return capped

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        use_cols = [c for c in ["distance_km",
                                "travel_time_est", "fuel_rate"] if c in df.columns]
        out = self.cap_outliers_iqr(df, use_cols)
        if self.normalize and use_cols:
            self.scaler = MinMaxScaler()
            out[use_cols] = self.scaler.fit_transform(out[use_cols])
        return out

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        use_cols = [c for c in ["distance_km",
                                "travel_time_est", "fuel_rate"] if c in df.columns]
        if self.scaler is not None and use_cols:
            out[use_cols] = self.scaler.transform(out[use_cols])
        return out

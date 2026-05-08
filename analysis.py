import numpy as np
import pandas as pd
from scipy import stats as _sp_stats

OPERATIONS = {
    "None (raw values)":           "none",
    "Subtract control mean":       "subtract",
    "Divide by control mean":      "divide",
    "Multiply by control mean":    "multiply",
    "Add control mean":            "add",
    "Percent change from control": "pct_change",
    "Log2 ratio vs control":       "log2_ratio",
    "Min-max (control range)":     "minmax",
}

OPERATION_Y_LABEL = {
    "none":       "{param}",
    "subtract":   "{param} − control mean",
    "divide":     "{param} / control mean",
    "multiply":   "{param} × control mean",
    "add":        "{param} + control mean",
    "pct_change": "% change from control ({param})",
    "log2_ratio": "log₂({param} / control mean)",
    "minmax":     "Min-max normalized {param}",
}


OUTLIER_METHODS = {
    "Z-score":                   "zscore",
    "IQR (interquartile range)": "iqr",
    "Percentile cutoff":         "percentile",
    "ROUT":                      "rout",
}

ROUT_Q_OPTIONS = {
    "0.1%  (most stringent)": 0.001,
    "1%    (default / Prism)": 0.01,
    "2%":                      0.02,
    "5%":                      0.05,
    "10%   (most lenient)":    0.10,
}


def _rout_single(values: np.ndarray, Q: float) -> np.ndarray:
    """
    ROUT outlier detection on a 1-D array (Motulsky & Brown 2006).

    Iteratively tests the most extreme robust residual using a Grubbs-like
    statistic computed from the median and scaled MAD. A point is flagged as
    an outlier when its Grubbs p-value < Q, where Q is the maximum FDR.
    Stops when no candidate exceeds the threshold or fewer than 4 points remain.

    Returns a boolean array (True = outlier).
    """
    n_orig = len(values)
    outlier_mask = np.zeros(n_orig, dtype=bool)
    active = np.arange(n_orig)

    while len(active) >= 4:
        n = len(active)
        vals = values[active]

        median = np.median(vals)
        mad = np.median(np.abs(vals - median))
        if mad == 0:
            break

        s = 1.4826 * mad          # scale factor: makes MAD consistent with σ under normality
        residuals = np.abs(vals - median) / s
        i_max = int(np.argmax(residuals))
        G = residuals[i_max]

        # Invert the Grubbs statistic to recover the underlying t-value
        denom = (n - 1) ** 2 - G ** 2 * n
        if denom <= 0:
            p = 0.0
        else:
            t_sq = G ** 2 * n * (n - 2) / denom
            if t_sq <= 0:
                break
            t_stat = float(np.sqrt(t_sq))
            # Multiply by 2·n to account for testing all n points (Bonferroni-like,
            # consistent with the FDR interpretation used by Prism's ROUT)
            p = float(2 * n * _sp_stats.t.sf(t_stat, df=n - 2))

        if p < Q:
            outlier_mask[active[i_max]] = True
            active = np.delete(active, i_max)
        else:
            break

    return outlier_mask


def remove_outliers(
    df: pd.DataFrame,
    param: str,
    wells: list,
    method: str,
    **kwargs,
) -> pd.DataFrame:
    """Remove outliers from `param` within each well independently. Returns filtered copy."""
    keep = pd.Series(True, index=df.index)

    for well in wells:
        mask = df['Source well'] == well
        vals = df.loc[mask, param].dropna()
        if len(vals) < 4:
            continue

        if method == 'zscore':
            thr = kwargs.get('threshold', 3.0)
            std = vals.std(ddof=1)
            if std == 0:
                continue
            z = (df.loc[mask, param] - vals.mean()) / std
            keep.loc[mask] = keep.loc[mask] & (z.abs() <= thr)

        elif method == 'iqr':
            k = kwargs.get('k', 1.5)
            q1, q3 = vals.quantile(0.25), vals.quantile(0.75)
            iqr = q3 - q1
            lo, hi = q1 - k * iqr, q3 + k * iqr
            keep.loc[mask] = keep.loc[mask] & df.loc[mask, param].between(lo, hi)

        elif method == 'percentile':
            lo_pct = kwargs.get('lo_pct', 2.5)
            hi_pct = kwargs.get('hi_pct', 97.5)
            lo = vals.quantile(lo_pct / 100)
            hi = vals.quantile(hi_pct / 100)
            keep.loc[mask] = keep.loc[mask] & df.loc[mask, param].between(lo, hi)

        elif method == 'rout':
            Q = kwargs.get('Q', 0.01)
            arr = df.loc[mask, param].to_numpy(dtype=float)
            valid = ~np.isnan(arr)
            outlier_flags = np.zeros(len(arr), dtype=bool)
            if valid.sum() >= 4:
                sub_outliers = _rout_single(arr[valid], Q)
                outlier_flags[valid] = sub_outliers
            # Map flags back to the DataFrame index
            well_idx = df.index[mask]
            keep.loc[well_idx] = keep.loc[well_idx] & ~pd.Series(outlier_flags, index=well_idx)

    return df[keep].copy()


def outlier_removal_summary(
    df_before: pd.DataFrame,
    df_after: pd.DataFrame,
    wells: list,
) -> pd.DataFrame:
    rows = []
    for well in wells:
        n_before = int((df_before['Source well'] == well).sum())
        n_after  = int((df_after['Source well']  == well).sum())
        n_removed = n_before - n_after
        pct = round(n_removed / n_before * 100, 1) if n_before > 0 else 0.0
        rows.append({
            'well': well,
            'n_before': n_before,
            'n_after':  n_after,
            'n_removed': n_removed,
            '% removed': pct,
        })
    return pd.DataFrame(rows)


def get_well_mean(df: pd.DataFrame, well: str, param: str) -> float:
    vals = df.loc[df['Source well'] == well, param].dropna()
    return float(vals.mean()) if len(vals) > 0 else np.nan


def get_control_stats(df: pd.DataFrame, control_wells: 'str | list[str]', param: str) -> dict:
    if isinstance(control_wells, str):
        control_wells = [control_wells]
    vals = df.loc[df['Source well'].isin(control_wells), param].dropna()
    if vals.empty:
        return {"mean": np.nan, "min": np.nan, "max": np.nan}
    return {"mean": float(vals.mean()), "min": float(vals.min()), "max": float(vals.max())}


def apply_normalization(
    series: pd.Series,
    ctrl_stats: dict,
    operation: str,
    minmax_config: dict | None = None,
) -> tuple:
    mean = ctrl_stats["mean"]
    ctrl_min = ctrl_stats["min"]
    ctrl_max = ctrl_stats["max"]

    if operation == "none":
        return series.copy(), None

    if operation == "subtract":
        if np.isnan(mean):
            return pd.Series(np.nan, index=series.index), "Control well has no valid data."
        return series - mean, None

    if operation == "divide":
        if np.isnan(mean):
            return pd.Series(np.nan, index=series.index), "Control well has no valid data."
        if mean == 0:
            return pd.Series(np.nan, index=series.index), "Control mean is 0 — division not possible."
        return series / mean, None

    if operation == "multiply":
        if np.isnan(mean):
            return pd.Series(np.nan, index=series.index), "Control well has no valid data."
        return series * mean, None

    if operation == "add":
        if np.isnan(mean):
            return pd.Series(np.nan, index=series.index), "Control well has no valid data."
        return series + mean, None

    if operation == "pct_change":
        if np.isnan(mean):
            return pd.Series(np.nan, index=series.index), "Control well has no valid data."
        if mean == 0:
            return pd.Series(np.nan, index=series.index), "Control mean is 0 — percent change not possible."
        return (series - mean) / mean * 100, None

    if operation == "log2_ratio":
        if np.isnan(mean):
            return pd.Series(np.nan, index=series.index), "Control well has no valid data."
        if mean == 0:
            return pd.Series(np.nan, index=series.index), "Control mean is 0 — log2 ratio not possible."
        with np.errstate(divide='ignore', invalid='ignore'):
            result = np.log2(series.to_numpy(dtype=float) / mean)
        return pd.Series(result, index=series.index), None

    if operation == "minmax":
        if minmax_config is not None:
            use_min = minmax_config["min_val"]
            use_max = minmax_config["max_val"]
        else:
            use_min = ctrl_min
            use_max = ctrl_max
        if np.isnan(use_min) or np.isnan(use_max):
            return pd.Series(np.nan, index=series.index), "Min or max well has no valid data."
        if use_max == use_min:
            return pd.Series(np.nan, index=series.index), "Min and max well means are equal — min-max normalization not possible."
        return (series - use_min) / (use_max - use_min), None

    raise ValueError(f"Unknown operation: {operation!r}")


def compute_well_stats(df: pd.DataFrame, param: str, wells: list) -> pd.DataFrame:
    rows = []
    for well in wells:
        vals = df.loc[df['Source well'] == well, param].dropna()
        n = len(vals)
        mean = float(vals.mean()) if n > 0 else np.nan
        ci95 = float(1.96 * vals.std(ddof=1) / np.sqrt(n)) if n >= 2 else np.nan
        rows.append({"well": well, "n": n, "mean": mean, "ci95": ci95})
    return pd.DataFrame(rows)


def build_wide_dataframe(df: pd.DataFrame, param: str, wells: list) -> pd.DataFrame:
    return pd.DataFrame({
        w: pd.Series(df.loc[df['Source well'] == w, param].to_numpy())
        for w in wells
    })

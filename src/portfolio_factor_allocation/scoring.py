import numpy as np
import pandas as pd


def _yearly_z_score(df, cols, date_col="date"):
    years = df[date_col]

    for c in cols:
        if c not in df.columns:
            raise KeyError(f"{c} not in DataFrame")

        def _z(s):
            std = s.std(ddof=0)
            if std == 0 or np.isnan(std):
                return pd.Series(0.0, index=s.index)
            return (s - s.mean()) / std

        df["z_" + c] = df.groupby(years)[c].transform(_z)
    return df


def _yearly_rank_score(df, cols, date_col="date"):
    years = df[date_col]

    for c in cols:
        if c not in df.columns:
            raise KeyError(f"{c} not in DataFrame")

        def _rank01(s):
            n = s.notna().sum()
            if n <= 1:
                return pd.Series(0.0, index=s.index)
            r = s.rank(method="average")
            return (r - 1) / (n - 1)

        df["rank_" + c] = df.groupby(years)[c].transform(_rank01)
    return df


# Function to create z- or rank-scored factors
def yearly_score(df, cols, method, date_col="date"):
    if date_col not in df.columns:
        raise KeyError(f"{date_col} not in DataFrame")

    df = df.copy()

    if method == "z":
        return _yearly_z_score(df, cols, date_col=date_col)
    elif method == "rank":
        return _yearly_rank_score(df, cols, date_col=date_col)
    else:
        raise ValueError("method must be 'z' or 'rank'")


# Function to create an average factor for the integrated approach
def append_avg_score(df, cols, method):
    if method == "z":
        prefix = "z_"
    elif method == "rank":
        prefix = "rank_"
    else:
        raise ValueError("method must be 'z' or 'rank'")

    df = df.copy()

    score_cols = [prefix + c for c in cols]
    for c in score_cols:
        if c not in df.columns:
            raise KeyError(f"{c} not in DataFrame")

    df[prefix + "int"] = df[score_cols].mean(axis=1)
    return df

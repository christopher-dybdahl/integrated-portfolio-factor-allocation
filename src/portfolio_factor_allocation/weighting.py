import cvxpy as cp
import numpy as np
import pandas as pd

from .covariance import get_covariance_matrix


def get_benchmark_weights(df, mkt_cap_col="market_cap"):
    total_cap = df[mkt_cap_col].sum()
    if total_cap == 0:
        return pd.Series(0.0, index=df.index)
    return df[mkt_cap_col] / total_cap


# Function to create TER and DEC portfolio weights for each factor provided
def percentile_portfolio_weights(
    df, cols, method, p, mkt_cap_col="market_cap", date_col="date"
):
    if method == "z":
        prefix = "z_"
    elif method == "rank":
        prefix = "rank_"
    else:
        raise ValueError("method must be 'z' or 'rank'")

    if mkt_cap_col not in df.columns:
        raise KeyError(f"{mkt_cap_col} not in DataFrame")

    if date_col not in df.columns:
        raise KeyError(f"{date_col} not in DataFrame")

    df = df.copy()

    years = df[date_col]

    df_weights = df[[date_col]].copy()

    for c in cols:
        score_col = prefix + c
        if score_col not in df.columns:
            raise KeyError(f"{score_col} not in DataFrame")

        # Calculate quantile threshold per year
        thresholds = df.groupby(years)[score_col].transform(lambda x: x.quantile(1 - p))

        # Identify companies in top percentile
        mask = (df[score_col] >= thresholds) & (df[score_col].notna())

        # Calculate total market cap of selected companies per year
        masked_mkt_cap = df[mkt_cap_col].where(mask, 0.0)
        yearly_total_cap = masked_mkt_cap.groupby(years).transform("sum")

        # Calculate weights
        weights = masked_mkt_cap / yearly_total_cap
        df_weights[f"weight_{score_col}"] = weights.fillna(0.0)

    return df_weights


# Function to create the BW weights for each factor provided
def bw_portfolio_weights(
    df,
    cols,
    method,
    n_subportfolios,
    high_multiplier,
    increment,
    multiple_power=1,
    mkt_cap_col="market_cap",
    date_col="date",
):
    if method == "z":
        prefix = "z_"
    elif method == "rank":
        prefix = "rank_"
    else:
        raise ValueError("method must be 'z' or 'rank'")

    if mkt_cap_col not in df.columns:
        raise KeyError(f"{mkt_cap_col} not in DataFrame")

    if date_col not in df.columns:
        raise KeyError(f"{date_col} not in DataFrame")

    df = df.copy()

    years = df[date_col]

    df_weights = df[[date_col]].copy()

    for c in cols:
        score_col = prefix + c
        if score_col not in df.columns:
            raise KeyError(f"{score_col} not in DataFrame")

        # Helper to calculate buckets safely
        def _get_buckets(x):
            try:
                return pd.qcut(x, n_subportfolios, labels=False, duplicates="drop")
            except ValueError:
                return pd.Series(np.nan, index=x.index)

        # Create temporary bucket column
        bucket_col = f"_temp_bucket_{c}"
        df[bucket_col] = df.groupby(years)[score_col].transform(_get_buckets)

        # Calculate total market cap per bucket per year
        bucket_caps = df.groupby([years, bucket_col])[mkt_cap_col].transform("sum")

        # Calculate number of buckets per year to ensure equal size partition
        max_bucket = df.groupby(years)[bucket_col].transform("max")
        n_buckets = max_bucket + 1

        # Base weights (equal size buckets)
        base_weights = (df[mkt_cap_col] / bucket_caps) * (1.0 / n_buckets)

        # Multipliers
        # high_multiplier for max_bucket, decreasing by increment
        multipliers = high_multiplier - (max_bucket - df[bucket_col]) * increment

        # Final weights
        df_weights[f"weight_{score_col}"] = (
            base_weights * (multipliers**multiple_power)
        ).fillna(0.0)

        # Cleanup
        df.drop(columns=[bucket_col], inplace=True)

    return df_weights


# Function to create the mixed portfolio weights
def factor_adjusted_weights(df, cols, factor_weights, method):
    if method == "z":
        prefix = "z_"
    elif method == "rank":
        prefix = "rank_"
    else:
        raise ValueError("method must be 'z' or 'rank'")

    if len(cols) != len(factor_weights):
        raise ValueError("cols and factor_weights must have the same length")

    weight_cols = [f"weight_{prefix}{c}" for c in cols]
    for c in weight_cols:
        if c not in df.columns:
            raise KeyError(f"{c} not in DataFrame")

    weighted_sum = 0
    for col, weight in zip(weight_cols, factor_weights):
        weighted_sum += df[col] * weight

    return weighted_sum


# Helper function for TE portfolio
def build_te_problem(Sigma, w_bench, te_target):
    """
    Build a tracking error constrained long only portfolio problem.

    Maximize s' w
    s.t. (w - w_bench)' Sigma (w - w_bench) <= te_target^2
         1' w = 1
         w >= 0

    Returns (prob, w_var, s_param).
    """
    Sigma = np.asarray(Sigma)
    w_bench = np.asarray(w_bench)
    n = len(w_bench)

    w = cp.Variable(n)
    s_param = cp.Parameter(n)

    te_quad = cp.quad_form(w - w_bench, Sigma)
    constraints = [
        cp.sum(w) == 1.0,
        w >= 0.0,
        te_quad <= te_target**2,
    ]

    objective = cp.Maximize(s_param @ w)
    prob = cp.Problem(objective, constraints)

    return prob, w, s_param


# Function to create the TE weights for each factor provided
def te_portfolio_weights(
    df,
    cols,
    method,
    tracking_error,
    returns_df,
    mkt_cap_col="market_cap",
    date_col="date",
):
    if method == "z":
        prefix = "z_"
    elif method == "rank":
        prefix = "rank_"
    else:
        raise ValueError("method must be 'z' or 'rank'")

    if mkt_cap_col not in df.columns:
        raise KeyError(f"{mkt_cap_col} not in DataFrame")
    if date_col not in df.columns:
        raise KeyError(f"{date_col} not in DataFrame")

    df = df.copy()
    years = np.sort(df[date_col].unique())

    # output frame
    df_weights = df[[date_col]].copy()
    for c in cols:
        df_weights[f"weight_{prefix}{c}"] = 0.0

    for year in years:
        year_mask = df[date_col] == year
        # ensure one row per PERMNO in this cross section
        df_year = df.loc[year_mask].drop_duplicates("PERMNO").copy()

        if df_year.empty:
            continue

        permnos = df_year["PERMNO"].values

        # covariance matrix for this year and universe
        Sigma = get_covariance_matrix(returns_df, year, permnos)
        # benchmark weights
        w_bench = get_benchmark_weights(df_year, mkt_cap_col).values

        # build cvxpy problem once for this year
        prob, w_var, s_param = build_te_problem(Sigma, w_bench, tracking_error)

        for c in cols:
            score_col = prefix + c
            if score_col not in df.columns:
                raise KeyError(f"{score_col} not in DataFrame")

            s = df_year[score_col].values
            s_param.value = s

            # solve with a cone solver, same style as cvxgrp notebooks
            w_var.value = w_bench
            prob.solve(
                solver=cp.SCS,
                warm_start=True,
                verbose=False,
                eps=1e-3,
            )

            if w_var.value is None:
                # solver failed: fall back to benchmark
                print(f"Optimization failed for year {year}, col {c}")
                optimal_weights = w_bench
            else:
                optimal_weights = np.asarray(w_var.value).ravel()

            # assign back
            df_weights.loc[df_year.index, f"weight_{score_col}"] = optimal_weights

    return df_weights

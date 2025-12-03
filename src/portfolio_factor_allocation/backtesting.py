import numpy as np
import pandas as pd


def calculate_strategy_returns(
    df_weights, df_returns, years_shift, n_month_rebalance, cols
):
    # Prepare returns
    df_ret = df_returns.copy()
    # Ensure date is datetime or period, extract year
    if not pd.api.types.is_period_dtype(df_ret["date"]):
        df_ret["date"] = pd.to_datetime(df_ret["date"]).dt.to_period("M")

    df_ret["year"] = df_ret["date"].dt.year

    # Calculate the year of the weights we want to match with
    df_ret["match_year"] = df_ret["year"] + years_shift

    # Prepare weights
    # df_weights should have 'date' as year integer
    df_w = df_weights.copy()
    # Rename date to match_year for merging
    df_w = df_w.rename(columns={"date": "match_year"})

    # Merge returns with weights
    # We keep all returns, and attach weights where available
    df_merged = pd.merge(df_ret, df_w, on=["PERMNO", "match_year"], how="left")

    # Sort by date to ensure chronological order
    df_merged = df_merged.sort_values("date")

    unique_dates = df_merged["date"].unique()
    results = {col: [] for col in cols}

    # Pre-group by date for performance
    grouped = df_merged.groupby("date")

    for col in cols:
        current_value = 1.0
        # Holdings: Series of $ value per PERMNO
        current_holdings = pd.Series(dtype=float)

        col_returns = []

        for i, date in enumerate(unique_dates):
            # Get data for this month
            try:
                month_data = grouped.get_group(date)
            except KeyError:
                col_returns.append(0.0)
                continue

            # Rebalance logic
            # Rebalance at the start (i==0) or every n months
            if i % n_month_rebalance == 0:
                # Get target weights for this month (based on the matched year)
                valid_weights = month_data.dropna(subset=[col])

                if not valid_weights.empty:
                    # Normalize weights to invest fully in available assets
                    w = valid_weights.set_index("PERMNO")[col]
                    w_sum = w.sum()
                    if w_sum != 0:
                        w = w / w_sum
                        current_holdings = w * current_value
                    else:
                        # Weights sum to 0? Keep cash or previous holdings?
                        # Assuming cash if weights are invalid
                        current_holdings = pd.Series(dtype=float)
                else:
                    # No weights available for this period
                    # If it's the start, we can't invest.
                    if i == 0:
                        current_holdings = pd.Series(dtype=float)
                    # Else keep previous holdings (drift)

            # Calculate Returns for this month
            if current_holdings.empty:
                col_returns.append(0.0)
                continue

            # Get returns vector for this month
            # We assume missing returns (delisting etc) are 0.0 or handled in data
            rets = month_data.set_index("PERMNO")["RET"]

            # Align holdings with returns
            # Use reindex to keep only held stocks, fill missing returns with 0
            aligned_rets = rets.reindex(current_holdings.index, fill_value=0.0)

            # Update holdings value based on returns
            new_holdings = current_holdings * (1 + aligned_rets)

            new_value = new_holdings.sum()

            if current_value != 0:
                ret = new_value / current_value - 1
            else:
                ret = 0.0

            col_returns.append(ret)

            # Update state
            current_value = new_value
            current_holdings = new_holdings

        results[col] = col_returns

    df_res = pd.DataFrame(results)
    df_res["date"] = unique_dates
    return df_res


def calculate_annualized_volatility(r):
    """Calculates annualized volatility."""
    if isinstance(r, (pd.DataFrame, pd.Series)):
        return r.std() * np.sqrt(12)
    else:
        return np.std(r, axis=0, ddof=1) * np.sqrt(12)


def calculate_sharpe_ratio(r, r_f=0.0):
    """Calculates annualized Sharpe Ratio."""
    if isinstance(r, (pd.DataFrame, pd.Series)):
        excess_ret = (
            r.sub(r_f, axis=0)
            if isinstance(r_f, (pd.DataFrame, pd.Series))
            else r - r_f
        )
        return (excess_ret.mean() / excess_ret.std()) * np.sqrt(12)
    else:
        excess_ret = r - r_f
        return (np.mean(excess_ret, axis=0) / np.std(r, axis=0, ddof=1)) * np.sqrt(12)


def calculate_tracking_error(r, r_m):
    """Calculates annualized Tracking Error."""
    if isinstance(r, (pd.DataFrame, pd.Series)):
        active_ret = r.sub(r_m, axis=0)
        return active_ret.std() * np.sqrt(12)
    else:
        # Ensure r_m is broadcastable or same shape
        if r_m.ndim == 1 and r.ndim == 2:
            r_m = r_m[:, np.newaxis]
        active_ret = r - r_m
        return np.std(active_ret, axis=0, ddof=1) * np.sqrt(12)


def calculate_information_ratio(r, r_m):
    """Calculates annualized Information Ratio."""
    if isinstance(r, (pd.DataFrame, pd.Series)):
        active_ret = r.sub(r_m, axis=0)
        te = active_ret.std()
        return (active_ret.mean() / te) * np.sqrt(12)
    else:
        if r_m.ndim == 1 and r.ndim == 2:
            r_m = r_m[:, np.newaxis]
        active_ret = r - r_m
        te = np.std(active_ret, axis=0, ddof=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            return (np.mean(active_ret, axis=0) / te) * np.sqrt(12)


def block_bootstrap_metrics(
    df_returns, df_market, block_size, n_sim, r_f=0.0, seed=None
):
    """
    Performs block bootstrapping of returns and calculates performance metrics.

    Args:
        df_returns (pd.DataFrame): DataFrame containing strategy return columns.
        df_market (pd.DataFrame or pd.Series): Market returns.
        block_size (int): Size of the block in rows (months).
        n_sim (int): Number of simulations.
        r_f (float, pd.Series, pd.DataFrame): Risk-free rate.
        seed (int): Random seed.

    Returns:
        dict: Dictionary of DataFrames for each metric (sharpe, volatility, information_ratio, tracking_error).
    """
    if seed is not None:
        np.random.seed(seed)

    n_obs = len(df_returns)
    if n_obs < block_size:
        raise ValueError("Block size cannot be larger than the number of observations.")

    # Number of blocks needed to cover the time series
    n_blocks = int(np.ceil(n_obs / block_size))

    cols = df_returns.columns

    # Convert to numpy for speed
    data_r = df_returns.values

    # Handle market returns
    if isinstance(df_market, pd.DataFrame):
        data_m = df_market.values.flatten()
    else:
        data_m = df_market.values

    # Handle risk-free rate
    data_rf = r_f
    is_rf_series = False
    if isinstance(r_f, (pd.Series, pd.DataFrame)):
        is_rf_series = True
        if isinstance(r_f, pd.DataFrame):
            data_rf = r_f.values.flatten()
        else:
            data_rf = r_f.values

    results_sharpe = []
    results_vol = []
    results_ir = []
    results_te = []

    for _ in range(n_sim):
        # Sample n_blocks starting indices (Moving Block Bootstrap)
        start_indices = np.random.randint(0, n_obs - block_size + 1, size=n_blocks)

        # Construct the bootstrap sample indices
        bootstrap_indices = []
        for start_idx in start_indices:
            bootstrap_indices.extend(range(start_idx, start_idx + block_size))

        # Trim to original length
        bootstrap_indices = bootstrap_indices[:n_obs]

        # Get the samples
        sample_r = data_r[bootstrap_indices]
        sample_m = data_m[bootstrap_indices]

        if is_rf_series:
            sample_rf = data_rf[bootstrap_indices]
            # Reshape for broadcasting if needed
            if sample_rf.ndim == 1:
                sample_rf = sample_rf[:, np.newaxis]
        else:
            sample_rf = data_rf

        # Calculate metrics
        sharpes = calculate_sharpe_ratio(sample_r, sample_rf)
        vols = calculate_annualized_volatility(sample_r)
        tes = calculate_tracking_error(sample_r, sample_m)
        irs = calculate_information_ratio(sample_r, sample_m)

        results_sharpe.append(sharpes)
        results_vol.append(vols)
        results_te.append(tes)
        results_ir.append(irs)

    return {
        "sharpe": pd.DataFrame(results_sharpe, columns=cols),
        "volatility": pd.DataFrame(results_vol, columns=cols),
        "tracking_error": pd.DataFrame(results_te, columns=cols),
        "information_ratio": pd.DataFrame(results_ir, columns=cols),
    }

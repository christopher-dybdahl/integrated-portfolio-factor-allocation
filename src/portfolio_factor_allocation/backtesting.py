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

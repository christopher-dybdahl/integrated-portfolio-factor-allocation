import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf


def get_covariance_matrix(returns_df, year, permnos, window_months=24, shrinkage=True):
    # Define the window
    end_date = pd.Timestamp(year=year, month=1, day=1)
    start_date = end_date - pd.DateOffset(months=window_months)

    # Filter returns
    mask = (
        (returns_df["date"] >= start_date)
        & (returns_df["date"] < end_date)
        & (returns_df["PERMNO"].isin(permnos))
    )
    df_subset = returns_df.loc[mask]

    if df_subset.empty:
        return np.eye(len(permnos))

    # Pivot to get matrix: index=date, columns=PERMNO
    pivot_ret = df_subset.pivot(index="date", columns="PERMNO", values="RET")

    # Reindex to ensure all permnos are present and in correct order
    pivot_ret = pivot_ret.reindex(columns=permnos)

    # Fill missing values with 0
    pivot_ret = pivot_ret.fillna(0.0)

    if shrinkage:
        lw = LedoitWolf().fit(pivot_ret.values)
        sigma = lw.covariance_
    else:
        # Calculate covariance
        sigma = pivot_ret.cov()
        # Fill NaN in covariance matrix
        sigma = sigma.fillna(0.0).values

    return sigma

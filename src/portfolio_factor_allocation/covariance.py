import numpy as np
import pandas as pd


def ledoit_wolf_shrinkage(Y, k=-1):
    """
    Implement Ledoit-Wolf (2003) covariance shrinkage.
    Y: T x N matrix of returns
    k: degrees of freedom adjustment. If k < 0, demean data and set k=1.
    """
    Y = np.array(Y)
    N, p = Y.shape

    if k < 0:
        # Demean the data
        Y = Y - Y.mean(axis=0)
        k = 1

    n = N - k  # effective sample size

    # Sample covariance
    sample = (Y.T @ Y) / n

    # Diagonal of sample covariance (variances)
    samplevar = np.diag(sample)
    sqrtvar = np.sqrt(samplevar)

    # Average correlation
    # rBar <- (sum(sample / outer(sqrtvar, sqrtvar)) - p) / (p * (p - 1))
    # outer(sqrtvar, sqrtvar) is matrix where M[i,j] = sqrtvar[i]*sqrtvar[j]
    outer_sqrtvar = np.outer(sqrtvar, sqrtvar)

    # Avoid division by zero if variance is 0
    with np.errstate(divide="ignore", invalid="ignore"):
        correlation = sample / outer_sqrtvar
    correlation[~np.isfinite(correlation)] = 0.0  # Handle 0 variance cases

    if p > 1:
        rBar = (np.sum(correlation) - p) / (p * (p - 1))
    else:
        rBar = 0.0

    # Target matrix
    # target <- rBar * outer(sqrtvar, sqrtvar)
    target = rBar * outer_sqrtvar
    # diag(target) <- samplevar
    np.fill_diagonal(target, samplevar)

    # Estimate pi
    # Y2 <- Y^2
    Y2 = Y**2
    # sample2 <- (t(Y2) %*% Y2) / n
    sample2 = (Y2.T @ Y2) / n
    # piMat <- sample2 - sample^2
    piMat = sample2 - sample**2
    # pihat <- sum(piMat)
    pihat = np.sum(piMat)

    # Estimate gamma
    # gammahat <- norm(c(sample - target), type = "2")^2
    gammahat = np.linalg.norm(sample - target, "fro") ** 2

    # Estimate rho
    # rho_diag <- sum(diag(piMat))
    rho_diag = np.sum(np.diag(piMat))

    # rho_off
    # term1 <- (t(Y^3) %*% Y) / n
    term1 = ((Y**3).T @ Y) / n

    # term2 <- rep.row(samplevar, p) * sample
    # rep.row(samplevar, p) creates matrix where row i is samplevar
    # So M[i,j] = samplevar[j]
    # In Python broadcasting: samplevar[None, :] * sample
    # term2 <- t(term2)
    # So term2[i,j] = samplevar[i] * sample[j,i] = samplevar[i] * sample[i,j] (since symmetric)
    # This corresponds to samplevar[:, None] * sample
    term2 = samplevar[:, None] * sample

    thetaMat = term1 - term2
    np.fill_diagonal(thetaMat, 0)

    # rho_off <- rBar * sum(outer(1/sqrtvar, sqrtvar) * thetaMat)
    # outer(1/sqrtvar, sqrtvar)[i,j] = (1/sqrtvar[i]) * sqrtvar[j] = sqrtvar[j] / sqrtvar[i]
    with np.errstate(divide="ignore", invalid="ignore"):
        outer_term = np.outer(1 / sqrtvar, sqrtvar)
    outer_term[~np.isfinite(outer_term)] = 0.0

    rho_off = rBar * np.sum(outer_term * thetaMat)

    # Shrinkage intensity
    rhohat = rho_diag + rho_off

    if gammahat != 0:
        kappahat = (pihat - rhohat) / gammahat
    else:
        kappahat = 0.0

    shrinkage = max(0, min(1, kappahat / n))

    # Shrinkage estimator
    sigmahat = shrinkage * target + (1 - shrinkage) * sample

    return sigmahat


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
        sigma = ledoit_wolf_shrinkage(pivot_ret.values)
    else:
        # Calculate covariance
        sigma = pivot_ret.cov()
        # Fill NaN in covariance matrix
        sigma = sigma.fillna(0.0).values

    return sigma

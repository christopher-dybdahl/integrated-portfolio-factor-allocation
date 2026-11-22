# Data

The firm-level input data for this project comes from CRSP and Compustat via
[WRDS](https://wrds-www.wharton.upenn.edu/) and cannot be redistributed under the
WRDS license. To reproduce the study you need a WRDS subscription and the
following inputs:

- `monthly_returns.csv` — CRSP monthly stock file (`crsp.msf`): `PERMNO`, `date`
  (YYYY-MM), `RET`, for all common shares (share codes 10/11) on NYSE/AMEX/NASDAQ,
  1962–2024.
- `factors.csv` — firm-year characteristics built from CRSP/Compustat
  (`comp.funda` joined via `crsp.ccmxpf_linktable`): `PERMNO`, year, ticker, name,
  market cap, shares, momentum (`ret_geo`), 36-month volatility (`vol_36m`),
  `value`, `profitability`, `investment`.
- `monthly_rf.csv` — monthly risk-free rate (Fama–French research factors).

The aggregated outputs in `summary/` and `active_summary/` (Sharpe ratios,
information ratios, tracking errors, raw and Romano–Wolf-adjusted p-values) are
derived results and are committed, so all figures and tables can be regenerated
without WRDS access.

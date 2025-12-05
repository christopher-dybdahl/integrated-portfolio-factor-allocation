# Integrated Portfolio Factor Allocation

This project implements and analyzes various multi-factor portfolio construction methodologies, with a primary focus on comparing **Integrated** versus **Mixed** factor allocation approaches.

The codebase allows for the construction of portfolios using different weighting schemes and constraints, aiming to replicate and extend findings from academic literature on factor investing.

## Factors

The project analyzes the following equity factors:
*   **W**: Momentum (`ret_geo`)
*   **L**: Low Volatility (`vol_36m`)
*   **V**: Value (`value`)
*   **C**: Investment (`investment`)
*   **R**: Profitability (`profitability`)

## Methodologies

The project implements several portfolio construction techniques:

1.  **Percentile Portfolios**:
    *   Constructs portfolios based on top percentiles (e.g., Terciles, Deciles) of factor scores, originally suggested by Fama & French (1992).
    *   Supports both Integrated (average score then sort) and Mixed (sort on individual factors then combine) approaches.

2.  **BW Weights**:
    *   Implements a weighting scheme inspired by Bender & Wang (2016), allocating weights based on bucketed rankings.

3.  **Tracking Error (TE) Optimization**:
    *   Constructs portfolios by maximizing factor exposure subject to a Tracking Error constraint relative to a market-cap weighted benchmark, suggested in Fitzgibbons et al. (2016).
    *   Utilizes **Ledoit-Wolf Covariance Shrinkage** for robust risk estimation.
    *   Solves the resulting Quadratically Constrained Quadratic Program (QCQP) using `cvxpy`.

## Analysis Workflow

The project performs two parallel analyses to ensure robust comparisons between Integrated and Mixed approaches:

1.  **Standard Construction**: 
    *   Compares Integrated and Mixed portfolios constructed with identical constraints (e.g., both Top-Decile).
    *   **Weight Generation**: Run `notebooks/portfolio_weights.ipynb`.

2.  **Active Risk Adjusted**: 
    *   Adjusts the Mixed portfolio to target the same ex-ante Tracking Error as the Integrated portfolio. This ensures that performance differences are due to the integration efficacy rather than simple leverage or risk exposure differences.
    *   **Weight Generation**: Run `notebooks/portfolio_active_weights.ipynb`.

**Performance & Testing**:
The analysis notebooks (`strategy_returns.ipynb` and `robust_hypothesis_testing.ipynb`) are designed to handle both workflows. You must specify which dataset to process (Standard or Active Risk) within the notebook configuration before running.

## Project Structure

*   `data/`: Contains raw and processed financial data (returns, factors).
*   `notebooks/`: Jupyter notebooks for data preprocessing, weight calculation, and analysis.
    *   `data_preprocessing.ipynb`: Cleans raw data and calculates factor scores.
    *   `vif.ipynb`: Checks for multicollinearity among factors.
    *   `portfolio_weights.ipynb`: Generates weights for standard portfolio comparisons.
    *   `portfolio_active_weights.ipynb`: Generates weights where Mixed portfolios are scaled to match the active risk of Integrated portfolios.
    *   `strategy_returns.ipynb`: Computes performance metrics and plots returns.
    *   `robust_hypothesis_testing.ipynb`: Performs statistical significance testing (Bootstrap, Romano-Wolf).
*   `src/portfolio_factor_allocation/`: Python package containing the core logic.
    *   `scoring.py`: Functions for calculating Z-scores and Rank-scores.
    *   `covariance.py`: Implementation of Ledoit-Wolf shrinkage and covariance matrix estimation.
    *   `weighting.py`: Functions for different portfolio weighting schemes (Percentile, BW, TE).

## Setup and Installation

This project uses [Poetry](https://python-poetry.org/) for dependency management.

1.  **Prerequisites**: Ensure you have Python 3.12+ and Poetry installed.

2.  **Install Dependencies**:
    ```bash
    poetry install
    ```

3.  **Register Kernel**:
    If you want to use the poetry environment in VS Code or Jupyter:
    ```bash
    poetry run python -m ipykernel install --user --name=integrated-portfolio-factor-allocation
    ```

## Data Setup

You have two options for setting up the data:

1.  **Use Pre-calculated Factor Scores**:
    *   Place your factor scores file at `data/factors.csv`.
    *   The file must contain the following columns:
        *   `PERMNO`: Unique stock identifier.
        *   `date`: Date of the observation.
        *   `market_cap`: Market capitalization.
        *   Factor columns (default expected: `value`, `investment`, `profitability`, `vol_36m`, `ret_geo`).
    *   Ensure `data/monthly_returns.csv` is also present for covariance estimation.

2.  **Generate from Raw Data**:
    *   Place raw CRSP and Compustat data files (`crsp_raw.csv`, `crsp_compustat_raw.csv`) in the `data/` directory.
    *   Run the `notebooks/data_preprocessing.ipynb` notebook to process the raw data and generate `factors.csv` and `monthly_returns.csv`.

## Usage

1.  Ensure your data is set up as described above.
2.  Run the notebooks in the `notebooks/` directory (e.g., `portfolio_weights.ipynb`) to generate portfolio weights and analyze performance.

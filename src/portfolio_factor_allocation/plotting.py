import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


def plot_sharpe_comparison(df_sharpe, factor_combs, portfolios):
    """
    Plots a comparison of Sharpe Ratios for Integrated vs Mixed strategies.

    Args:
        df_sharpe (pd.DataFrame): DataFrame containing Sharpe Ratios.
                                  Columns are factor combinations (e.g., "V_W").
                                  Rows are strategy suffixes (e.g., "_int_ter", "_mix_ter").
        factor_combs (list): List of factor combinations to plot (e.g., ["V_W", "V_C"]).
        portfolios (list): List of portfolio types to compare (e.g., ["TER", "DEC", "BW"]).
                           Assumes suffixes are formatted as f"_int_{lower(portfolio)}" and f"_mix_{lower(portfolio)}".
    """

    n_portfolios = len(portfolios)
    n_factors = len(factor_combs)

    fig, axes = plt.subplots(
        n_portfolios, 1, figsize=(15, 4 * n_portfolios), sharex=True
    )
    if n_portfolios == 1:
        axes = [axes]

    # Colors
    color_int = "#d3d3d3"  # Light grey for Integrated
    color_pos_diff = "green"  # Green for positive difference (Int > Mix)
    color_neg_diff = "red"  # Red for negative difference (Int < Mix)

    for i, portfolio in enumerate(portfolios):
        ax = axes[i]
        suffix_int = f"_int_{portfolio.lower()}"
        suffix_mix = f"_mix_{portfolio.lower()}"

        # Extract data for this portfolio type across all factors
        sr_int = []
        sr_mix = []
        labels = []

        for factor in factor_combs:
            if factor in df_sharpe.columns:
                # Check if rows exist
                if suffix_int in df_sharpe.index and suffix_mix in df_sharpe.index:
                    sr_int.append(df_sharpe.loc[suffix_int, factor])
                    sr_mix.append(df_sharpe.loc[suffix_mix, factor])
                    labels.append(factor)

        x = np.arange(len(labels))
        width = 0.6

        sr_int = np.array(sr_int)
        sr_mix = np.array(sr_mix)

        # Plot Integrated bars (base)
        ax.bar(x, sr_int, width, label="Integrated", color=color_int)

        # Calculate difference
        diff = sr_int - sr_mix

        # Base is the minimum of the two
        base = np.minimum(sr_int, sr_mix)

        # Difference arrays
        diff_pos = np.maximum(sr_int - sr_mix, 0)  # Int > Mix
        diff_neg = np.maximum(sr_mix - sr_int, 0)  # Mix > Int

        # Plot Base (Grey)
        ax.bar(x, base, width, color=color_int, label="Base (Min)")

        # Plot Positive Diff (Green) - Stacked on Base
        # This represents the extra SR of Int over Mix
        ax.bar(x, diff_pos, width, bottom=base, color=color_pos_diff, label="Int > Mix")

        # Plot Negative Diff (Red) - Stacked on Base
        # This represents the extra SR of Mix over Int
        ax.bar(x, diff_neg, width, bottom=base, color=color_neg_diff, label="Mix > Int")

        ax.set_ylabel("SR p.a.")
        ax.set_title(f"Sharpe Ratio Comparison: {portfolio}")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=270)

        # Only add legend to the first plot to avoid clutter
        if i == 0:
            # Custom legend
            from matplotlib.patches import Patch

            legend_elements = [
                Patch(facecolor=color_int, label="Common Component"),
                Patch(facecolor=color_pos_diff, label="Int Outperformance"),
                Patch(facecolor=color_neg_diff, label="Mix Outperformance"),
            ]
            ax.legend(handles=legend_elements, loc="upper left", bbox_to_anchor=(1, 1))

    plt.tight_layout()
    plt.show()


def plot_sr_diff_and_pvalues(df_sharpe, df_p_values, portfolios):
    """
    Plots the difference in Sharpe Ratios (Int - Mix) and the associated p-values.

    Args:
        df_sharpe (pd.DataFrame): DataFrame containing Sharpe Ratios.
                                  Rows should include keys like f"_int_{port}" and f"_mix_{port}".
                                  Columns are factors.
        df_p_values (pd.DataFrame): DataFrame containing p-values.
                                    Rows are portfolios (e.g. "TER", "DEC").
                                    Columns are factors.
        portfolios (list): List of portfolio names (e.g. ["ter", "dec", "bw"]).
    """
    # Transpose p-values to have factors as rows (x-axis) and portfolios as columns
    df_pval = df_p_values.T

    # Parameters
    factors = df_sharpe.columns.tolist()
    n_factors = len(factors)
    n_portfolios = len(portfolios)

    fig, ax1 = plt.subplots(figsize=(15, 6))
    width = 0.2
    x = np.arange(n_factors)

    # Colors and markers
    # Base colors (Grey)
    base_colors = {"dec": "#a9a9a9", "ter": "#d3d3d3", "bw": "#808080", "te": "#505050"}
    # Significant at 10% (Yellows)
    sig10_colors = {
        "dec": "#FFD700",
        "ter": "#F0E68C",
        "bw": "#DAA520",
        "te": "#B8860B",
    }
    # Significant at 5% (Greens)
    sig5_colors = {
        "dec": "#32CD32",
        "ter": "#90EE90",
        "bw": "#228B22",
        "te": "#006400",
    }

    markers = {"dec": "+", "ter": "2", "bw": "D", "te": "s"}

    # Plot Bars (SR Diff) on Left Axis
    for i, port in enumerate(portfolios):
        # Offset x for grouped bars
        offset = (i - n_portfolios / 2 + 0.5) * width

        # Get p-values for coloring
        p_col = None
        if port.upper() in df_pval.columns:
            p_col = port.upper()
        elif port.lower() in df_pval.columns:
            p_col = port.lower()
        elif port in df_pval.columns:
            p_col = port

        current_pvals = (
            df_pval[p_col].values if p_col is not None else np.ones(n_factors)
        )

        # Calculate diffs and colors
        diffs = []
        bar_colors = []
        port_key = port.lower()

        for idx, factor in enumerate(factors):
            try:
                s_int = df_sharpe.loc[f"_int_{port_key}", factor]
                s_mix = df_sharpe.loc[f"_mix_{port_key}", factor]
                diffs.append(s_int - s_mix)
            except KeyError:
                diffs.append(0)

            # Determine color
            pval = current_pvals[idx]
            if pval < 0.05:
                bar_colors.append(sig5_colors.get(port_key, "green"))
            elif pval < 0.10:
                bar_colors.append(sig10_colors.get(port_key, "yellow"))
            else:
                bar_colors.append(base_colors.get(port_key, "gray"))

        ax1.bar(
            x + offset,
            diffs,
            width,
            label=port.upper(),
            color=bar_colors,
        )

    ax1.set_ylabel("SR diff int-mix (bars)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(factors, rotation=90)
    ax1.grid(axis="y", linestyle="-", alpha=0.3)
    ax1.axhline(0, color="black", linewidth=0.8)

    # Plot P-values on Right Axis
    ax2 = ax1.twinx()
    ax2.set_ylabel("p-value (symbols)")
    ax2.set_ylim(0, 1.05)

    # Plot markers
    for i, port in enumerate(portfolios):
        if port.upper() in df_pval.columns:
            col_name = port.upper()
        elif port.lower() in df_pval.columns:
            col_name = port.lower()
        elif port in df_pval.columns:
            col_name = port
        else:
            continue

        offset = (i - n_portfolios / 2 + 0.5) * width
        pvals = df_pval[col_name].values

        # Plot markers
        ax2.scatter(
            x + offset,
            pvals,
            marker=markers.get(port.lower(), "o"),
            color="black",
            s=40,
            label=f"{port} (pval)",
            zorder=10,
        )

    # Add significance lines
    ax2.axhline(0.05, color="black", linestyle="--", linewidth=1.5, label="5%-pval")
    ax2.axhline(0.10, color="gray", linestyle=":", linewidth=1.5, label="10%-pval")

    # Legend
    # Create custom handles for bars (Base colors)
    bar_handles = []
    for p in portfolios:
        bar_handles.append(
            plt.Rectangle(
                (0, 0),
                1,
                1,
                color=base_colors.get(p.lower(), "gray"),
                label=p.upper(),
            )
        )

    # Create custom handles for markers to match the style
    marker_handles = []
    for p in portfolios:
        # Check if p is in df_pval columns (handling case sensitivity)
        if (
            p in df_pval.columns
            or p.upper() in df_pval.columns
            or p.lower() in df_pval.columns
        ):
            marker_handles.append(
                Line2D(
                    [0],
                    [0],
                    color="w",
                    marker=markers.get(p.lower(), "o"),
                    markerfacecolor="k",
                    markeredgecolor="k",
                    label=p.upper(),
                )
            )

    # Add significance lines to legend
    sig_handles = [
        Line2D([0], [0], color="black", linestyle="--", label="5%-pval"),
        Line2D([0], [0], color="gray", linestyle=":", label="10%-pval"),
    ]

    final_handles = bar_handles + marker_handles + sig_handles
    # Place legend outside
    ax2.legend(
        handles=final_handles,
        loc="upper left",
        bbox_to_anchor=(1.05, 1),
        borderaxespad=0.0,
    )

    plt.tight_layout()
    plt.show()

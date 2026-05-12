"""
app/pages/portfolio.py

Portfolio Optimizer page.
Currently a structured placeholder showing the top 10 stocks
by composite score with a simple equal-weight allocation.
Full Modern Portfolio Theory optimization planned for Stage 5.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import requests
import os

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")


@st.cache_data(ttl=3600)
def fetch_top_stocks(limit: int = 20) -> list[dict]:
    """Fetches the top stocks by composite score from the leaderboard."""
    try:
        r = requests.get(
            f"{API_BASE_URL}/api/leaderboard",
            params={
                "sort_by": "composite_score",
                "limit":   limit,
                "verdict": "APPROVED_OR_FLAGGED",
            },
            timeout=10,
        )
        r.raise_for_status()
        return r.json().get("entries", [])
    except Exception as e:
        st.error(f"⚠️ Error fetching portfolio data: {e}")
        return []


def main():
    st.markdown("## 💼 Portfolio Optimizer")
    st.markdown(
        "Suggested allocations based on composite score rankings. "
        "Full Modern Portfolio Theory optimization with risk-parity "
        "weighting is planned for the next release."
    )

    st.info(
        "🚧 **Coming in Stage 5:** Mean-Variance Optimization using "
        "forecast returns and historical covariance matrix. "
        "Currently showing equal-weight allocation across top-ranked stocks."
    )

    st.divider()

    n_stocks = st.slider(
        "Number of stocks in portfolio",
        min_value=5,
        max_value=20,
        value=10,
        step=1,
    )

    stocks = fetch_top_stocks(limit=n_stocks)

    if not stocks:
        st.warning("No portfolio data available.")
        return

    df = pd.DataFrame(stocks)

    # Simple equal-weight allocation
    df["allocation_pct"] = round(100.0 / len(df), 2)

    # Expected portfolio return (weighted average of upside_pct)
    expected_return = (df["upside_pct"] * df["allocation_pct"] / 100).sum()
    avg_mape        = df["mape"].mean()
    avg_dir_acc     = df["directional_accuracy"].mean()

    # Summary metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Stocks", n_stocks)
    c2.metric("Expected 30d Return", f"{expected_return:.1f}%")
    c3.metric("Avg Model MAPE", f"{avg_mape:.1f}%")
    c4.metric("Avg Directional Accuracy", f"{avg_dir_acc:.1f}%")

    st.divider()

    # Allocation table
    st.markdown("### Suggested Allocation")
    st.dataframe(
        df[[
            "company", "sector", "current_price",
            "forecast_price", "upside_pct",
            "composite_score", "allocation_pct"
        ]].rename(columns={
            "company":       "Company",
            "sector":        "Sector",
            "current_price": "Price (₹)",
            "forecast_price":"Target (₹)",
            "upside_pct":    "Upside %",
            "composite_score":"Score",
            "allocation_pct": "Allocation %",
        }),
        use_container_width=True,
        hide_index=True,
        column_config={
            "Price (₹)":   st.column_config.NumberColumn(format="₹%.2f"),
            "Target (₹)":  st.column_config.NumberColumn(format="₹%.2f"),
            "Upside %":    st.column_config.NumberColumn(format="%.1f%%"),
            "Score":       st.column_config.ProgressColumn(
                min_value=0, max_value=100, format="%.1f"
            ),
            "Allocation %":st.column_config.NumberColumn(format="%.1f%%"),
        }
    )

    # Allocation pie chart
    st.markdown("### Sector Allocation")
    sector_alloc = (
        df.groupby("sector")["allocation_pct"]
        .sum()
        .reset_index()
    )

    fig = px.pie(
        sector_alloc,
        names="sector",
        values="allocation_pct",
        title="Portfolio Allocation by Sector",
        hole=0.4,
        color_discrete_sequence=px.colors.sequential.Blues_r,
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="#E8F0FE",
        title_font_color="#E8F0FE",
        height=400,
    )
    st.plotly_chart(fig, use_container_width=True)


main()

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def render_price_chart(df: pd.DataFrame) -> None:
    """
    Renders a Plotly price chart with Close, SMA-20, EMA-21, EMA-50,
    Bollinger Band upper and lower, and a volume subplot below.
    Expects a dataframe with columns: date, close, sma_20, ema_21,
    ema_50, bb_upper, bb_lower, volume.
    """
    st.markdown("<h3 style='margin-top: 2rem; margin-bottom: 1rem; color:#FFFFFF;'>Recent Price & Technical Indicators</h3>", unsafe_allow_html=True)
    
    if df.empty:
        st.warning("No data available to plot.")
        return

    # Use the last 100 days for the chart
    chart_data = df.tail(100).copy()
    
    # Create subplots: 2 rows, shared x-axis
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, 
        vertical_spacing=0.05, row_heights=[0.7, 0.3]
    )

    # 1. Close price (solid teal)
    if "close" in chart_data.columns:
        fig.add_trace(
            go.Scatter(x=chart_data["date"], y=chart_data["close"], name="Close", 
                       line=dict(color="teal", width=2, dash="solid")),
            row=1, col=1
        )
        
    # 2. SMA-20 (dashed gold)
    if "sma_20" in chart_data.columns:
        fig.add_trace(
            go.Scatter(x=chart_data["date"], y=chart_data["sma_20"], name="SMA-20", 
                       line=dict(color="gold", width=1.5, dash="dash")),
            row=1, col=1
        )
        
    # 3. EMA-21 (dotted white)
    if "ema_21" in chart_data.columns:
        fig.add_trace(
            go.Scatter(x=chart_data["date"], y=chart_data["ema_21"], name="EMA-21", 
                       line=dict(color="white", width=1.5, dash="dot")),
            row=1, col=1
        )
        
    # 4. EMA-50 (dotted gray)
    if "ema_50" in chart_data.columns:
        fig.add_trace(
            go.Scatter(x=chart_data["date"], y=chart_data["ema_50"], name="EMA-50", 
                       line=dict(color="gray", width=1.5, dash="dot")),
            row=1, col=1
        )
        
    # 5. Bollinger Bands (thin red dashed)
    if "bb_upper" in chart_data.columns:
        fig.add_trace(
            go.Scatter(x=chart_data["date"], y=chart_data["bb_upper"], name="BB Upper", 
                       line=dict(color="red", width=1, dash="dash")),
            row=1, col=1
        )
    if "bb_lower" in chart_data.columns:
        fig.add_trace(
            go.Scatter(x=chart_data["date"], y=chart_data["bb_lower"], name="BB Lower", 
                       line=dict(color="red", width=1, dash="dash")),
            row=1, col=1
        )
        
    # 6. Volume (navy bars)
    if "volume" in chart_data.columns:
        fig.add_trace(
            go.Bar(x=chart_data["date"], y=chart_data["volume"], name="Volume", 
                   marker_color="navy"),
            row=2, col=1
        )

    fig.update_layout(
        height=500, margin=dict(l=0, r=0, t=20, b=0),
        paper_bgcolor="#0D1B3E", plot_bgcolor="#0D1B3E",
        font=dict(color="#FFFFFF"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    # Update axes styling
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="#1E3A6E")
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="#1E3A6E")

    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

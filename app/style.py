import streamlit as st

def inject_custom_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

    /* Global Typography */
    html, body, [class*="css"]  {
        font-family: 'Inter', sans-serif !important;
        color: #94A3B8;
    }
    
    h1, h2, h3, h4, h5, h6 {
        color: #FFFFFF !important;
        font-weight: 600 !important;
    }
    
    /* Global Background */
    .stApp {
        background-color: #0D1B3E;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #0D1B3E !important;
        border-right: 1px solid #1E3A6E !important;
    }

    /* Cards */
    .stCard {
        background-color: #142252;
        border: 1px solid #1E3A6E;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1rem;
    }
    
    .stInfoCard {
        background-color: #142252;
        border: 1px solid #1E3A6E;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        border-left: 4px solid #00B4D8 !important;
    }

    /* Metric Values */
    [data-testid="stMetricValue"] {
        color: #FFB703 !important;
        font-weight: 700 !important;
    }

    /* Selectbox */
    div[data-baseweb="select"] > div {
        background-color: #142252 !important;
        border-color: #1E3A6E !important;
        color: #FFFFFF !important;
    }
    
    /* Selectbox dropdown items */
    li[role="option"] {
        background-color: #142252 !important;
        color: #FFFFFF !important;
    }
    li[role="option"]:hover {
        background-color: #1E3A6E !important;
    }
    
    /* Button */
    .stButton > button {
        background-color: #00B4D8 !important;
        color: #FFFFFF !important;
        border: none !important;
        font-weight: 600 !important;
        width: 100% !important;
        border-radius: 8px !important;
        transition: all 0.2s ease !important;
    }
    .stButton > button:hover {
        background-color: #0096B4 !important;
        box-shadow: 0 4px 6px rgba(0,0,0,0.2) !important;
    }
    
    /* Badges */
    .badge-approved {
        background-color: #06D6A0;
        color: #FFFFFF;
        font-weight: 700;
        padding: 0.5rem 1rem;
        border-radius: 8px;
        display: inline-block;
        font-size: 1.1rem;
    }
    
    .badge-flagged {
        background-color: #FFB703;
        color: #142252;
        font-weight: 700;
        padding: 0.5rem 1rem;
        border-radius: 8px;
        display: inline-block;
        font-size: 1.1rem;
    }
    
    .badge-rejected {
        background-color: #EF476F;
        color: #FFFFFF;
        font-weight: 700;
        padding: 0.5rem 1rem;
        border-radius: 8px;
        display: inline-block;
        font-size: 1.1rem;
    }
    
    /* Tag Pills */
    .tag-bullish { background-color: rgba(6, 214, 160, 0.2); color: #06D6A0; padding: 2px 8px; border-radius: 12px; font-weight: 600; font-size: 0.85rem;}
    .tag-bearish { background-color: rgba(239, 71, 111, 0.2); color: #EF476F; padding: 2px 8px; border-radius: 12px; font-weight: 600; font-size: 0.85rem;}
    .tag-neutral { background-color: rgba(148, 163, 184, 0.2); color: #94A3B8; padding: 2px 8px; border-radius: 12px; font-weight: 600; font-size: 0.85rem;}
    </style>
    """, unsafe_allow_html=True)

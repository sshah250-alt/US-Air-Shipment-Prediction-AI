import streamlit as st
import pandas as pd
import pydeck as pdk
import requests
import json
import os
import math
import time
import numpy as np
from datetime import date
import datetime

# ==========================================
# 1. CONFIGURATION & STYLING
# ==========================================
st.set_page_config(
    page_title="US Shipment Transit Time Predictor", 
    page_icon="✈️", 
    layout="wide"
)

# Force Dark Theme & Custom CSS
st.markdown("""
    <style>
    /* Main Background - Dark */
    .stApp {
        background-color: #0E1117;
        color: white;
    }
    /* Sidebar Background */
    [data-testid="stSidebar"] {
        background-color: #262730;
    }
    /* Inputs (Dropdowns, Sliders, DatePickers) text color */
    .stSelectbox, .stSlider, .stDateInput, div[data-baseweb="select"] {
        color: white;
    }
    /* Metrics Styling */
    [data-testid="stMetricValue"] {
        font-size: 24px;
        color: #00FF00; /* Bright Green for numbers */
    }
    /* Success/Error Message Text */
    .stAlert {
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. CREDENTIALS & DATA SETUP
# ==========================================

# Retrieves secrets from environment variables
DATABRICKS_URL = os.environ.get("DATABRICKS_URL")
DATABRICKS_TOKEN = os.environ.get("DATABRICKS_TOKEN")

# Coordinate Dictionary
LOCATIONS = {
    # --- ORIGINS ---
    "Warehouse_NYC": {"lat": 40.7128, "lon": -74.0060},
    "Warehouse_LA":  {"lat": 34.0522, "lon": -118.2437},
    "Warehouse_CHI": {"lat": 41.8781, "lon": -87.6298},
    "Warehouse_MIA": {"lat": 25.7617, "lon": -80.1918},
    "Warehouse_DAL": {"lat": 32.7

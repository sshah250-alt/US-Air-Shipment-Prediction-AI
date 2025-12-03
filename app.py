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
    "Warehouse_DAL": {"lat": 32.7767, "lon": -96.7970},
    "Warehouse_SEA": {"lat": 47.6062, "lon": -122.3321},
    "Warehouse_ATL": {"lat": 33.7490, "lon": -84.3880},
    "Warehouse_DEN": {"lat": 39.7392, "lon": -104.9903},
    "Warehouse_SF":  {"lat": 37.7749, "lon": -122.4194},
    "Warehouse_BOS": {"lat": 42.3601, "lon": -71.0589},
    "Warehouse_HOU": {"lat": 29.7604, "lon": -95.3698},

    # --- DESTINATIONS ---
    "New York":      {"lat": 40.7128, "lon": -74.0060},
    "Los Angeles":   {"lat": 34.0522, "lon": -118.2437},
    "Chicago":       {"lat": 41.8781, "lon": -87.6298},
    "Miami":         {"lat": 25.7617, "lon": -80.1918},
    "Dallas":        {"lat": 32.7767, "lon": -96.7970},
    "Seattle":       {"lat": 47.6062, "lon": -122.3321},
    "Atlanta":       {"lat": 33.7490, "lon": -84.3880},
    "Denver":        {"lat": 39.7392, "lon": -104.9903},
    "San Francisco": {"lat": 37.7749, "lon": -122.4194},
    "Boston":        {"lat": 42.3601, "lon": -71.0589},
    "Houston":       {"lat": 29.7604, "lon": -95.3698},
    "Portland":      {"lat": 45.5152, "lon": -122.6784},
    "Detroit":       {"lat": 42.3314, "lon": -83.0458},
    "Phoenix":       {"lat": 33.4484, "lon": -112.0740},
    "Minneapolis":   {"lat": 44.9778, "lon": -93.2650}
}

ORIGIN_OPTIONS = [key for key in LOCATIONS.keys() if key.startswith("Warehouse")]
DEST_OPTIONS = [key for key in LOCATIONS.keys() if not key.startswith("Warehouse")]

# ==========================================
# 3. HELPER FUNCTIONS
# ==========================================

def calculate_distance(lat1, lon1, lat2, lon2):
    """Calculates approximate distance in miles (Haversine formula)."""
    R = 3958.8 # Earth radius miles
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return int(R * c)

def get_curve_points(start_lat, start_lon, end_lat, end_lon, num_points=30):
    """Generates Lat/Lon points along a curved path for the map."""
    lats = np.linspace(start_lat, end_lat, num_points)
    lons = np.linspace(start_lon, end_lon, num_points)
    
    curve_factor = 5.0 # Height of the arc
    path = []
    for i in range(num_points):
        progress = i / num_points
        arc_height = math.sin(progress * math.pi) * curve_factor
        path.append([lons[i], lats[i] + arc_height])
        
    return path

def get_prediction(data_payload):
    """Sends payload to Databricks Serving Endpoint."""
    if not DATABRICKS_URL or not DATABRICKS_TOKEN:
        return "Error: Credentials missing. Set DATABRICKS_URL and DATABRICKS_TOKEN."
        
    headers = {
        "Authorization": f"Bearer {DATABRICKS_TOKEN}", 
        "Content-Type": "application/json"
    }
    
    # Databricks expects specific JSON structure
    payload = {"dataframe_split": data_payload}
    
    try:
        response = requests.post(DATABRICKS_URL, headers=headers, data=json.dumps(payload))
        if response.status_code == 200:
            result = response.json()
            if 'predictions' in result:
                return float(result['predictions'][0])
            else:
                return f"Unexpected API format: {result}"
        else:
            return f"API Error {response.status_code}: {response.text}"
    except Exception as e:
        return f"Connection Failed: {e}"


# ==========================================
# 4. UI LAYOUT & MAP
# ==========================================

st.title("🇺🇸 US Shipment Transit Time Predictor")
st.markdown("""
**Welcome to the Intelligent Logistics Dashboard.** This AI-powered application predicts the estimated **Transit Time** and **Shipping Cost** for cargo moving between major United States logistics hubs.  
Configure your shipment details in the sidebar and click **Predict** to visualize the flight path and receive real-time inference.
""")

col_map, col_inputs = st.columns([1.5

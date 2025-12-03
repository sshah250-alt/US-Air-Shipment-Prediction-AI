import streamlit as st
import pandas as pd
import pydeck as pdk
import requests
import json
import os
import math
from datetime import date

# --- Page Config (Dark Mode & Wide Layout) ---
st.set_page_config(page_title="SkyStream AI: Logistics Cloud", page_icon="📡", layout="wide")

# --- CUSTOM CSS: Force Dark Theme & Visuals ---
st.markdown("""
    <style>
    /* Main Background */
    .stApp {
        background-color: #0E1117;
        color: white;
    }
    /* Sidebar Background */
    [data-testid="stSidebar"] {
        background-color: #262730;
    }
    /* Dropdowns and Inputs */
    .stSelectbox, .stSlider, .stDateInput {
        color: white;
    }
    /* Success/Error Message Text */
    .stAlert {
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# --- 1. Secure Connection Setup ---
DATABRICKS_URL = os.environ.get("DATABRICKS_URL")
DATABRICKS_TOKEN = os.environ.get("DATABRICKS_TOKEN")

# --- 2. Locations Data ---
LOCATIONS = {
    # --- ORIGINS (Matches 'Origin_Warehouse') ---
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

    # --- DESTINATIONS (Matches 'Destination') ---
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

# --- Helper: Calculate Distance ---
def calculate_distance(lat1, lon1, lat2, lon2):
    R = 3958.8 # Earth radius miles
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return int(R * c)

# --- 3. API Function ---
def get_prediction(data_payload):
    if not DATABRICKS_URL or not DATABRICKS_TOKEN:
        return "Error: Missing Credentials."
    headers = {"Authorization": f"Bearer {DATABRICKS_TOKEN}", "Content-Type": "application/json"}
    payload = {"dataframe_split": data_payload}
    try:
        response = requests.post(DATABRICKS_URL, headers=headers, data=json.dumps(payload))
        if response.status_code == 200:
            return response.json()['predictions'][0]
        else:
            return f"Error {response.status_code}: {response.text}"
    except Exception as e:
        return f"Connection Failed: {e}"

# --- UI Layout ---
st.title("📡 SkyStream AI: Logistics Cloud")

col_map, col_inputs = st.columns([1.5, 1])

with col_inputs:
    st.subheader("📦 Configure Shipment")
    
    # 1. Routes
    origin_name = st.selectbox("📍 Origin Warehouse", ORIGIN_OPTIONS, index=0)
    dest_name = st.selectbox("🏁 Destination City", DEST_OPTIONS, index=1)
    
    # 2. Distance (Auto)
    origin_coords = LOCATIONS[origin_name]
    dest_coords = LOCATIONS[dest_name]
    real_distance = calculate_distance(origin_coords['lat'], origin_coords['lon'], dest_coords['lat'], dest_coords['lon'])
    
    # 3. Cargo Details
    weight = st.slider("Weight (kg)", 1, 1000, 150)
    courier = st.selectbox("Courier", ["FedEx", "DHL", "UPS", "USPS", "OnTrac", "Amazon Logistics", "LaserShip"])
    
    # 4. Dates
    delivery_date = st.date_input("Expected Delivery Date", value=date.today())
    
    # 5. Automated Cost
    cost = (real_distance * 0.1) + (weight * 0.5)
    
    c1, c2 = st.columns(2)
    c1.metric("Distance", f"{real_distance} mi")
    c2.metric("Est. Cost", f"${cost:.2f}")

    # Visual Box
    st.markdown(f"""
        <div style="height:80px; background:#262730; border:2px dashed #4CAF50; border-radius:10px; display:flex; justify-content:center; align-items:center;">
            <b style="font-size:20px; color:white;">📦 {weight} kg | {delivery_date}</b>
        </div>
        <br>
    """, unsafe_allow_html=True)
    
    predict_btn = st.button("🚀 Let's Predict Time", type="primary", use_container_width=True)

# --- Map ---
layer_data = [
    {"name": origin_name, "lat": origin_coords["lat"], "lon": origin_coords["lon"], "color": [0, 255, 0]},
    {"name": dest_name, "lat": dest_coords["lat"], "lon": dest_coords["lon"], "color": [255, 0, 0]}
]

# Flight path arc
arc_data = [{"source": [origin_coords["lon"], origin_coords["lat"]], "target": [dest_coords["lon"], dest_coords["lat"]]}]

# Center the view between the two points
mid_lat = (origin_coords["lat"] + dest_coords["lat"]) / 2
mid_lon = (origin_coords["lon"] + dest_coords["lon"]) / 2
view_state = pdk.ViewState(latitude=mid_lat, longitude=mid_lon, zoom=3, pitch=45)

with col_map:
    st.pydeck_chart(pdk.Deck(
        layers=[
            # Cities (Scatterplot)
            pdk.Layer(
                "ScatterplotLayer",
                data=layer_data,
                get_position="[lon, lat]",
                get_fill_color="color",
                get_radius=80000,  # Radius in meters
                pickable=True
            ),
            # Flight Path (Arc)
            pdk.Layer(
                "ArcLayer",
                data=arc_data,
                get_source_position="source",
                get_target_position="target",
                get_width=6,
                get_source_color=[0, 255, 0, 180], # Green at origin
                get_target_color=[255, 0, 0, 180], # Red at destination
                get_tilt=30
            )
        ],
        initial_view_state=view_state,

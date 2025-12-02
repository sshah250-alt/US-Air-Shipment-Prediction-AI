import streamlit as st
import pandas as pd
import pydeck as pdk
import requests
import json
import os
import math

# --- Page Config ---
st.set_page_config(page_title="SkyStream AI: Logistics Cloud", page_icon="📡", layout="wide")

# --- 1. Secure Connection Setup ---
DATABRICKS_URL = os.environ.get("DATABRICKS_URL")
DATABRICKS_TOKEN = os.environ.get("DATABRICKS_TOKEN")

# --- 2. Locations Data ---
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

# --- Helper: Calculate Distance (Haversine Formula) ---
def calculate_distance(lat1, lon1, lat2, lon2):
    R = 3958.8 # Earth radius in miles
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return int(R * c)

# --- 3. API Function ---
def get_prediction(data_payload):
    if not DATABRICKS_URL or not DATABRICKS_TOKEN:
        return "Error: Missing Credentials. Check Render Env Vars."
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
st.title("📡 SkyStream AI: Live Logistics Inference")

col_map, col_inputs = st.columns([1.5, 1])

with col_inputs:
    st.subheader("📦 Configure Shipment")
    
    # 1. Route Selection
    origin_name = st.selectbox("📍 Origin Warehouse", ORIGIN_OPTIONS, index=0)
    dest_name = st.selectbox("🏁 Destination City", DEST_OPTIONS, index=1)
    
    # 2. Calculate Distance Automatically
    origin_coords = LOCATIONS[origin_name]
    dest_coords = LOCATIONS[dest_name]
    real_distance = calculate_distance(origin_coords['lat'], origin_coords['lon'], dest_coords['lat'], dest_coords['lon'])
    
    # 3. Cargo Details
    weight = st.slider("Weight (kg)", 1, 1000, 150)
    courier = st.selectbox("Courier", ["FedEx", "DHL", "UPS", "USPS", "OnTrac", "Amazon Logistics", "LaserShip"])
    
    # 4. Automated Cost Calculation (Hidden Logic)
    # Formula: $0.10 per mile + $0.50 per kg
    cost = (real_distance * 0.1) + (weight * 0.5)
    
    # Display calculated metrics cleanly
    c1, c2 = st.columns(2)
    c1.metric("Distance", f"{real_distance} mi")
    c2.metric("Est. Cost", f"${cost:.2f}")

    # Visual Box
    st.markdown(f"""
        <div style="height:80px; background:#e0e0e0; border:2px dashed #999; border-radius:10px; display:flex; justify-content:center; align-items:center;">
            <b style="font-size:20px; color:#333;">📦 {weight} kg | ${cost:.2f}</b>
        </div>
        <br>
    """, unsafe_allow_html=True)
    
    predict_btn = st.button("🚀 Predict via Databricks API", type="primary", use_container_width=True)

# --- Map ---
layer_data = [
    {"name": origin_name, "lat": origin_coords["lat"], "lon": origin_coords["lon"], "color": [0, 255, 0]},
    {"name": dest_name, "lat": dest_coords["lat"], "lon": dest_coords["lon"], "color": [255, 0, 0]}
]
arc_data = [{"source": [origin_coords["lon"], origin_coords["lat"]], "target": [dest_coords["lon"], dest_coords["lat"]]}]

view_state = pdk.ViewState(latitude=39.0, longitude=-98.0, zoom=3, pitch=40)

with col_map:
    st.pydeck_chart(pdk.Deck(
        layers=[
            pdk.Layer("ScatterplotLayer", data=layer_data, get_position="[lon, lat]", get_color="color", get_radius=100000),
            pdk.Layer("ArcLayer", data=arc_data, get_source_position="source", get_target_position="target", get_width=5, get_source_color=[0,255,0], get_target_color=[255,0,0])
        ],
        initial_view_state=view_state,
        map_style="mapbox://styles/mapbox/dark-v10"
    ))

# --- Trigger Prediction ---
if predict_btn:
    # Prepare Data with ALL required columns (Cost and Distance are now computed)
    input_data = {
        "columns": [
            "Carrier", 
            "Origin_Warehouse", 
            "Destination", 
            "Shipment_Month", 
            "Distance_miles",     
            "Weight_kg",          
            "Cost",               # Auto-Calculated
            "Status"              
        ],
        "data": [[
            courier, 
            origin_name,   
            dest_name,     
            "December",    
            real_distance, 
            weight, 
            cost,          
            "On Time"      
        ]]
    }
    
    with st.spinner("Sending request to Databricks Cloud..."):
        prediction = get_prediction(input_data)
        
        if isinstance(prediction, (int, float)):
            st.success(f"Estimated Transit Time: **{prediction:.2f} Days**")
            st.balloons()
        else:
            st.error(prediction)

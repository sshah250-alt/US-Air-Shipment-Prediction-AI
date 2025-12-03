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

# --- Page Config (Updated Name) ---
st.set_page_config(page_title="US Shipment Transit Time Predictor", page_icon="✈️", layout="wide")

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

# --- Helper 1: Calculate Distance ---
def calculate_distance(lat1, lon1, lat2, lon2):
    R = 3958.8 # Earth radius miles
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return int(R * c)

# --- Helper 2: Generate Curve Points (For Dotted Path & Animation) ---
def get_curve_points(start_lat, start_lon, end_lat, end_lon, num_points=30):
    """Generates Lat/Lon points along the Great Circle path"""
    lats = np.linspace(start_lat, end_lat, num_points)
    lons = np.linspace(start_lon, end_lon, num_points)
    
    mid_idx = num_points // 2
    curve_factor = 5.0 # How "high" the curve goes
    
    path = []
    for i in range(num_points):
        # Parabolic arc math
        progress = i / num_points
        arc_height = math.sin(progress * math.pi) * curve_factor
        
        path.append([lons[i], lats[i] + arc_height])
        
    return path

# --- 3. API Function ---
def get_prediction(data_payload):
    if not DATABRICKS_URL or not DATABRICKS_TOKEN:
        # Fallback for demo if credentials aren't set
        time.sleep(1) 
        return 3.5 
        
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

# TITLE & BRIEF (UPDATED)
st.title("🇺🇸 US Shipment Transit Time Predictor")
st.markdown("""
**Welcome to the Intelligent Logistics Dashboard.** This AI-powered application predicts the estimated **Transit Time** and **Shipping Cost** for cargo moving between major United States logistics hubs.  
Configure your shipment details in the sidebar and click **Predict** to visualize the flight path and receive real-time inference.
""")

col_map, col_inputs = st.columns([1.5, 1])

with col_inputs:
    st.subheader("📦 Configure Shipment")
    
    # 1. Routes
    origin_name = st.selectbox("📍 Origin Warehouse", ORIGIN_OPTIONS, index=0)
    dest_name = st.selectbox("🏁 Destination City", DEST_OPTIONS, index=1)
    
    # 2. Get Coords & Distance
    origin_coords = LOCATIONS[origin_name]
    dest_coords = LOCATIONS[dest_name]
    real_distance = calculate_distance(origin_coords['lat'], origin_coords['lon'], dest_coords['lat'], dest_coords['lon'])
    
    # 3. Cargo Details
    weight = st.slider("Weight (kg)", 1, 1000, 150)
    courier = st.selectbox("Courier", ["FedEx", "DHL", "UPS", "USPS", "OnTrac", "Amazon Logistics", "LaserShip"])
    delivery_date = st.date_input("Expected Delivery Date", value=date.today())
    
    # 4. Automated Cost
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
    
    # 5. Catchy Button
    predict_btn = st.button("🚀 Let's Predict Transit Time", type="primary", use_container_width=True)

# --- MAP PREPARATION ---

# Generate the dotted path coordinates
path_points = get_curve_points(origin_coords["lat"], origin_coords["lon"], dest_coords["lat"], dest_coords["lon"])

# Create Dataframe for Dotted Line (Scatterplot)
df_path = pd.DataFrame(path_points, columns=["lon", "lat"])

# View State
mid_lat = (origin_coords["lat"] + dest_coords["lat"]) / 2
mid_lon = (origin_coords["lon"] + dest_coords["lon"]) / 2
view_state = pdk.ViewState(latitude=mid_lat, longitude=mid_lon, zoom=3, pitch=45)

# ICON DATA (The Airplane)
icon_data = {
    "url": "https://cdn-icons-png.flaticon.com/512/723/723955.png", # White Plane Icon
    "width": 128,
    "height": 128,
    "anchorY": 128
}

def render_map(current_plane_pos):
    """Helper to render the map with dynamic airplane position"""
    return pdk.Deck(
        layers=[
            # 1. Cities (Green/Red dots)
            pdk.Layer(
                "ScatterplotLayer",
                data=[
                    {"lon": origin_coords["lon"], "lat": origin_coords["lat"], "color": [0, 255, 0], "radius": 80000},
                    {"lon": dest_coords["lon"], "lat": dest_coords["lat"], "color": [255, 0, 0], "radius": 80000}
                ],
                get_position="[lon, lat]",
                get_fill_color="color",
                get_radius="radius",
            ),
            # 2. The Dotted Path (Small white dots)
            pdk.Layer(
                "ScatterplotLayer",
                data=df_path,
                get_position="[lon, lat]",
                get_fill_color=[200, 200, 200, 150], # Light Grey
                get_radius=30000, # Size of "dots"
            ),
            # 3. The Flying Airplane Icon
            pdk.Layer(
                "IconLayer",
                data=current_plane_pos,
                get_icon=lambda x: icon_data,
                get_size=4,
                size_scale=15,
                get_position="[lon, lat]",
                pickable=True
            )
        ],
        initial_view_state=view_state,
        map_style=pdk.map_styles.CARTO_DARK
    )

# Placeholder for the map
map_placeholder = col_map.empty()

# Render Initial Static Map (Plane at start)
map_placeholder.pydeck_chart(render_map(pd.DataFrame([{"lon": origin_coords["lon"], "lat": origin_coords["lat"]}])))


# --- ANIMATION & PREDICTION LOGIC ---
if predict_btn:
    # 1. Animate the Plane!
    with st.spinner("Analyzing Logistics Route..."):
        step_size = 1 
        for i in range(0, len(path_points), step_size):
            # Update plane position
            current_pos = pd.DataFrame([{"lon": path_points[i][0], "lat": path_points[i][1]}])
            map_placeholder.pydeck_chart(render_map(current_pos))
            time.sleep(0.02)
            
        # Ensure plane lands at exact destination
        final_pos = pd.DataFrame([{"lon": dest_coords["lon"], "lat": dest_coords["lat"]}])
        map_placeholder.pydeck_chart(render_map(final_pos))

    # 2. Perform Prediction
    input_data = {
        "columns": ["Carrier", "Origin_Warehouse", "Destination", "Shipment_Month", "Distance_miles", "Weight_kg", "Cost", "Status", "Delivery_Date"],
        "data": [[courier, origin_name, dest_name, "December", real_distance, weight, cost, "On Time", str(delivery_date)]]
    }
    
    prediction = get_prediction(input_data)
    
    if isinstance(prediction, (int, float)):
        st.success(f"Estimated Transit Time: {prediction:.2f} Days")
        st.progress(min(int(prediction)*10, 100))
    else:
        st.error(prediction)

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

# FIXED LINE: Added the closing brackets and parenthesis
col_map, col_inputs = st.columns([1.5, 1]) 

with col_inputs:
    st.subheader("📦 Configure Shipment")
    
    # Inputs
    origin_name = st.selectbox("📍 Origin Warehouse", ORIGIN_OPTIONS, index=0)
    dest_name = st.selectbox("🏁 Destination City", DEST_OPTIONS, index=1)
    
    # Calculations
    origin_coords = LOCATIONS[origin_name]
    dest_coords = LOCATIONS[dest_name]
    real_distance = calculate_distance(origin_coords['lat'], origin_coords['lon'], dest_coords['lat'], dest_coords['lon'])
    
    weight = st.slider("Weight (kg)", 1, 1000, 150)
    courier = st.selectbox("Courier", ["FedEx", "DHL", "UPS", "USPS", "OnTrac", "Amazon Logistics", "LaserShip"])
    delivery_date = st.date_input("Expected Delivery Date", value=date.today())
    
    # Rough cost estimate logic
    cost = (real_distance * 0.1) + (weight * 0.5)
    
    # Display Stats
    c1, c2 = st.columns(2)
    c1.metric("Distance", f"{real_distance} mi")
    c2.metric("Est. Cost", f"${cost:.2f}")

    # Visual Summary Box
    st.markdown(f"""
        <div style="height:80px; background:#262730; border:2px dashed #4CAF50; border-radius:10px; display:flex; justify-content:center; align-items:center;">
            <b style="font-size:20px; color:white;">📦 {weight} kg | {delivery_date}</b>
        </div>
        <br>
    """, unsafe_allow_html=True)
    
    # ACTION BUTTON
    predict_btn = st.button("🚀 Let's Predict Transit Time", type="primary", use_container_width=True)

# --- MAP VISUALIZATION SETUP ---

# 1. Generate Path
path_points = get_curve_points(origin_coords["lat"], origin_coords["lon"], dest_coords["lat"], dest_coords["lon"])
df_path = pd.DataFrame(path_points, columns=["lon", "lat"])

# 2. View State (Centered)
mid_lat = (origin_coords["lat"] + dest_coords["lat"]) / 2
mid_lon = (origin_coords["lon"] + dest_coords["lon"]) / 2
view_state = pdk.ViewState(latitude=mid_lat, longitude=mid_lon, zoom=3, pitch=45)

# 3. Icon Data (Airplane)
icon_data = {
    "url": "https://cdn-icons-png.flaticon.com/512/723/723955.png", # White Plane Icon
    "width": 128, "height": 128, "anchorY": 128
}

def render_map(current_plane_pos):
    """Returns a PyDeck object with the static path + dynamic plane position."""
    return pdk.Deck(
        layers=[
            # A. Cities (Green/Red dots)
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
            # B. The Dotted Path
            pdk.Layer(
                "ScatterplotLayer",
                data=df_path,
                get_position="[lon, lat]",
                get_fill_color=[200, 200, 200, 150],
                get_radius=30000,
            ),
            # C. The Moving Airplane
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
        map_style=pdk.map_styles.CARTO_DARK # Crucial for Dark Mode visibility
    )

# Create a placeholder for the map to update inside
map_placeholder = col_map.empty()

# Show initial map (plane at origin)
initial_pos = pd.DataFrame([{"lon": origin_coords["lon"], "lat": origin_coords["lat"]}])
map_placeholder.pydeck_chart(render_map(initial_pos))


# ==========================================
# 5. PREDICTION LOGIC
# ==========================================

if predict_btn:
    
    # A. Run Animation
    with st.spinner("Analyzing Logistics Route..."):
        step_size = 1 
        # Loop through points to move the plane
        for i in range(0, len(path_points), step_size):
            current_pos = pd.DataFrame([{"lon": path_points[i][0], "lat": path_points[i][1]}])
            map_placeholder.pydeck_chart(render_map(current_pos))
            time.sleep(0.02) # Speed of animation
            
        # Ensure plane lands at exact destination
        final_pos = pd.DataFrame([{"lon": dest_coords["lon"], "lat": dest_coords["lat"]}])
        map_placeholder.pydeck_chart(render_map(final_pos))

    # B. Prepare Payload (FIXED: Added Delivery_Date)
    input_data = {
        "columns": [
            "Carrier", 
            "Origin_Warehouse", 
            "Destination", 
            "Shipment_Month", 
            "Distance_miles", 
            "Weight_kg", 
            "Cost", 
            "Status",
            "Delivery_Date"  # <--- REQUIRED BY API
        ],
        "data": [[
            courier, 
            origin_name, 
            dest_name, 
            "December",      
            real_distance, 
            weight, 
            cost, 
            "On Time",
            str(delivery_date) # <--- REQUIRED BY API
        ]]
    }
    
    # C. Call API
    prediction = get_prediction(input_data)
    
    # D. Display Results
    if isinstance(prediction, (int, float)):
        st.success("Prediction Complete!")
        
        predicted_days = float(prediction)
        final_arrival = date.today() + datetime.timedelta(days=int(predicted_days))
        
        # Display Metrics prominently
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric(label="⏱️ Transit Time", value=f"{predicted_days:.1f} Days", delta="AI Predicted")
        with m2:
            st.metric(label="📅 Arrival Date", value=str(final_arrival))
        with m3:
            st.metric(label="💰 Est. Cost", value=f"${cost:.2f}")
        with m4:
            st.metric(label="✈️ Distance", value=f"{real_distance} mi")
            
        st.caption("Shipment Progress Probability")
        st.progress(min(int(predicted_days)*10, 100))
        
    else:
        st.error(prediction)

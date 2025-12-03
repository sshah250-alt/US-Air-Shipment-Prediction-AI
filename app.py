import streamlit as st
import pydeck as pdk
import pandas as pd
import datetime
import numpy as np

# --- 1. CONFIGURATION & STYLING ---
st.set_page_config(page_title="ShipTime AI", layout="wide")

# Custom CSS for Background and Typography
st.markdown("""
    <style>
    /* Main Background - Subtle Gradient */
    .stApp {
        background: linear-gradient(to right, #f8f9fa, #e9ecef);
        color: #333333;
    }
    
    /* Sidebar Background */
    section[data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #ddd;
    }

    /* Titles */
    h1 {
        color: #1A3A68; /* Navy Blue */
        font-family: 'Helvetica', sans-serif;
    }
    
    /* Button Styling */
    div.stButton > button {
        background-color: #FF4B4B;
        color: white;
        font-size: 18px;
        font-weight: bold;
        border-radius: 10px;
        padding: 0.5rem 1rem;
        border: none;
        width: 100%;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transition: all 0.3s;
    }
    div.stButton > button:hover {
        background-color: #D43F3F;
        transform: scale(1.02);
    }
    </style>
""", unsafe_allow_html=True)

# --- 2. DATA SETUP ---
# Dictionary of major US logistics hubs with Lat/Lon
LOCATIONS = {
    "New York (JFK)": {"lat": 40.6413, "lon": -73.7781},
    "Los Angeles (LAX)": {"lat": 33.9416, "lon": -118.4085},
    "Chicago (ORD)": {"lat": 41.9742, "lon": -87.9073},
    "Houston (IAH)": {"lat": 29.9902, "lon": -95.3368},
    "Miami (MIA)": {"lat": 25.7617, "lon": -80.1918},
    "San Francisco (SFO)": {"lat": 37.6213, "lon": -122.3790},
    "Seattle (SEA)": {"lat": 47.4502, "lon": -122.3088},
    "Atlanta (ATL)": {"lat": 33.6407, "lon": -84.4277},
    "Denver (DEN)": {"lat": 39.8561, "lon": -104.6737}
}

# --- 3. SIDEBAR: CONFIGURATION ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2830/2830305.png", width=50) # Placeholder Icon
    st.title("📦 Configure Shipment")
    st.markdown("---")
    
    # Origin & Dest
    origin_name = st.selectbox("📍 Origin Warehouse", options=LOCATIONS.keys(), index=0)
    dest_name = st.selectbox("🏁 Destination City", options=LOCATIONS.keys(), index=1)
    
    # Shipment Details
    weight = st.slider("Weight (kg)", 1, 1000, 150)
    courier = st.selectbox("Courier Service", ["FedEx", "UPS", "DHL", "USPS Priority"])
    ship_date = st.date_input("Expected Ship Date", datetime.date.today())
    
    st.markdown("---")
    
    # PREDICT BUTTON
    predict_btn = st.button("🚀 Let's Predict Transit Time")

# --- 4. MAIN AREA ---

# Title and Brief
st.title("🇺🇸 US Shipment Transit Time Predictor")
st.markdown("""
**Welcome to the Intelligent Logistics Dashboard.** This tool utilizes advanced machine learning to estimate transit times and shipping costs across the United States. 
Select your route in the sidebar to visualize the flight path and get instant predictions.
""")

# --- 5. MAP VISUALIZATION ---

# Get coordinates
origin_coords = LOCATIONS[origin_name]
dest_coords = LOCATIONS[dest_name]

# Calculate Midpoint for view state
mid_lat = (origin_coords["lat"] + dest_coords["lat"]) / 2
mid_lon = (origin_coords["lon"] + dest_coords["lon"]) / 2

# Create Map Data
map_data = pd.DataFrame({
    "start_lat": [origin_coords["lat"]],
    "start_lon": [origin_coords["lon"]],
    "end_lat": [dest_coords["lat"]],
    "end_lon": [dest_coords["lon"]],
    "name": [f"{origin_name} -> {dest_name}"]
})

# PyDeck Layers
# 1. Arc Layer (The Flight Path)
arc_layer = pdk.Layer(
    "ArcLayer",
    data=map_data,
    get_source_position=["start_lon", "start_lat"],
    get_target_position=["end_lon", "end_lat"],
    get_source_color=[255, 0, 0, 200],   # Red at origin
    get_target_color=[0, 255, 0, 200],   # Green at dest
    get_width=5,
    get_tilt=30,
    pickable=True,
    auto_highlight=True,
)

# 2. Scatterplot Layer (The Cities)
points_data = [
    {"position": [origin_coords["lon"], origin_coords["lat"]], "color": [255, 0, 0, 255], "radius": 10000},
    {"position": [dest_coords["lon"], dest_coords["lat"]], "color": [0, 255, 0, 255], "radius": 10000},
]
scatter_layer = pdk.Layer(
    "ScatterplotLayer",
    data=points_data,
    get_position="position",
    get_fill_color="color",
    get_radius="radius",
    pickable=True,
)

# View State (Centers the map on the US/Route)
view_state = pdk.ViewState(
    latitude=mid_lat,
    longitude=mid_lon,
    zoom=3,
    pitch=40,
)

# Render Map
st.pydeck_chart(pdk.Deck(
    map_style="mapbox://styles/mapbox/light-v9", # LIGHT STYLE FOR VISIBILITY
    initial_view_state=view_state,
    layers=[arc_layer, scatter_layer],
    tooltip={"html": "<b>Route:</b> {name}", "style": {"color": "white"}}
))


# --- 6. PREDICTION RESULTS ---

if predict_btn:
    # ---------------------------------------------------------
    # MOCK API CALL (Replace this with your real Databricks call)
    # ---------------------------------------------------------
    import time
    with st.spinner("Calculating flight path and logistics inference..."):
        time.sleep(1.5) # Simulate API delay
        
        # Mock logic
        distance = np.sqrt((origin_coords["lat"]-dest_coords["lat"])**2 + (origin_coords["lon"]-dest_coords["lon"])**2) * 69 # rough miles
        est_cost = (distance * 0.1) + (weight * 0.5)
        arrival_date = ship_date + datetime.timedelta(days=int(distance/500) + 1)
        
    st.success("Prediction Complete!")
    
    # Display Metrics in columns
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(label="✈️ Estimated Distance", value=f"{int(distance)} mi")
    
    with col2:
        st.metric(label="💰 Estimated Cost", value=f"${est_cost:.2f}")
        
    with col3:
        st.metric(label="📅 Arrival Date", value=f"{arrival_date}")

    # JSON output for debugging (optional, can remove)
    # st.json({"origin": origin_name, "destination": dest_name, "cost": est_cost})

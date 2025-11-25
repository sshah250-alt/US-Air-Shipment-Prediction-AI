import streamlit as st
import pandas as pd
import pydeck as pdk
import requests
import json
import time
import os

# --- Page Config ---
st.set_page_config(page_title="SkyStream AI (Live)", page_icon="📡", layout="wide")

# --- 1. Credentials Setup ---
# We try to get secrets from Streamlit (local) or Environment (Render)
try:
    DATABRICKS_URL = st.secrets["DATABRICKS_URL"]
    DATABRICKS_TOKEN = st.secrets["DATABRICKS_TOKEN"]
except:
    # Fallback for Render Environment Variables
    DATABRICKS_URL = os.environ.get("DATABRICKS_URL")
    DATABRICKS_TOKEN = os.environ.get("DATABRICKS_TOKEN")

if not DATABRICKS_URL or not DATABRICKS_TOKEN:
    st.error("🚨 Missing Connection Secrets! Please set DATABRICKS_URL and DATABRICKS_TOKEN.")
    st.stop()

# --- 2. The API Function ---
def get_prediction_from_databricks(data_dict):
    """
    Sends data to Databricks Model Serving and returns the prediction.
    """
    headers = {
        "Authorization": f"Bearer {DATABRICKS_TOKEN}",
        "Content-Type": "application/json",
    }
    
    # Databricks expects "dataframe_split" format usually
    payload = {
        "dataframe_split": data_dict
    }
    
    try:
        response = requests.post(DATABRICKS_URL, headers=headers, data=json.dumps(payload))
        
        if response.status_code != 200:
            st.error(f"Databricks Error ({response.status_code}): {response.text}")
            return None
            
        # Parse result: {"predictions": [5.4]}
        result = response.json()
        return result["predictions"][0]
        
    except Exception as e:
        st.error(f"Connection Failed: {e}")
        return None

# --- Locations Data ---
LOCATIONS = {
    "New York (JFK Warehouse)": {"lat": 40.6413, "lon": -73.7781},
    "Los Angeles (Port of LA)": {"lat": 33.7288, "lon": -118.2620},
    "Chicago (O'Hare Hub)": {"lat": 41.9742, "lon": -87.9073},
    "Dallas (DFW Center)": {"lat": 32.8998, "lon": -97.0403},
    "Miami (MIA Gateway)": {"lat": 25.7959, "lon": -80.2870},
    "Seattle (Tacoma Hub)": {"lat": 47.4502, "lon": -122.3088},
    "Atlanta (Hartsfield)": {"lat": 33.6407, "lon": -84.4277},
    "Denver (Rocky Mtn Hub)": {"lat": 39.8561, "lon": -104.6737}
}

# --- UI Layout ---
st.title("📡 SkyStream AI: Live Databricks Inference")

col_map, col_inputs = st.columns([1.5, 1])

with col_inputs:
    st.subheader("📦 Configure Shipment")
    
    origin_name = st.selectbox("📍 Origin", list(LOCATIONS.keys()), index=1)
    dest_name = st.selectbox("🏁 Destination", list(LOCATIONS.keys()), index=0)
    origin_coords = LOCATIONS[origin_name]
    dest_coords = LOCATIONS[dest_name]
    
    weight = st.slider("Weight (kg)", 1, 1000, 50)
    courier = st.selectbox("Courier", ["FedEx", "DHL", "UPS", "USPS"])
    mode = st.radio("Mode", ["Air", "Truck", "Rail"], horizontal=True)
    
    # Simple Visual Box
    st.markdown(f"""
        <div style="height:100px; background:#ddd; border:2px dashed #999; display:flex; justify-content:center; align-items:center;">
            <b>{weight} kg Package</b>
        </div>
    """, unsafe_allow_html=True)
    
    predict_btn = st.button("🚀 Predict via Databricks", type="primary", use_container_width=True)

# --- Map (PyDeck) ---
with col_map:
    layer_data = [
        {"name": origin_name, "lat": origin_coords["lat"], "lon": origin_coords["lon"], "color": [0, 255, 0]},
        {"name": dest_name, "lat": dest_coords["lat"], "lon": dest_coords["lon"], "color": [255, 0, 0]}
    ]
    arc_data = [{"source": [origin_coords["lon"], origin_coords["lat"]], "target": [dest_coords["lon"], dest_coords["lat"]]}]

    st.pydeck_chart(pdk.Deck(
        layers=[
            pdk.Layer("ScatterplotLayer", data=layer_data, get_position="[lon, lat]", get_color="color", get_radius=100000),
            pdk.Layer("ArcLayer", data=arc_data, get_source_position="source", get_target_position="target", get_width=5, get_source_color=[0,255,0], get_target_color=[255,0,0])
        ],
        initial_view_state=pdk.ViewState(latitude=39.0, longitude=-98.0, zoom=3, pitch=40)
    ))

# --- Prediction Logic ---
if predict_btn:
    # 1. Prepare Data (Pandas Split Format)
    # Must match training columns perfectly!
    df_input = pd.DataFrame({
        'Carrier': [courier],
        'Origin': [origin_name.split(" (")[0]],
        'Destination': [dest_name.split(" (")[0]],
        'Mode': [mode],
        'Shipment_Month': ["December"], # Dynamic month if needed
        'Distance': [2000],
        'Weight': [weight],
        'Quantity': [1]
    })
    
    # Convert to dictionary format required by Databricks
    # orient='split' gives: {"columns": [...], "data": [[...]]}
    data_dict = df_input.to_dict(orient='split')
    
    with st.spinner("Calling Databricks Endpoint..."):
        pred_days = get_prediction_from_databricks(data_dict)
        
        if pred_days is not None:
            st.success(f"Databricks Prediction: **{pred_days:.2f} Days**")
            st.balloons()

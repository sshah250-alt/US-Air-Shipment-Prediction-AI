✈️ US Shipment Transit Time Predictor

An intelligent logistics dashboard that leverages Machine Learning to predict shipment transit times between major US logistics hubs.

📖 Overview

This project is an end-to-end Machine Learning application designed to optimize logistics planning. It allows users to select origin/destination warehouses, shipment weight, and courier services to receive an instant Transit Time prediction.

The application consists of:

Frontend: A high-performance Streamlit dashboard featuring 3D geospatial visualizations (PyDeck).

Backend: A Databricks Model Serving Endpoint that hosts the trained ML model.

Model: A robust regression pipeline trained on historical shipment data, tracked and managed via MLflow.

✨ Key Features

Real-Time Inference: Connects directly to Databricks Model Serving for sub-second predictions.

Interactive Geospatial Map: Visualizes the flight path using a curved "Great Circle" arc with a moving airplane animation using PyDeck.

Dynamic UI: Custom dark-mode interface with progress bars and clear metric displays.

Cost Estimation: Real-time calculation of shipping costs based on distance and weight formulas.

🛠️ Tech Stack

Language: Python

Frontend: Streamlit, PyDeck (Deck.gl), CSS

Machine Learning: Scikit-Learn, XGBoost, MLflow

Cloud & MLOps: Databricks (Unified Analytics Platform), Databricks Model Serving

🧠 Model Development

The core of this application is a Machine Learning pipeline developed in Databricks.

Experimentation

We utilized MLflow to run an extensive hyperparameter search across 7 model families, generating over 100 unique model configurations:

Linear Models: OLS, Ridge, Lasso, ElasticNet

Tree-Based: Decision Trees, Random Forest, XGBoost, Gradient Boosting

Others: SVM (SVR), K-Nearest Neighbors (KNN), Neural Networks (MLP)

🏆 The Champion Model

After rigorous evaluation based on RMSE (Root Mean Squared Error), the champion model selected was Lasso Regression.

Model: sklearn.linear_model.Lasso

Hyperparameter: alpha=0.01

Performance: RMSE ~0.96 Days.

Why it won: The logistics data exhibited strong linear relationships (e.g., Distance vs. Time). While complex models like Neural Networks were tested, Lasso provided the best balance of accuracy and generalization by effectively filtering out noise through L1 regularization.

🚀 How to Run Locally

1. Clone the Repository

git clone [https://github.com/your-username/us-shipment-predictor.git](https://github.com/your-username/us-shipment-predictor.git)
cd us-shipment-predictor



2. Install Dependencies

pip install -r requirements.txt



3. Set Environment Variables

The app requires connection credentials for the Databricks Serving Endpoint. Create a .env file or export them in your terminal:

export DATABRICKS_URL="https://<your-workspace>[.cloud.databricks.com/serving-endpoints/](https://.cloud.databricks.com/serving-endpoints/)<endpoint-name>/invocations"
export DATABRICKS_TOKEN="dapi..."



4. Run the Streamlit App

streamlit run app.py



📂 Project Structure

├── app.py                   # Main Streamlit application
├── requirements.txt         # Python dependencies
├── notebooks/
│   └── training_script.py   # Databricks ML training & grid search script
├── assets/                  # Images and screenshots
└── README.md                # Project documentation



🔌 API Integration

The application sends a JSON payload to the Databricks Serving Endpoint with the following schema:

{
  "dataframe_split": {
    "columns": [
      "Carrier", "Origin_Warehouse", "Destination", "Shipment_Month", 
      "Distance_miles", "Weight_kg", "Cost", "Status", "Delivery_Date"
    ],
    "data": [[ "FedEx", "Warehouse_NYC", "Los Angeles", "December", 2445, 150, 319.5, "On Time", "2025-12-03" ]]
  }
}



Note: The Delivery_Date is required by the model signature but dropped during inference preprocessing.

🔮 Future Improvements

Multi-Output Prediction: Upgrade the model to predict both Transit Time and Shipping Cost simultaneously using MultiOutputRegressor.

Weather Integration: Incorporate live weather API data to account for delays.

Carrier Specifics: Add more granular data for specific carrier service levels (e.g., "Overnight" vs "Ground").

📜 License

This project is licensed under the MIT License.

Created by Suman Shah

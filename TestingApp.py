import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import OneHotEncoder
from geopy.geocoders import Nominatim
import folium
from streamlit_folium import st_folium

# Load Models and Preprocessor
models = {
    "SGD": joblib.load('sgd_best_model.pkl'),
    "LightGBM": joblib.load('lgb_best_model.pkl'),
}
preprocessor = joblib.load('preprocessor.pkl')

# Initialize geolocator
geolocator = Nominatim(user_agent="geoapi")

# Title of the app
st.title("CO2 Emission Prediction App")

# Instructions for the user
st.write("""
This app predicts the **Total CO2 Emission** based on the Area and Year input.
You can interact with the map below to select a region or manually enter an area name.
""")

# Model Selection (get model names from the dictionary keys)
model_choice = st.selectbox("Select Model:", list(models.keys()))

# Interactive map for area selection
st.subheader("Select a Region on the Map")
m = folium.Map(location=[20, 0], zoom_start=2)  # Centered globally

# Add click functionality to the map
clicked_location = st_folium(m, width=700, height=450)

# Reverse geocoding to get area name from clicked location
selected_area = None
if clicked_location and "last_clicked" in clicked_location:
    lat, lon = clicked_location["last_clicked"]["lat"], clicked_location["last_clicked"]["lng"]
    location = geolocator.reverse((lat, lon), language="en")
    if location:
        selected_area = location.raw.get("address", {}).get("country", "Unknown")
        st.success(f"You selected: {selected_area}")
    else:
        st.warning("Unable to fetch the area name. Try another location.")
else:
    st.info("Click on the map to select a region.")

# Fallback manual input for area
area = st.text_input("Or Enter the Area (e.g., Country or Region):", value=selected_area or "")

# Input for Year
year = st.number_input("Enter the Year (e.g., 2023):", min_value=1900, max_value=2100, step=1)

# Prediction Button
if st.button("Predict"):
    if area and year:
        try:
            # Prepare Input Data:
            new_data = pd.DataFrame({'area': [area], 'year': [year]})

            # Encode 'area':
            new_data_encoded = pd.get_dummies(new_data, columns=['area'], drop_first=True)

            # Get feature names from ColumnTransformer
            feature_names = preprocessor.get_feature_names_out()

            # Handle missing columns (if any) to match training data:
            missing_cols = set(feature_names) - set(new_data_encoded.columns)
            missing_data = pd.DataFrame(0, index=new_data_encoded.index, columns=list(missing_cols))
            new_data_encoded = pd.concat([new_data_encoded, missing_data], axis=1)
            new_data_encoded = new_data_encoded[feature_names]  # Reorder columns to match training data

            # Scale features:
            new_data_scaled = preprocessor.transform(new_data_encoded)  # Use the preprocessor/scaler used during training

            # Get selected model
            model = models[model_choice]

            # Make predictions
            prediction = model.predict(new_data_scaled)[0]

            # Display result
            st.success(f"Predicted Total CO2 Emission for {area} in {year}: {prediction:.2f}")
        except Exception as e:
            st.error(f"Error during prediction: {e}")
    else:
        st.warning("Please provide valid inputs for both Area and Year.")

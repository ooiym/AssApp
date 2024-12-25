import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Load Models and Preprocessor
models = {
    "SGD": joblib.load('sgd_best_model.pkl'),
    "LightGBM": joblib.load('lgb_best_model.pkl'),
    "Lasso": joblib.load('best_lasso_model.pkl'),
    "Elastic Net": joblib.load('best_elasticnet_model.pkl'),
}
preprocessor = joblib.load('preprocessor.pkl')
scaler = joblib.load('scaler.pkl')
# Load areas from text file
try:
    with open('areas.txt', 'r') as file:
        area_list = [line.strip() for line in file if line.strip()]
except FileNotFoundError:
    st.error("The 'areas.txt' file is missing. Please add it to the working directory.")
    area_list = []
# Title of the app
st.title("CO2 Emission Prediction App")

# Instructions for the user
st.write("""
This app predicts the **Total CO2 Emission** based on the Area and Year input.
Please enter the values below to get the prediction.
""")

# Model Selection (get model names from the dictionary keys)
model_choice = st.selectbox("Select Model:", ["All Models"] + list(models.keys()))

# Input form for Area and Year
# Dropdown for Area Selection
area = st.selectbox("Select the Area:", area_list)
year = st.number_input("Enter the Year (e.g., 2023):", min_value=1900, max_value=2100, step=1)

# Input fields for additional features
savanna_fires = st.number_input("Savanna Fires:", min_value=0.0)
forest_fires = st.number_input("Forest Fires:", min_value=0.0)
crop_residues = st.number_input("Crop Residues:", min_value=0.0)
rice_cultivation = st.number_input("Rice Cultivation:", min_value=0.0)
drained_organic_soils = st.number_input("Drained Organic Soils (CO2):", min_value=0.0)
pesticides_manufacturing = st.number_input("Pesticides Manufacturing:", min_value=0.0)
food_transport = st.number_input("Food Transport:", min_value=0.0)
forestland = st.number_input("Forestland:", min_value=0.0)
manure_management = st.number_input("Manure Management:", min_value=0.0)
fires_in_organic_soils = st.number_input("Fires in Organic Soils:", min_value=0.0)
fires_in_humid_tropical_forests = st.number_input("Fires in Humid Tropical Forests:", min_value=0.0)
on_farm_energy_use = st.number_input("On-Farm Energy Use:", min_value=0.0)
rural_population = st.number_input("Rural Population:", min_value=0.0)
urban_population = st.number_input("Urban Population:", min_value=0.0)
total_population_male = st.number_input("Total Population - Male:", min_value=0.0)
total_population_female = st.number_input("Total Population - Female:", min_value=0.0)
average_temperature = st.number_input("Average Temperature (°C):", min_value=-50.0) 

if st.button("Predict"):
    if area and year:
        try:
            # Create a DataFrame with all input data
            new_data = pd.DataFrame({
                'area': [area],
                'year': [year],
                'Savanna fires': [savanna_fires],
                'Forest fires': [forest_fires],
                'Crop Residues': [crop_residues],
                'Rice Cultivation': [rice_cultivation],
                'Drained organic soils (CO2)': [drained_organic_soils],
                'Pesticides Manufacturing': [pesticides_manufacturing],
                'Food Transport': [food_transport],
                'Forestland': [forestland],
                'Manure Management': [manure_management],
                'Fires in organic soils': [fires_in_organic_soils],
                'Fires in humid tropical forests': [fires_in_humid_tropical_forests],
                'On-farm energy use': [on_farm_energy_use],
                'Rural population': [rural_population],
                'Urban population': [urban_population],
                'Total Population - Male': [total_population_male],
                'Total Population - Female': [total_population_female],
                'Average Temperature °C': [average_temperature]
            })

            # Preprocess the data (adjust this based on your preprocessor)
            # Assuming preprocessor handles all features
            new_data_transformed = preprocessor.transform(new_data) 

            # Scale features
            new_data_scaled = scaler.transform(new_data_transformed)

            if model_choice == "All Models":
                # Display predictions for all models
                st.subheader("Predictions from All Models:")
                results = {}
                for model_name, model in models.items():
                    prediction = model.predict(new_data_scaled)[0]
                    results[model_name] = prediction
                    st.write(f"{model_name}: {prediction:.2f}")

                # Optionally, display as a DataFrame
                st.write("Prediction Summary:")
                results_df = pd.DataFrame.from_dict(results, orient='index', columns=['Prediction'])
                results_df['Prediction'] = results_df['Prediction'].map('{:,.2f}'.format)
                results_df = results_df.style.set_properties(subset=['Prediction'], **{'text-align': 'right'})
                st.dataframe(results_df)
            else:
                # Get the selected model and make a prediction
                model = models[model_choice]
                prediction = model.predict(new_data_scaled)[0]
                st.success(f"Predicted Total CO2 Emission for {area} in {year} using {model_choice}: {prediction:.2f}")
        except Exception as e:
            st.error(f"Error during prediction: {e}")
    else:
        st.warning("Please provide valid inputs for all fields.")

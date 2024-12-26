import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Load Models and Preprocessor
models = {
    "SGD": joblib.load('sgd_best_model.pkl'),
    "LightGBM": joblib.load('lgb_best_model.pkl'),
    "Lasso": joblib.load('lasso_model.pkl'),
    "Elastic Net": joblib.load('elasticnet_model.pkl'),
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

# --- Theme Selection ---
st.sidebar.subheader("Theme")
theme = st.sidebar.selectbox("Select Theme", ["Light", "Dark"])
if theme == "Light":
    st.markdown(
        """
        <style>
        .stApp{
            background-color: #f0f0f0;
            color: #000000;
        }
        .stTextInput > div > div > input {
            color: #000000;
        }
        .stSelectbox label, .stNumberInput label{
            color: #000000; 
        }
        .stButton > p {
            color: #000000; 
        }
        .stSidebar{
            color: #EFF3EA;
        }
        
       </style>
       """,
       unsafe_allow_html=True,
    )

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
forestland = st.number_input("Forestland:", min_value=0.0)
urban_population = st.number_input("Urban Population:", min_value=0.0)
average_temperature = st.number_input("Average Temperature (°C):", min_value=-50.0) 

# forest_fires = st.number_input("Forest Fires:", min_value=0.0)
# crop_residues = st.number_input("Crop Residues:", min_value=0.0)
# rice_cultivation = st.number_input("Rice Cultivation:", min_value=0.0)
# drained_organic_soils = st.number_input("Drained Organic Soils (CO2):", min_value=0.0)
# pesticides_manufacturing = st.number_input("Pesticides Manufacturing:", min_value=0.0)
# food_transport = st.number_input("Food Transport:", min_value=0.0)
# manure_management = st.number_input("Manure Management:", min_value=0.0)
# fires_in_organic_soils = st.number_input("Fires in Organic Soils:", min_value=0.0)
# fires_in_humid_tropical_forests = st.number_input("Fires in Humid Tropical Forests:", min_value=0.0)
# on_farm_energy_use = st.number_input("On-Farm Energy Use:", min_value=0.0)
# rural_population = st.number_input("Rural Population:", min_value=0.0)
# total_population_male = st.number_input("Total Population - Male:", min_value=0.0)
# total_population_female = st.number_input("Total Population - Female:", min_value=0.0)

if st.button("Predict"):
    if area and year:
        try:
            # Create a DataFrame with all input data
            new_data = pd.DataFrame({
                'area': [area],
                'year': [year],
                'savanna_fires': [savanna_fires],
                'forest_fires': [0],
                'crop_residues': [0],
                'rice_cultivation': [0],
                'drained_organic_soils_(co2)': [0],
                'pesticides_manufacturing': [0],
                'food_transport': [0],
                'forestland': [forestland],
                'manure_management': [0],
                'fires_in_organic_soils': [0],
                'fires_in_humid_tropical_forests': [0],
                'on-farm_energy_use': [0],
                'food_household_consumption': [0], 
                'food_processing': [0], 
                'on-farm_electricity_use': [0], 
                'agrifood_systems_waste_disposal': [0], 
                'fertilizers_manufacturing': [0], 
                'total_population_-male': [0], 
                'savanna_fires': [0], 
                'drained_organic_soils(co2)': [0], 
                'rice_cultivation': [0], 
                'rural_population': [0], 
                'urban_population': [urban_population], 
                'average_temperature_°c': [average_temperature], 
                'food_retail': [0], 
                'food_packaging': [0], 
                'net_forest_conversion': [0], 
                'manure_left_on_pasture': [0], 
                'total_population_-_male': [0],
                'total_population_-_female': [0] 
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

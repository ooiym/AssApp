import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import plotly.express as px  # For choropleth map
import pycountry  # For ISO alpha codes

# Load Models and Preprocessor
models = {
    "SGD": joblib.load('sgd_best_model.pkl'),
    "LightGBM": joblib.load('lgb_best_model.pkl'),
    "Lasso": joblib.load('best_lasso_model.pkl'),
    "Elastic Net": joblib.load('best_elasticnet_model.pkl'),
}
preprocessor = joblib.load('preprocessor.pkl')
scaler = joblib.load('scaler.pkl')  # Load the scaler

# --- Create Choropleth Map for Area Selection ---
def get_iso_alpha(country_name):
    try:
        country = pycountry.countries.get(name=country_name)
        return country.alpha_3
    except:
        return None

# Assuming 'temp2' is your DataFrame with 'Area', 'total_emission', 'total_fire_emissions'
CO2_df = temp2[['Area', 'total_emission', 'total_fire_emissions']]
mean_CO2_df = CO2_df.groupby('Area').mean()

# Normalize emissions (you can adjust scaling method if needed)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
mean_CO2 = scaler.fit_transform(mean_CO2_df)
normalized_emission = pd.DataFrame(mean_CO2, columns=['mean_CO2_emission', 'mean_fire_emissions'], index=mean_CO2_df.index)
normalized_emission['Area'] = normalized_emission.index
normalized_emission['iso_alpha'] = normalized_emission['Area'].apply(get_iso_alpha)
normalized_emission['mean_fire_emissions'] = normalized_emission['mean_fire_emissions'].fillna(0)

# Create the choropleth map using Plotly Express
fig = px.choropleth(
    normalized_emission,
    locations="iso_alpha",
    color="mean_CO2_emission", 
    hover_name="Area",
    color_continuous_scale=["blue", "green", "yellow", "orange", "red"]
)

# (Optional: Add scatter_geo for fire emissions)
# ... (Add scatter_geo data to the choropleth map) ...

# Update layout
fig.update_layout(
    title={'text': "Mean Agrifood CO2 emissions and mean fire emissions by country", 'x': 0.5, 'xanchor': 'center'},
    autosize=False,
    height=600,
    width=1200
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

# Input Method Selection
input_method = st.radio("Select Input Method:", ["Type Area Name", "Select from Map"])

if input_method == "Type Area Name":
    area = st.text_input("Enter the Area (e.g., Country or Region):")
else:  # input_method == "Select from Map"
    # Display the choropleth map
    st.plotly_chart(fig) 

    # Get selected area (using st.session_state if needed for interactivity)
    if st.session_state.get("selected_area"):
        area = st.session_state.selected_area["Area"]  # Or access the area name from fig
    else:
        area = None

year = st.number_input("Enter the Year (e.g., 2023):", min_value=1900, max_value=2100, step=1)

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
            new_data_scaled = scaler.transform(new_data_encoded)

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
                st.dataframe(results_df)
            else:
                # Get the selected model and make a prediction
                model = models[model_choice]
                prediction = model.predict(new_data_scaled)[0]
                st.success(f"Predicted Total CO2 Emission for {area} in {year} using {model_choice}: {prediction:.2f}")
        except Exception as e:
            st.error(f"Error during prediction: {e}")
    else:
        st.warning("Please provide valid inputs for both Area and Year.")

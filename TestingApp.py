import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

# Load Models and Preprocessor
models = {
    "SGD": joblib.load('sgd_best_model.pkl'),
    "LightGBM": joblib.load('lgb_best_model.pkl'),
    "Lasso": joblib.load('lasso_model.pkl'),
    "Elastic Net": joblib.load('elasticnet_model.pkl'),
}
preprocessor = joblib.load('preprocessor.pkl')
scaler = joblib.load('scaler.pkl')

# Load dataset for visualizations
try:
    data = pd.read_csv("Agrofood_co2_emission.csv")
except FileNotFoundError:
    st.error("The dataset 'Agrofood_co2_emission.csv' is missing.")
    data = None

# Title of the app
st.title("CO2 Emission Prediction and Visualization App")

# Instructions for the user
st.write("""
This app predicts the **Total CO2 Emission** based on the Area and Year input.
It also provides data visualizations to explore global emission and temperature trends.
""")

# Add a sidebar for visualization and layout options
st.sidebar.header("Customization Options")
viz_choice = st.sidebar.selectbox("Choose Visualization Type:", ["None", "Global Map", "Line Graph"])
theme_choice = st.sidebar.radio("Select Theme:", ["Light", "Dark"])
st.markdown(f"""
<style>
body {{
    background-color: {'#FFFFFF' if theme_choice == 'Light' else '#2E2E2E'};
    color: {'#000000' if theme_choice == 'Light' else '#FFFFFF'};
}}
</style>
""", unsafe_allow_html=True)

# Visualization Section
if data is not None and viz_choice != "None":
    st.header("Visualization")
    if viz_choice == "Global Map":
        fig = px.choropleth(
            data,
            locations="Area",
            locationmode="country names",
            color="Total CO2 Emission",
            hover_name="Area",
            animation_frame="Year",
            title="Global CO2 Emissions Over Time",
        )
        st.plotly_chart(fig)
    elif viz_choice == "Line Graph":
        fig = px.line(
            data,
            x="Year",
            y=["Total CO2 Emission", "Average Temperature (°C)"],
            color="Area",
            title="Emission and Temperature Trends",
        )
        st.plotly_chart(fig)

# Model Selection and Input Section
st.header("Prediction")
model_choice = st.selectbox("Select Model:", ["All Models"] + list(models.keys()))
area = st.selectbox("Select the Area:", data["Area"].unique() if data is not None else [])
year = st.number_input("Enter the Year (e.g., 2023):", min_value=1900, max_value=2100, step=1)
savanna_fires = st.number_input("Savanna Fires:", min_value=0.0)
forestland = st.number_input("Forestland:", min_value=0.0)
urban_population = st.number_input("Urban Population:", min_value=0.0)
average_temperature = st.number_input("Average Temperature (°C):", min_value=-50.0)

if st.button("Predict"):
    if area and year:
        try:
            new_data = pd.DataFrame({
                'area': [area],
                'year': [year],
                'savanna_fires': [savanna_fires],
                'forestland': [forestland],
                'urban_population': [urban_population],
                'average_temperature': [average_temperature],
            })

            new_data_transformed = preprocessor.transform(new_data)
            new_data_scaled = scaler.transform(new_data_transformed)

            if model_choice == "All Models":
                st.subheader("Predictions from All Models:")
                results = {model_name: model.predict(new_data_scaled)[0] for model_name, model in models.items()}
                for model_name, prediction in results.items():
                    st.write(f"{model_name}: {prediction:.2f}")
            else:
                prediction = models[model_choice].predict(new_data_scaled)[0]
                st.success(f"Predicted Total CO2 Emission for {area} in {year} using {model_choice}: {prediction:.2f}")
        except Exception as e:
            st.error(f"Error during prediction: {e}")
    else:
        st.warning("Please provide valid inputs for all fields.")

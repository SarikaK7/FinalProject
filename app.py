import streamlit as st
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from PIL import Image  # Required to process the image

# Load the trained model
try:
    model = pickle.load(open('model.sav', 'rb'))
    expected_features = model.feature_names_in_ if hasattr(model, 'feature_names_in_') else None
except Exception as e:
    st.error(f"Error loading model: {str(e)}")
    st.stop()

# Title
st.markdown(
    """
    <style>
        .title {
            background-color: #6495ed;  /* Set your desired background color here */
            color: black;               /* Set text color */
            padding: 10px;              /* Add some padding around the title */
            font-size: 24px;            /* Set font size */
            text-align: center;         /* Center the text */
        }
    </style>
    <div class="title">
        Used Car Price Prediction
    </div>
    """, unsafe_allow_html=True)

st.write("Enter the car details below to predict its final price.")

# **Image Upload**
st.subheader("Upload Car Image (Optional)")
uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    # Open the image using PIL
    image = Image.open(uploaded_image)
    st.image(image, caption='Uploaded Image', use_container_width=True)
    st.write("")
    st.write("Image uploaded successfully!")

# User Inputs (Modify based on expected features)
col1, col2 = st.sidebar.columns(2)  # Split the sidebar into two columns

with col1:
    body_type = st.selectbox("Body Type", ["Sedan", "SUV", "Hatchback", "Coupe", "Convertible", "Wagon", "Van"])
    horsepower = st.number_input("Horsepower (HP)", min_value=50, max_value=1000, step=5)
    city_fuel_economy = st.number_input("City Fuel Economy (km/L)", min_value=5, max_value=50, step=1)
    highway_fuel_economy = st.number_input("Highway Fuel Economy (km/L)", min_value=5, max_value=50, step=1)
    fuel_tank_volume = st.number_input("Fuel Tank Volume (Liters)", min_value=20, max_value=200, step=1)
    maximum_seating = st.selectbox("Maximum Seating", [2, 4, 5, 7, 8])
    make_name = st.selectbox("Make Name", ["Toyota", "Honda", "Ford", "BMW", "Audi", "Mercedes", "Chevrolet"])

with col2:
    fuel_type = st.selectbox("Fuel Type", ["Petrol", "Diesel", "Electric", "Hybrid"])
    length = st.number_input("Car Length (cm)", min_value=300, max_value=600, step=1)
    height = st.number_input("Car Height (cm)", min_value=100, max_value=250, step=1)
    engine_displacement = st.number_input("Engine Displacement (L)", min_value=0.5, max_value=8.0, step=0.1)
    mileage = st.number_input("Mileage (km)", min_value=0, max_value=500000, step=1000)
    engine_cylinders = st.selectbox("Engine Cylinders", [3, 4, 5, 6, 8])
    listing_color = st.selectbox("Listing Color", ["Black", "White", "Red", "Blue", "Silver", "Gray", "Green"])

# Categorical Inputs
# body_type = st.selectbox("Body Type", ["Sedan", "SUV", "Hatchback", "Coupe", "Convertible", "Wagon", "Van"])
# fuel_type = st.selectbox("Fuel Type", ["Petrol", "Diesel", "Electric", "Hybrid"])
# maximum_seating = st.selectbox("Maximum Seating", [2, 4, 5, 7, 8])
# engine_cylinders = st.selectbox("Engine Cylinders", [3, 4, 5, 6, 8])
# make_name = st.selectbox("Make Name", ["Toyota", "Honda", "Ford", "BMW", "Audi", "Mercedes", "Chevrolet"])
# listing_color = st.selectbox("Listing Color", ["Black", "White", "Red", "Blue", "Silver", "Gray", "Green"])

# Create user input DataFrame for numeric features
user_data = pd.DataFrame({
    "HorsePower": [horsepower],
    "City Fuel Economy": [city_fuel_economy],
    "Highway Fuel Economy": [highway_fuel_economy],
    "Fuel Tank Volume": [fuel_tank_volume],
    "Length": [length],
    "Height": [height],
    "Engine Displacement": [engine_displacement],
    "Mileage": [mileage]
})

# Create user input DataFrame for categorical features
categorical_data = pd.DataFrame({
    "BodyType": [body_type],
    "FuelType": [fuel_type],
    "MaximumSeating": [maximum_seating],
    "EngineCylinders": [engine_cylinders],
    "MakeName": [make_name],
    "ListingColor": [listing_color]
})

# Apply OneHotEncoding to categorical features
encoder = OneHotEncoder(drop='first', sparse_output=False)
encoded_categorical_data = encoder.fit_transform(categorical_data)

# Create DataFrame for encoded categorical data
encoded_df = pd.DataFrame(encoded_categorical_data, columns=encoder.get_feature_names_out(categorical_data.columns))

# Concatenate numeric and categorical features
user_data = pd.concat([user_data, encoded_df], axis=1)

# **Ensure feature order matches the model**
if expected_features is not None:
    # Align columns to match the model's expected features
    missing_columns = set(expected_features) - set(user_data.columns)
    if missing_columns:
        st.warning(f"Missing expected features: {missing_columns}")
        # Add missing columns with default values (e.g., 0)
        for col in missing_columns:
            user_data[col] = 0
    extra_columns = set(user_data.columns) - set(expected_features)
    if extra_columns:
        st.warning(f"Dropping extra columns: {extra_columns}")
        user_data = user_data.drop(columns=extra_columns)
    
    # Reorder the columns to match the model's expected feature order
    user_data = user_data[expected_features]

# Display entered data
st.subheader("Entered Car Details")
st.dataframe(user_data)

# Prediction
if st.button("Predict Car Price"):
    try:
        # Make prediction
        predicted_price = model.predict(user_data)
        st.success(f'Predicted Car Price: ₹{np.round(predicted_price[0], 2)}')
    except Exception as e:  
        st.error(f"Prediction Error: {str(e)}")

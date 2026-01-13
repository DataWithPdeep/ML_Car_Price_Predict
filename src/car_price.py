import streamlit as st
import pandas as pd
import pickle
import os
from PIL import Image

st.set_page_config(layout="wide")

# ===== LOAD MODEL =====
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "..", "Model", "linear_model.pkl")

with open(model_path, "rb") as f:
    model = pickle.load(f)

# ================= USER INPUT FUNCTION =================
def get_user_input():
    horsepower = st.sidebar.number_input('Horsepower (No)', 0, 1000, 300)
    torque = st.sidebar.number_input('Torque (No)', 0, 1500, 400)

    make = st.sidebar.selectbox(
        'Make',
        ['Aston Martin', 'Audi', 'BMW', 'Bentley', 'Ford', 'Mercedes-Benz', 'Nissan']
    )

    body_size = st.sidebar.selectbox(
        'Body Size',
        ['Compact', 'Large', 'Midsize']
    )

    body_style = st.sidebar.selectbox(
        'Body Style',
        [
            'Cargo Minivan', 'Cargo Van', 'Convertible', 'Convertible SUV',
            'Coupe', 'Hatchback', 'Passenger Minivan', 'Passenger Van',
            'Pickup Truck', 'SUV', 'Sedan', 'Wagon'
        ]
    )

    engine_aspiration = st.sidebar.selectbox(
        'Engine Aspiration',
        [
            'Electric Motor', 'Naturally Aspirated', 'Supercharged',
            'Turbocharged', 'Twin-Turbo', 'Twincharged'
        ]
    )

    drivetrain = st.sidebar.selectbox(
        'Drivetrain',
        ['4WD', 'AWD', 'FWD', 'RWD']
    )

    transmission = st.sidebar.selectbox(
        'Transmission',
        ['Automatic', 'Manual']
    )

    return {
        'Horsepower_No': horsepower,
        'Torque_No': torque,
        'Make': make,
        'Body Size': body_size,
        'Body Style': body_style,
        'Engine Aspiration': engine_aspiration,
        'Drivetrain': drivetrain,
        'Transmission': transmission
    }

# ================= MAIN IMAGE & TITLE =================
image_banner = Image.open(r'E:\ML_Pro\MLCar_price_Predict\Images\Pic 2.PNG')
st.image(image_banner, use_container_width=True)

st.markdown(
    "<h1 style='text-align: center;'>Vehicle Price Prediction App</h1>",
    unsafe_allow_html=True
)

left_col, right_col = st.columns(2)

with left_col:
    st.header("Feature Details")

user_data = get_user_input()
st.write(user_data)

with right_col:
    st.header("Predict Vehicle Price")

# ================= PREDICTION =================
if st.button("Predict"):
    #  DataFrame banao
    input_df = pd.DataFrame([user_data])

    #  One-hot encoding
    input_df = pd.get_dummies(input_df)

    #  TRAINING FEATURES SE MATCH KARO
    input_df = input_df.reindex(columns=feature_columns, fill_value=0)

    # Prediction
    prediction = model.predict(input_df)

    st.subheader("Predicted Price")
    st.success(f"$ {prediction[0]:,.2f}")

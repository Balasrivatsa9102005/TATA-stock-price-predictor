import streamlit as st
import pickle
import numpy as np
import os
from streamlit_option_menu import option_menu


st.set_page_config(page_title="Stock Price Predictor", layout="centered")



@st.cache_resource
def load_model(filename):
    model_path = os.path.join("models", filename)
    with open(model_path, "rb") as file:
        model = pickle.load(file)
    return model

models = {
    "Tata Motors": load_model("TATA_motors_xgb.pkl"),
    "Tata Steel": load_model("TATA_steel_xgb.pkl"),
    "TCS": load_model("TCS_xgb.pkl")
}


def predict(model, open_p, high_p, low_p, close_p):
    try:
        data = np.array([[open_p, high_p, low_p, close_p]])
        prediction = model.predict(data)
        return round(float(prediction[0]), 2)
    except Exception as e:
        return f"Prediction failed: {e}"

selected = option_menu(
    menu_title="Stock Navigator",  
    options=["Tata Motors", "Tata Steel", "TCS"],  
    icons=["car-front-fill", "building", "cpu"],  
    menu_icon="cast",  
    default_index=0,
    orientation="horizontal",
)


st.title("📈 Stock Close Price Predictor")
st.markdown(f"Predict the next day's **closing price** for **{selected}** using ML-powered models (XGBoost FTW).")


st.subheader(f"Enter {selected} stock details (Today)")

open_p = st.number_input("Open Price", min_value=0.0, format="%.2f")
high_p = st.number_input("High Price", min_value=0.0, format="%.2f")
low_p = st.number_input("Low Price", min_value=0.0, format="%.2f")
close_p = st.number_input("Current Close Price", min_value=0.0, format="%.2f")


if st.button("Predict Next Day's Close"):
    model = models[selected]
    result = predict(model, open_p, high_p, low_p, close_p)

    if isinstance(result, str) and "failed" in result:
        st.error(result)
    else:
        st.success(f"📊 Predicted Close for Tomorrow: ₹{result}")

st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray; font-size: 14px;'>
        Built with ❤️ by <strong>Bala Srivatsa</strong><br>
        <a href='https://github.com/Balasrivatsa9102005' target='_blank' style='text-decoration: none; color: #4078c0;'>GitHub</a> | 
        <a href='https://www.linkedin.com/in/bala-srivatsa-panigrahi-7856a9325/' target='_blank' style='text-decoration: none; color: #0e76a8;'>LinkedIn</a>
    </div>
    """,
    unsafe_allow_html=True
)


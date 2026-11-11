
import streamlit as st
import joblib
import pandas as pd

st.set_page_config(
    page_title="CNC Tool Wear Detection",
    page_icon="⚙️"
)

st.title("⚙️ CNC Tool Wear Detection")
st.write("Enter machine values to predict tool condition.")

# Load trained model
saved = joblib.load("cnc_tool_wear_model.pkl")
model = saved["model"]
features = saved["features"]

st.subheader("Enter Machine Values")

values = {}

for feature in features:
    values[feature] = st.number_input(
        feature,
        value=0.0,
        format="%.4f"
    )

if st.button("Predict Tool Condition"):
    input_data = pd.DataFrame([values])
    prediction = model.predict(input_data)[0]

    st.subheader("Prediction Result")

    if str(prediction).lower() == "worn":
        st.error("🔴 TOOL CONDITION: WORN")
    else:
        st.success("🟢 TOOL CONDITION: UNWORN")

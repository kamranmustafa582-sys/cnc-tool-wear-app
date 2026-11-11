
import streamlit as st
import pandas as pd
import joblib
import os

# ---------------- PAGE CONFIGURATION ----------------

st.set_page_config(
    page_title="CNC Tool Wear Detection",
    page_icon="⚙️",
    layout="wide"
)

# ---------------- CUSTOM DESIGN ----------------

st.markdown("""
<style>
.main {
    background-color: #f5f7fa;
}
.block-container {
    padding-top: 2rem;
}
[data-testid="stMetric"] {
    background-color: white;
    padding: 18px;
    border-radius: 12px;
    border: 1px solid #e0e5ec;
}
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL ----------------

MODEL_PATH = "cnc_tool_wear_model.pkl"

@st.cache_resource
def load_model():
    saved_data = joblib.load(MODEL_PATH)

    model = saved_data["model"]
    features = saved_data["features"]

    return model, features


# ---------------- HEADER ----------------

st.title("⚙️ CNC Machine Tool Wear Detection")

st.write(
    "An intelligent Machine Learning application "
    "to predict CNC machine tool condition."
)

st.divider()

# ---------------- MODEL CHECK ----------------

if not os.path.exists(MODEL_PATH):
    st.error(
        "Model file not found! "
        "Please place cnc_tool_wear_model.pkl "
        "in the same folder as app.py."
    )
    st.stop()

try:
    model, features = load_model()

except Exception as e:
    st.error(f"Unable to load model: {e}")
    st.stop()

st.success("Machine Learning model loaded successfully!")

# ---------------- INPUT SECTION ----------------

st.subheader("📊 Enter Machine Parameters")

st.write(
    "Enter all seven machine values below "
    "to predict the tool condition."
)

with st.form("prediction_form"):

    col1, col2 = st.columns(2)

    with col1:
        z1_command = st.number_input(
            "Z1 Command Position",
            value=0.0,
            format="%.6f"
        )

        z1_actual = st.number_input(
            "Z1 Actual Position",
            value=0.0,
            format="%.6f"
        )

        m1_sequence = st.number_input(
            "M1 Sequence Number",
            value=0.0,
            format="%.6f"
        )

        m1_feedrate = st.number_input(
            "M1 Current Feedrate",
            value=0.0,
            format="%.6f"
        )

    with col2:
        s1_command = st.number_input(
            "S1 Command Position",
            value=0.0,
            format="%.6f"
        )

        s1_actual = st.number_input(
            "S1 Actual Position",
            value=0.0,
            format="%.6f"
        )

        s1_voltage = st.number_input(
            "S1 DC Bus Voltage",
            value=0.0,
            format="%.6f"
        )

    predict_button = st.form_submit_button(
        "🔍 Predict Tool Condition",
        use_container_width=True
    )

# ---------------- PREDICTION ----------------

if predict_button:

    input_values = {
        "Z1_CommandPosition": z1_command,
        "Z1_ActualPosition": z1_actual,
        "M1_sequence_number": m1_sequence,
        "M1_CURRENT_FEEDRATE": m1_feedrate,
        "S1_CommandPosition": s1_command,
        "S1_ActualPosition": s1_actual,
        "S1_DCBusVoltage": s1_voltage
    }

    # Preserve the exact feature order used in training
    input_df = pd.DataFrame(
        [[input_values[feature] for feature in features]],
        columns=features
    )

    try:
        prediction = model.predict(input_df)[0]

        probabilities = model.predict_proba(input_df)[0]

        classes = list(model.classes_)

        worn_index = classes.index("worn")
        unworn_index = classes.index("unworn")

        worn_probability = probabilities[worn_index] * 100
        unworn_probability = probabilities[unworn_index] * 100

        st.divider()
        st.subheader("📈 Prediction Results")

        if prediction == "worn":

            st.error("⚠️ PREDICTED CONDITION: WORN")

            st.warning(
                "The model predicts that the cutting tool "
                "may be worn. Inspect the tool before "
                "continuing production."
            )

        else:

            st.success("✅ PREDICTED CONDITION: UNWORN")

            st.info(
                "The model predicts that the cutting tool "
                "is in an unworn condition."
            )

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Unworn Class Probability",
                f"{unworn_probability:.2f}%"
            )

        with col2:
            st.metric(
                "Worn Class Probability",
                f"{worn_probability:.2f}%"
            )

        st.subheader("Probability Distribution")

        probability_data = pd.DataFrame({
            "Condition": ["Unworn", "Worn"],
            "Probability (%)": [
                unworn_probability,
                worn_probability
            ]
        })

        st.bar_chart(
            probability_data.set_index("Condition")
        )

        with st.expander("View Entered Machine Values"):
            st.dataframe(
                input_df,
                use_container_width=True
            )

        st.caption(
            "Probabilities are model estimates, not "
            "guaranteed measures of actual tool condition."
        )

    except Exception as e:
        st.error(f"Prediction error: {e}")

# ---------------- FOOTER ----------------

st.divider()

st.caption(
    "CNC Tool Wear Detection | Random Forest Machine Learning Model"
)

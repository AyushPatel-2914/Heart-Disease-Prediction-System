import streamlit as st
import numpy as np
import joblib

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Heart Disease Prediction System",
    page_icon="❤️",
    layout="wide"
)

# --------------------------------------------------
# Load Model & Scaler
# --------------------------------------------------
model = joblib.load("heart_disease_model.pkl")
scaler = joblib.load("scaler.pkl")

# --------------------------------------------------
# Custom CSS (FINAL THEME)
# --------------------------------------------------
st.markdown("""
<style>

/* ===== Main background ===== */
.main {
    background-color: #0b1d3a;  /* dark blue */
}

/* ===== Global text color ===== */
html, body, [class*="css"] {
    color: #9cff9c !important; /* light green */
}

/* ===== Sidebar ===== */
section[data-testid="stSidebar"] {
    background-color: #800000; /* maroon */
}

/* Sidebar text */
section[data-testid="stSidebar"] * {
    color: #ffffff !important;
}

/* ===== Headings ===== */
h1, h2, h3, h4 {
    color: #9cff9c !important;
}

/* ===== Cards ===== */
.card {
    background-color: #112a52;
    padding: 20px;
    border-radius: 12px;
    box-shadow: 0px 0px 15px rgba(0,0,0,0.4);
}

/* ===== Buttons ===== */
.stButton > button {
    background-color: #9cff9c;
    color: #0b1d3a;
    border-radius: 10px;
    font-weight: bold;
    padding: 10px 20px;
}

.stButton > button:hover {
    background-color: #7de87d;
    color: black;
}

/* ===== Inputs ===== */
input, select, textarea {
    background-color: #112a52 !important;
    color: #9cff9c !important;
}

/* ===== Footer ===== */
footer {
    color: #9cff9c !important;
}

</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# Header
# --------------------------------------------------
st.title("❤️ Heart Disease Prediction System")
st.write(
    "A **machine learning–based clinical decision support system** "
    "for predicting the likelihood of heart disease."
)

st.markdown("---")

# --------------------------------------------------
# Sidebar Inputs
# --------------------------------------------------
st.sidebar.header("🧑 Patient Information")

age = st.sidebar.slider("Age", 1, 120, 45)

sex_label = st.sidebar.selectbox("Sex", ["Female", "Male"])
sex = 1 if sex_label == "Male" else 0

cp_label = st.sidebar.selectbox(
    "Chest Pain Type",
    ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"]
)
cp_mapping = {
    "Typical Angina": 0,
    "Atypical Angina": 1,
    "Non-anginal Pain": 2,
    "Asymptomatic": 3
}
cp = cp_mapping[cp_label]

st.sidebar.header("🩺 Clinical Measurements")

trestbps = st.sidebar.slider("Resting Blood Pressure (mm Hg)", 80, 200, 120)
chol = st.sidebar.slider("Serum Cholesterol (mg/dl)", 100, 600, 200)

fbs_label = st.sidebar.selectbox("Fasting Blood Sugar > 120 mg/dl", ["No", "Yes"])
fbs = 1 if fbs_label == "Yes" else 0

restecg_label = st.sidebar.selectbox(
    "Resting ECG Result",
    ["Normal", "ST-T Abnormality", "Left Ventricular Hypertrophy"]
)
restecg_mapping = {
    "Normal": 0,
    "ST-T Abnormality": 1,
    "Left Ventricular Hypertrophy": 2
}
restecg = restecg_mapping[restecg_label]

thalach = st.sidebar.slider("Maximum Heart Rate Achieved", 60, 220, 150)

exang_label = st.sidebar.selectbox("Exercise Induced Angina", ["No", "Yes"])
exang = 1 if exang_label == "Yes" else 0

oldpeak = st.sidebar.slider("ST Depression", 0.0, 6.0, 1.0)

slope_label = st.sidebar.selectbox(
    "Slope of ST Segment",
    ["Upsloping", "Flat", "Downsloping"]
)
slope_mapping = {
    "Upsloping": 0,
    "Flat": 1,
    "Downsloping": 2
}
slope = slope_mapping[slope_label]

ca = st.sidebar.selectbox("Number of Major Vessels", [0, 1, 2, 3])

thal_label = st.sidebar.selectbox(
    "Thalassemia",
    ["Normal", "Fixed Defect", "Reversible Defect", "Unknown"]
)
thal_mapping = {
    "Normal": 0,
    "Fixed Defect": 1,
    "Reversible Defect": 2,
    "Unknown": 3
}
thal = thal_mapping[thal_label]

# --------------------------------------------------
# Prediction Section
# --------------------------------------------------
st.markdown("### 🔍 Prediction Result")

if st.button("🔍 Predict Heart Disease"):
    input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
                             thalach, exang, oldpeak, slope, ca, thal]])

    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    st.markdown("<div class='card'>", unsafe_allow_html=True)

    if prediction == 1:
        st.error("⚠️ **High Risk of Heart Disease**")
        st.write(f"**Predicted Probability:** `{probability:.2%}`")
        st.write(
            "The patient shows a **significant likelihood of heart disease**. "
            "Further medical evaluation is recommended."
        )
    else:
        st.success("✅ **Low Risk of Heart Disease**")
        st.write(f"**Predicted Probability:** `{probability:.2%}`")
        st.write(
            "The patient shows a **low likelihood of heart disease**. "
            "Maintain a healthy lifestyle and regular check-ups."
        )

    st.markdown("</div>", unsafe_allow_html=True)

# --------------------------------------------------
# Footer
# --------------------------------------------------
st.markdown("---")
st.caption(
    "⚕️ **Educational Use Only** — This application does not replace professional medical advice."
)

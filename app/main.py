import streamlit as st
import pickle
import joblib
import numpy as np
import pandas as pd

# Load model and selected features
model = pickle.load(open('models/heart_attack_model.pkl', 'rb'))
selected_features = joblib.load(open('models/selected_features.pkl', 'rb'))

st.title("❤️ Heart Attack Prediction App")

st.markdown("Provide the following medical information to check the risk of a heart attack.")

# Input form with correct column names
age = st.slider("Age", 20, 80, 45)
sex = st.selectbox("Sex (1 = Male, 0 = Female)", [0, 1])
cp = st.selectbox("Chest Pain Type (cp)", [0, 1, 2, 3])
trtbps = st.number_input("Resting Blood Pressure (trtbps)", 80, 200, 120)
chol = st.number_input("Serum Cholesterol (chol)", 100, 400, 240)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (fbs)", [0, 1])
restecg = st.selectbox("Resting ECG (restecg)", [0, 1, 2])
thalachh = st.number_input("Max Heart Rate Achieved (thalachh)", 60, 220, 150)
exng = st.selectbox("Exercise Induced Angina (exng)", [0, 1])
oldpeak = st.slider("ST Depression (oldpeak)", 0.0, 6.0, 1.0)
slp = st.selectbox("Slope of ST Segment (slp)", [0, 1, 2])
caa = st.selectbox("Major Vessels Colored (caa)", [0, 1, 2, 3])
thall = st.selectbox("Thalassemia (thall)", [0, 1, 2])

# Collect input in a dictionary
full_input = {
    'age': age,
    'sex': sex,
    'cp': cp,
    'trtbps': trtbps,
    'chol': chol,
    'fbs': fbs,
    'restecg': restecg,
    'thalachh': thalachh,
    'exng': exng,
    'oldpeak': oldpeak,
    'slp': slp,
    'caa': caa,
    'thall': thall
}

# Convert to DataFrame
input_df = pd.DataFrame([full_input])

# Prediction
if st.button("Predict"):
    try:
        # Filter to only selected features
        input_selected = input_df[selected_features]

        # Predict
        prediction = model.predict(input_selected)
        probabilities = model.predict_proba(input_selected)[0]

        st.subheader("🔍 Prediction Results")
        st.write("✅ Probability of No Risk:", round(probabilities[0] * 100, 2), "%")
        st.write("⚠️ Probability of Heart Attack Risk:", round(probabilities[1] * 100, 2), "%")

        if prediction[0] == 1:
            st.error("🚨 High Risk of Heart Attack Detected!")
        else:
            st.success("💚 No Risk of Heart Attack Detected.")

        # Optional: Show input used
        with st.expander("See input features used for prediction"):
            st.dataframe(input_selected)

    except KeyError as e:
        st.error(f"Missing feature in input: {e}")
    except Exception as e:
        st.error(f"Prediction failed: {e}")

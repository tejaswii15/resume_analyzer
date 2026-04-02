import streamlit as st
import joblib
import numpy as np
model = joblib.load("job_role_model.pkl")
label_encoders = joblib.load("label_encoders.pkl")
st.title("Resume Job Role Predictor")
st.write("Enter candidate details to predict suitable Job Role")
years_exp = st.number_input("Years of Experience", min_value=0, max_value=50, step=1)
skills = st.selectbox(
    "Skills",
    label_encoders["Skills"].classes_
)
education = st.selectbox(
    "Education",
    label_encoders["Education"].classes_
)
skills_encoded = label_encoders["Skills"].transform([skills])[0]
education_encoded = label_encoders["Education"].transform([education])[0]
input_data = np.array([[years_exp, skills_encoded, education_encoded]])
if st.button("Predict Job Role"):
    prediction = model.predict(input_data)[0]
    job_role = label_encoders["JobRole"].inverse_transform([prediction])[0]
    st.success(f" Predicted Job Role: {job_role}")
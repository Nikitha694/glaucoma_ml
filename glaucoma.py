import streamlit as st
import pandas as pd
import pickle

# ----------------------------
# Load the trained model
# ----------------------------
try:
    with open("glaucomapickle.pkl", "rb") as f:
        model = pickle.load(f)
except FileNotFoundError:
    st.error("❌ Model file 'glaucomapickle.pkl' not found. Please upload it to the same folder.")
    st.stop()

# ----------------------------
# Streamlit App UI
# ----------------------------
st.title("👁️ Glaucoma Prediction App")

st.sidebar.header("Enter Patient Details")

# Input fields
age = st.sidebar.slider("Age", 20, 90, 50)
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
visual_acuity = st.sidebar.number_input("Visual Acuity (0-1)", min_value=0.0, max_value=1.0, step=0.01)
iop = st.sidebar.number_input("Intraocular Pressure (IOP)", min_value=5.0, max_value=50.0, step=0.1)
cdr = st.sidebar.number_input("Cup-to-Disc Ratio (CDR)", min_value=0.1, max_value=1.0, step=0.01)
family_history = st.sidebar.selectbox("Family History of Glaucoma", ["No", "Yes"])
medical_history = st.sidebar.text_input("Medical History (e.g. Diabetes, Hypertension)")
medication = st.sidebar.text_input("Medication Usage")
visual_field = st.sidebar.number_input("Visual Field Test Result (score)", min_value=0.0, max_value=100.0, step=0.1)
oct = st.sidebar.number_input("OCT Result", min_value=0.0, max_value=200.0, step=0.1)
pachymetry = st.sidebar.number_input("Pachymetry (µm)", min_value=400.0, max_value=700.0, step=0.1)
cataract = st.sidebar.selectbox("Cataract Status", ["No", "Yes"])
angle_closure = st.sidebar.selectbox("Angle Closure Status", ["No", "Yes"])
visual_symptoms = st.sidebar.text_input("Visual Symptoms")

# ----------------------------
# Prepare input DataFrame
# ----------------------------
input_df = pd.DataFrame([{
    "Age": age,
    "Gender": gender,
    "Visual Acuity Measurements": visual_acuity,
    "Intraocular Pressure (IOP)": iop,
    "Cup-to-Disc Ratio (CDR)": cdr,
    "Family History": family_history,
    "Medical History": medical_history,
    "Medication Usage": medication,
    "Visual Field Test Results": visual_field,
    "Optical Coherence Tomography (OCT) Results": oct,
    "Pachymetry": pachymetry,
    "Cataract Status": cataract,
    "Angle Closure Status": angle_closure,
    "Visual Symptoms": visual_symptoms
}])

# ----------------------------
# Prediction
# ----------------------------
if st.sidebar.button("Predict"):
    try:
        prediction = model.predict(input_df)[0]
        st.subheader("🔮 Prediction Result")
        st.write(f"**Predicted Diagnosis:** {prediction}")
    except Exception as e:
        st.error(f"⚠️ Error during prediction: {e}")

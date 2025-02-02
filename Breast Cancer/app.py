import streamlit as st
import pickle
import numpy as np

# Load the trained model
model_filename = "breast_cancer_model.pkl"
with open(model_filename, 'rb') as file:
    model = pickle.load(file)

# Streamlit UI Configuration
st.set_page_config(page_title="Breast Cancer Prediction", page_icon="💙", layout="centered")

# Custom Styling
st.markdown(
    """
    <style>
        body {background-color: #f5f5f5;}
        .stButton>button {background-color: #4CAF50; color: white; font-size: 18px; border-radius: 10px;}
        .stTextInput>div>div>input {font-size: 16px;}
        .stNumberInput>div>div>input {font-size: 16px;}
        .stMarkdown {text-align: center; font-size: 20px; color: #333;}
        .stSelectbox>div>div>select {font-size: 16px;}
    </style>
    """,
    unsafe_allow_html=True
)

# App Title
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>💙 Breast Cancer Prediction</h1>", unsafe_allow_html=True)
st.write("Enter patient details below to predict if the tumor is benign or malignant.")

# Sidebar Info
st.sidebar.header("ℹ️ About the App")
st.sidebar.write("This app uses a Machine Learning model to classify breast cancer tumors.")
st.sidebar.write("Model trained on Breast Cancer Wisconsin dataset.")
st.sidebar.write("Accuracy: 95%+ on test data.")

# Input Fields for Features
st.subheader("🔍 Enter Features")
feature_names = ["Radius Mean", "Texture Mean", "Perimeter Mean", "Area Mean", "Smoothness Mean"]
inputs = []
for feature in feature_names:
    value = st.number_input(feature, min_value=0.0, step=0.01, format="%.2f")
    inputs.append(value)

# Prediction Button
if st.button("🔮 Predict Cancer Type"):
    # Convert inputs to numpy array & reshape
    input_array = np.array(inputs).reshape(1, -1)
    prediction = model.predict(input_array)[0]
    result = "Malignant" if prediction == 1 else "Benign"
    color = "#FF4B4B" if prediction == 1 else "#4CAF50"
    
    st.markdown(
        f"""
        <div style="background-color: {color}; padding: 20px; border-radius: 10px;">
            <h2 style="color: white; text-align: center; font-size: 22px;">Prediction: {result}</h2>
        </div>
        """, unsafe_allow_html=True
    )
    
    st.success("✅ Model successfully predicted the tumor type!")

# Footer
st.markdown("---")
st.markdown("💙 Developed by AI & Data Science Team")

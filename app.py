import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
import shap
import matplotlib.pyplot as plt

st.set_page_config(page_title="Student Score Predictor", layout="wide")
st.title("Student Score Predictor 🧠")

# Step 1: Upload CSV
uploaded_file = st.file_uploader("Upload CSV file of student data", type=["csv"])
if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.write("Sample data:")
    st.dataframe(data.head())
    
    # Step 2: Prepare features and train simple model
    if "final_score" not in data.columns:
        st.warning("CSV must include 'final_score' column for training.")
    else:
        X = data.drop("final_score", axis=1)
        y = data["final_score"]
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        model = MLPRegressor(hidden_layer_sizes=(64,32), activation='relu', max_iter=500, random_state=42)
        model.fit(X_scaled, y)
        
        predictions = model.predict(X_scaled)
        data['predicted_score'] = predictions
        st.success("Predictions complete!")
        st.dataframe(data)
        
        # Step 3: Feature importance with SHAP
        st.subheader("Feature Importance (SHAP)")
        explainer = shap.Explainer(model, X_scaled)
        shap_values = explainer(X_scaled)
        fig, ax = plt.subplots()
        shap.summary_plot(shap_values, X_scaled, feature_names=X.columns, show=False)
        st.pyplot(fig)
        
        # Step 4: Download predictions
        csv = data.to_csv(index=False).encode()
        st.download_button(
            label="Download predictions as CSV",
            data=csv,
            file_name='predicted_scores.csv',
            mime='text/csv',
        )

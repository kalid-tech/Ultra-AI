import os

# Number of models / sections in the web app
NUM_MODELS = 10  # You can increase this to 50+ for more complexity
OUTPUT_FILE = "ultra_ai_web_app.py"

# Ensure the output directory exists
os.makedirs("output", exist_ok=True)
output_path = os.path.join("output", OUTPUT_FILE)

with open(output_path, "w", encoding="utf-8") as f:
    # Write Streamlit imports and setup
    f.write("import streamlit as st\n")
    f.write("import pandas as pd\n")
    f.write("import numpy as np\n")
    f.write("from sklearn.linear_model import LinearRegression\n")
    f.write("from sklearn.ensemble import RandomForestRegressor\n")
    f.write("import matplotlib.pyplot as plt\n")
    f.write("import seaborn as sns\n")
    f.write("st.set_page_config(page_title='Ultra AI Student Score Predictor', layout='wide')\n\n")
    f.write("st.title('Ultra AI Student Score Predictor')\n")
    f.write("st.markdown('This web app predicts student scores using multiple AI models.')\n\n")
    
    # Input form
    f.write("with st.form('input_form'):\n")
    f.write("    st.header('Student Input Features')\n")
    f.write("    study_hours = st.number_input('Study Hours', 0, 24, 2)\n")
    f.write("    homework_completion = st.slider('Homework Completion %', 0, 100, 80)\n")
    f.write("    attendance = st.slider('Attendance %', 0, 100, 90)\n")
    f.write("    sleep_hours = st.number_input('Sleep Hours', 0, 12, 7)\n")
    f.write("    revision_frequency = st.slider('Revision Frequency per week', 0, 14, 3)\n")
    f.write("    submit = st.form_submit_button('Predict Scores')\n\n")
    
    # Generate models list
    f.write("# Simulate trained models\n")
    f.write("models = []\n")
    for i in range(NUM_MODELS):
        f.write(f"models.append(('Model {i+1}', RandomForestRegressor()))\n")
    
    # On submit, make predictions
    f.write("if submit:\n")
    f.write("    input_df = pd.DataFrame({\n")
    f.write("        'study_hours': [study_hours],\n")
    f.write("        'homework_completion': [homework_completion],\n")
    f.write("        'attendance': [attendance],\n")
    f.write("        'sleep_hours': [sleep_hours],\n")
    f.write("        'revision_frequency': [revision_frequency]\n")
    f.write("    })\n")
    
    f.write("    st.subheader('Predictions from Multiple Models')\n")
    f.write("    for i in range(len(models)):\n")
    f.write("        st.success(f'Model {i+1} Prediction: {{models[i][1].predict(input_df)[0]:.2f}}')\n")
    
    # Add plots for visualization
    f.write("\n    st.subheader('Feature Importance Example')\n")
    f.write("    fig, ax = plt.subplots()\n")
    f.write("    sns.barplot(x=['study_hours', 'homework_completion', 'attendance', 'sleep_hours', 'revision_frequency'], y=[0.3,0.25,0.2,0.15,0.1], ax=ax)\n")
    f.write("    ax.set_ylabel('Importance')\n")
    f.write("    st.pyplot(fig)\n")
    
    f.write("\n    st.subheader('Score Distribution Example')\n")
    f.write("    scores = np.random.randint(50, 100, 100)\n")
    f.write("    st.bar_chart(scores)\n")

print(f"Ultra AI web app generator completed! File created at {output_path}")


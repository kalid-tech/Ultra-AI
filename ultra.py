
import streamlit as st
import pandas as pd
import numpy as np
import json, hashlib, os
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="Ultra AI Portal", layout="wide", page_icon="🎓")

# ---------------------- File Setup ----------------------
for file in ["users.json", "attendance.json", "feedback.json", "scores.json"]:
    if not os.path.exists(file):
        with open(file, "w") as f: json.dump({}, f)

# Load JSON files
with open("users.json","r") as f: users=json.load(f)
with open("attendance.json","r") as f: attendance=json.load(f)
with open("feedback.json","r") as f: feedback=json.load(f)
with open("scores.json","r") as f: scores=json.load(f)

# ---------------------- Session State ----------------------
if "logged_in" not in st.session_state: st.session_state["logged_in"]=False
if "user" not in st.session_state: st.session_state["user"]=None
if "notifications" not in st.session_state: st.session_state["notifications"]=[]

# ---------------------- Helper Functions ----------------------
def hash_password(pw): return hashlib.sha256(pw.encode()).hexdigest()

def register(username,password,role):
    if username in users: st.error("User already exists!")
    else:
        users[username] = {"password_hash":hash_password(password), "role":role}
        with open("users.json","w") as f: json.dump(users,f)
        st.success("Registered successfully!")

def login(username,password):
    if username in users and users[username]["password_hash"]==hash_password(password):
        st.session_state["logged_in"]=True
        st.session_state["user"]=username
        st.success(f"Welcome {username}!")
    else: st.error("Invalid credentials")

# ---------------------- Authentication ----------------------
if not st.session_state["logged_in"]:
    choice = st.sidebar.selectbox("Login/Register", ["Login","Register"])
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    role = st.selectbox("Role", ["Student","Family","Admin"])
    
    if choice=="Register" and st.button("Register"): register(username,password,role)
    if choice=="Login" and st.button("Login"): login(username,password)
else:
    user = st.session_state["user"]
    role = users[user]["role"]
    st.sidebar.write(f"Logged in as {user} ({role})")
    if st.sidebar.button("Logout"):
        st.session_state["logged_in"]=False
        st.experimental_rerun()

    # ---------------------- Admin Dashboard ----------------------
    if role=="Admin":
        st.title("Ultra AI - Admin Dashboard")
        tab1, tab2, tab3, tab4 = st.tabs(["Students","Attendance","Scores","Feedbacks"])
        
        with tab1:
            st.subheader("Manage Students")
            student_name = st.text_input("Add Student")
            if st.button("Add Student"):
                if student_name not in users:
                    users[student_name] = {"password_hash":hash_password("student123"), "role":"Student"}
                    with open("users.json","w") as f: json.dump(users,f)
                    st.success(f"{student_name} added with default password student123")

            # Student Ranking
            if scores:
                df_scores = pd.DataFrame({s: scores[s] for s in scores})
                avg_scores = df_scores.mean().sort_values(ascending=False)
                st.subheader("Student Ranking")
                st.dataframe(avg_scores.reset_index().rename(columns={"index":"Student",0:"Average Score"}))

        with tab2:
            st.subheader("Attendance Records")
            st.write(attendance)
            fig = px.bar(pd.DataFrame({stu: len(attendance[stu]) for stu in attendance}, index=[0]).T, labels={'index':'Student',0:'Days Present'})
            st.plotly_chart(fig, use_container_width=True)

        with tab3:
            st.subheader("Predict Scores & Performance")
            selected_student = st.selectbox("Select Student", [s for s in users if users[s]["role"]=="Student"])
            hours = st.number_input("Hours Studied", 0, 24)
            quizzes = st.number_input("Quizzes Taken", 0, 10)
            if st.button("Predict"):
                X = np.array([[hours, quizzes]])
                model = LinearRegression()
                model.coef_ = np.array([5, 2])
                model.intercept_ = 50
                predicted = model.predict(X)[0]
                if selected_student not in scores: scores[selected_student]=[]
                scores[selected_student].append(float(predicted))
                with open("scores.json","w") as f: json.dump(scores,f)
                st.success(f"{selected_student} predicted score: {predicted:.2f}")
                st.session_state["notifications"].append(f"New score predicted for {selected_student}")

            # Animated charts
            if scores:
                df_scores = pd.DataFrame({s: scores[s] for s in scores})
                st.subheader("Scores Over Time")
                st.line_chart(df_scores)
                st.subheader("Average Scores")
                st.bar_chart(df_scores.mean().to_frame("Average Score"))

        with tab4:
            st.subheader("Feedbacks")
            st.write(feedback)

        # Export Reports
        st.subheader("Export Reports")
        if st.button("Export All Data to Excel"):
            with pd.ExcelWriter("UltraAI_Reports.xlsx") as writer:
                pd.DataFrame({s: scores[s] for s in scores}).to_excel(writer, sheet_name="Scores")
                pd.DataFrame({s: attendance[s] for s in attendance}).to_excel(writer, sheet_name="Attendance")
                pd.DataFrame({s: feedback[s] for s in feedback}).to_excel(writer, sheet_name="Feedback")
            st.success("Exported to UltraAI_Reports.xlsx")

    # ---------------------- Student Dashboard ----------------------
    if role=="Student":
        st.title("Ultra AI - Student Portal")
        st.subheader("Mark Attendance")
        if st.button("Mark Present"):
            today = str(datetime.today().date())
            if user not in attendance: attendance[user]=[]
            attendance[user].append(today)
            with open("attendance.json","w") as f: json.dump(attendance,f)
            st.success("Attendance marked!")
            st.session_state["notifications"].append(f"{user} marked attendance today")

        st.subheader("Submit Feedback")
        fb = st.text_area("Your Feedback")
        if st.button("Submit Feedback"):
            if user not in feedback: feedback[user]=[]
            feedback[user].append(fb)
            with open("feedback.json","w") as f: json.dump(feedback,f)
            st.success("Feedback submitted!")
            st.session_state["notifications"].append(f"{user} submitted feedback")

        st.subheader("Your Scores")
        if user in scores: st.line_chart(pd.DataFrame(scores[user], columns=["Score"]))
        else: st.write("No scores yet!")

    # ---------------------- Family Dashboard ----------------------
    if role=="Family":
        st.title("Ultra AI - Family Portal")
        st.subheader("Track Students")
        for stu in users:
            if users[stu]["role"]=="Student":
                st.write(f"**{stu}**")
                st.write(f"Attendance: {len(attendance.get(stu,[]))} days")
                st.write(f"Feedback: {feedback.get(stu,[])}")
                st.write(f"Scores: {scores.get(stu,[])}")

    # ---------------------- Live Notifications ----------------------
    if st.session_state["notifications"]:
        st.sidebar.subheader("Notifications")
        for note in st.session_state["notifications"]:
            st.sidebar.success(note)
        st.session_state["notifications"]=[]

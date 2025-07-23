import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Employee Salary Predictor", layout="centered")
st.title("Exact Employee Salary Prediction")

@st.cache_data
def load_data():
    df = pd.read_csv("employee_salary_data.csv")

    # Encode categorical features
    le_edu = LabelEncoder()
    le_dept = LabelEncoder()
    df["education_level"] = le_edu.fit_transform(df["education_level"])
    df["department"] = le_dept.fit_transform(df["department"])

    return df, le_edu, le_dept

@st.cache_resource
def train_model(df):
    X = df.drop("salary", axis=1)
    y = df["salary"]
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

df, le_edu, le_dept = load_data()
model = train_model(df)
feature_names = df.drop("salary", axis=1).columns.tolist()

st.header("📋 Enter Employee Details")
age = st.slider("Age", 18, 65, 30)
experience = st.slider("Years of Experience", 0, 40, 5)
education = st.selectbox("Education Level", le_edu.classes_)
department = st.selectbox("Department", le_dept.classes_)

# Prepare input
input_data = pd.DataFrame([[
    age,
    experience,
    le_edu.transform([education])[0],
    le_dept.transform([department])[0]
]], columns=feature_names)

if st.button(" Predict Salary"):
    prediction = model.predict(input_data)[0]
    st.success(f" Predicted Salary: ₹{prediction:,.2f}")
